"""Where does the classifier go wrong? Error analysis on TRAINING out-of-fold predictions only.

Error categories (predicted arc vs reference arc):
  reverse twin  - same road, opposite direction (two-way street)
  adjacent      - predicted arc is the successor/predecessor of the reference arc
  shares node   - other arc meeting the reference arc at a node
  label far     - reference arc is > 30 m from the GPS fix (label/reference problem)
  other         - parallel roads, 2+ hops away
  no candidates - reference arc not inside the 50 m candidate set

Also checks, for reverse-twin errors, whether the GPS actually moves along or against the
reference arc direction over a +-10 s window.

Run from the repo root:  python experiments/error_analysis.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.candidates import FEATURES_V1  # noqa: E402
from pathsnap.dataset import build_split, excluded_train_trips, stack  # noqa: E402
from pathsnap.hmm import decode_split  # noqa: E402
from pathsnap.metrics import top1_from_proba  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"


def categorise(trips, meta, pred):
    x = meta.merge(pred, on=["trip", "pt"], how="left")
    x["ok"] = x["arc"] == x["true_arc"]
    cat = []
    for r in x.itertuples():
        if r.ok:
            cat.append("correct"); continue
        if np.isnan(r.arc):
            cat.append("no candidates"); continue
        arcs = trips[r.trip].arcs
        p, t = arcs[int(r.arc)], arcs[int(r.true_arc)]
        if p[0] == t[1] and p[1] == t[0]:
            cat.append("reverse twin")
        elif p[0] == t[1] or p[1] == t[0]:
            cat.append("adjacent")
        elif set(p) & set(t):
            cat.append("shares node")
        elif r.true_d > 30:
            cat.append("label far")
        else:
            cat.append("other")
    x["cat"] = cat
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trips = build_split("train")
    bad = excluded_train_trips()
    c, meta = stack(trips, [t for t in trips if t not in bad])
    oof = np.zeros(len(c))
    for a, b in GroupKFold(5).split(c, c["y"], c["trip"]):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        oof[b] = m.fit(c.iloc[a][FEATURES_V1].values, c["y"].values[a]).predict_proba(
            c.iloc[b][FEATURES_V1].values)[:, 1]

    before = categorise(trips, meta, top1_from_proba(c, oof))
    after = categorise(trips, meta, decode_split(trips, c, oof, dict(beta=5, w_emit=1.0))[0])
    tab = pd.DataFrame({"classifier_only": before["cat"].value_counts(normalize=True),
                        "classifier_plus_hmm": after["cat"].value_counts(normalize=True)}).fillna(0)
    tab.to_csv(OUT / "error_categories_train_oof.csv")
    print(tab.round(4))

    # speed profile of reverse-twin errors (classifier only)
    speed = c.groupby(["trip", "pt"])["speed"].first().rename("speed").reset_index()
    b2 = before.merge(speed, on=["trip", "pt"], how="left")
    bins = pd.cut(b2["speed"], [-1, 0.5, 1, 2, 4, 8, 100], labels=["<0.5", "0.5-1", "1-2", "2-4", "4-8", ">8"])
    by_speed = b2.assign(twin=b2["cat"] == "reverse twin").groupby(bins, observed=False)["twin"].agg(["mean", "size"])
    by_speed.columns = ["reverse_twin_error_rate", "observations"]
    by_speed.to_csv(OUT / "twin_errors_by_speed.csv")
    print(by_speed.round(3))

    # does the GPS move along or against the reference arc? (after HMM)
    rows = []
    for r in after.itertuples():
        T = trips[r.trip]
        a = T.arcs[int(r.true_arc)]
        u = T.nxy[a[1]] - T.nxy[a[0]]
        if np.linalg.norm(u) == 0:
            continue
        i0, i1 = max(0, r.pt - 10), min(len(T.pxy) - 1, r.pt + 10)
        mv = T.pxy[i1] - T.pxy[i0]
        rows.append(dict(trip=r.trip, cat=r.cat, along=float(mv @ u / np.linalg.norm(u)),
                         moved=float(np.linalg.norm(mv))))
    d = pd.DataFrame(rows)
    moving = d[d.moved > 15]
    chk = moving.groupby(moving["cat"] == "reverse twin").apply(
        lambda g: pd.Series(dict(observations=len(g), share_gps_against_reference=(g.along < -5).mean())),
        include_groups=False)
    chk.index = ["other observations", "reverse-twin errors"]
    chk.to_csv(OUT / "reference_direction_check.csv")
    print(chk.round(3))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    tab.drop(index="correct").sort_values("classifier_only").plot.barh(ax=ax[0], color=["#ff9800", "#1565c0"])
    ax[0].set_title("Error share of all observations (training OOF)"); ax[0].set_xlabel("share of observations")
    by_speed["reverse_twin_error_rate"].plot.bar(ax=ax[1], rot=0, color="#d32f2f")
    ax[1].set_title("Reverse-twin error rate by speed (classifier only)"); ax[1].set_xlabel("speed (m/s)")
    plt.tight_layout(); plt.savefig(OUT / "error_analysis.png", dpi=120)


if __name__ == "__main__":
    main()

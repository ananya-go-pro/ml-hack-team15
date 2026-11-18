"""Incremental accuracy ladder: every change we tried, measured the same way.

For each step:
  * decision metric  = Top-1 on out-of-fold predictions over the 78 usable training trips
                       (5-fold GroupKFold by trip). Decoder parameters, when used, are also
                       picked on these OOF predictions.
  * reported metric  = Top-1 / route continuity on the 10 student-test trips (never used to
                       decide anything).
The evaluator partition is never read.

Run from the repo root (after hpo/bayes_opt_lightgbm.py):  python experiments/accuracy_ladder.py
(--plot-only redraws the figure from outputs/accuracy_ladder.csv)
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.candidates import FEATURES_V1, FEATURES_V2  # noqa: E402
from pathsnap.dataset import build_split, excluded_train_trips, stack  # noqa: E402
from pathsnap.hmm import decode_split  # noqa: E402
from pathsnap.metrics import sequence_metrics, top1_from_proba  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
SEED = 42
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8)
BETAS = [2, 3, 5, 7, 10, 20]


def lgbm(params):
    return lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)


def oof_proba(c, F, params):
    oof = np.zeros(len(c))
    for a, b in GroupKFold(5).split(c, c["y"], c["trip"]):
        oof[b] = lgbm(params).fit(c.iloc[a][F].values, c["y"].values[a]).predict_proba(c.iloc[b][F].values)[:, 1]
    return oof


def plot(df):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(df)); w = .38
    ax.bar(x - w / 2, df.train_oof_top1, w, label="training OOF Top-1 (decision metric)", color="#7e57c2")
    ax.bar(x + w / 2, df.student_test_top1, w, label="student-test Top-1 (reported only)", color="#26a69a")
    ax.axhline(.95, ls="--", c="k", lw=1, label="95% target")
    ax.set_xticks(x, df.step); ax.set_ylim(.8, 1.0); ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Accuracy ladder: each experiment measured on the same partitions")
    for i, v in enumerate(df.student_test_top1):
        ax.text(i + w / 2, v + .003, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "accuracy_ladder.png", dpi=120)


def main():
    if "--plot-only" in sys.argv:
        plot(pd.read_csv(OUT / "accuracy_ladder.csv"))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = build_split("train", extra_features=True), build_split("student_test", extra_features=True)
    bad = excluded_train_trips()
    c, meta = stack(tr, [t for t in tr if t not in bad])
    tc, tm = stack(te)
    tuned = dict(joblib.load(ROOT / "models" / "tuned_lightgbm.joblib")["params"], subsample_freq=1)

    rows, cache = [], {}

    def run(step, desc, F, params, hmm=None, pick_beta=False, extra_hmm=None):
        key = (tuple(F), json.dumps(params, sort_keys=True))
        if key not in cache:
            m = lgbm(params).fit(c[F].values, c["y"].values)
            cache[key] = (oof_proba(c, F, params), m.predict_proba(tc[F].values)[:, 1])
        oof, p = cache[key]
        hp = None
        if hmm is None:
            tr_top1 = sequence_metrics(top1_from_proba(c, oof), meta, "train", continuity=False)[0]["top1"]
            st = sequence_metrics(top1_from_proba(tc, p), tm, "student_test")[0]
        else:
            hp = dict(hmm, **(extra_hmm or {}))
            if pick_beta:
                scores = {b: sequence_metrics(decode_split(tr, c, oof, dict(hp, beta=b))[0], meta, "train",
                                              continuity=False)[0]["top1"] for b in BETAS}
                hp["beta"] = max(scores, key=scores.get)
            tr_top1 = sequence_metrics(decode_split(tr, c, oof, hp)[0], meta, "train", continuity=False)[0]["top1"]
            st = sequence_metrics(decode_split(te, tc, p, hp)[0], tm, "student_test")[0]
        rows.append(dict(step=step, description=desc, features=len(F),
                         hmm=("none" if hp is None else f"beta={hp['beta']}, w={hp['w_emit']}"
                              + (", signed" if hp.get("signed_back") else "")),
                         train_oof_top1=tr_top1, student_test_top1=st["top1"],
                         student_test_trip_min=st["top1_trip_min"],
                         student_test_continuity=st["route_continuity"]))
        print(rows[-1], flush=True)

    base_hmm = dict(beta=10.0, w_emit=1.0)
    run("E0", "Baseline: untuned LightGBM, 17 features, classifier-only Top-1", FEATURES_V1, UNTUNED)
    run("E1", "+ HMM/Viterbi, default decoder (beta=10 m)", FEATURES_V1, UNTUNED, hmm=base_hmm)
    run("E1b", "Ablation: signed same-arc distance in transitions", FEATURES_V1, UNTUNED, hmm=base_hmm,
        extra_hmm=dict(signed_back=True))
    run("E2", "+ decoder beta chosen on training OOF", FEATURES_V1, UNTUNED, hmm=base_hmm, pick_beta=True)
    run("E3a", "23 features (wider heading/context, twin flag), classifier-only", FEATURES_V2, UNTUNED)
    run("E3b", "23 features + HMM (beta chosen on training OOF)", FEATURES_V2, UNTUNED, hmm=base_hmm,
        pick_beta=True)
    run("E4a", "Bayesian-tuned LightGBM, 17 features, classifier-only", FEATURES_V1, tuned)
    run("E4b", "Bayesian-tuned LightGBM + HMM (beta chosen on training OOF)", FEATURES_V1, tuned, hmm=base_hmm,
        pick_beta=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "accuracy_ladder.csv", index=False)
    print(df.round(4).to_string())

    plot(df)


if __name__ == "__main__":
    main()

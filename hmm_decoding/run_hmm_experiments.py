"""HMM / Viterbi route reconstruction experiments.

1. Out-of-fold (5-fold GroupKFold) classifier probabilities on the training trips.
2. Grid over decoder parameters (beta, w_emit) on those OOF probabilities -> pick the best.
   Decoder parameters are chosen on TRAINING data only.
3. Refit the classifier on all training trips, predict the student-test trips and compare
   classifier-only Top-1 with classifier + HMM/Viterbi.
4. Per-trip analysis, "where HMM hurt" analysis, plots and Folium maps.

Run from the repo root:  python hmm_decoding/run_hmm_experiments.py [--model tuned|untuned]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import folium
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.candidates import FEATURES_V1  # noqa: E402
from pathsnap.data import load_trip, project  # noqa: E402
from pathsnap.dataset import build_split, excluded_train_trips, stack  # noqa: E402
from pathsnap.hmm import DEFAULT_HMM, decode_split  # noqa: E402
from pathsnap.metrics import classifier_metrics, sequence_metrics, top1_from_proba  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
SEED = 42
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8)
GRID_BETA = [2, 3, 5, 7, 10, 20, 40]
GRID_W = [0.5, 1.0, 2.0]


def get_params(which):
    if which == "tuned":
        return dict(joblib.load(ROOT / "models" / "tuned_lightgbm.joblib")["params"], subsample_freq=1)
    return UNTUNED


def onehot_metrics(cand, pred):
    """Candidate-row macro F1 when the decoded arc is marked 1 and every other candidate 0."""
    chosen = cand[["trip", "pt", "arc"]].merge(pred.assign(sel=1), on=["trip", "pt", "arc"], how="left")
    return classifier_metrics(cand, chosen["sel"].fillna(0).values.astype(float))


def trip_map(split, tid, top, hmm, path, focus=None):
    """Folium map. focus=(lat_min, lon_min, lat_max, lon_max) zooms to a window instead of the whole trip."""
    track, nodes, arcs, route = load_trip(split, tid)
    lat, lon = track[:, 1], track[:, 0]
    fm = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15, tiles="OpenStreetMap")

    def seg(a):
        return [[nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]], [nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0]]]

    # road network near the trip: arcs whose two end nodes are both within 120 m of some GPS fix
    lon0, lat0 = lon.mean(), lat.mean()
    near = cKDTree(project(lon, lat, lon0, lat0)).query(project(nodes[:, 0], nodes[:, 1], lon0, lat0),
                                                        distance_upper_bound=120)[0] < np.inf
    g_net = folium.FeatureGroup("Road network (within 120 m)")
    for a in np.flatnonzero(near[arcs[:, 0]] & near[arcs[:, 1]]):
        folium.PolyLine(seg(a), color="#616161", weight=1.5, opacity=.6).add_to(g_net)
    g_ref = folium.FeatureGroup("Reference route (.route)")
    for a in route:
        folium.PolyLine(seg(a), color="green", weight=9, opacity=.4).add_to(g_ref)
    g_clf = folium.FeatureGroup("Classifier-only Top-1", show=focus is not None)
    for a, ok in top[["arc", "ok"]].drop_duplicates().values:
        folium.PolyLine(seg(int(a)), color="#ff9800" if ok else "#d32f2f", weight=6, opacity=.9,
                        dash_array="6 6").add_to(g_clf)
    g_hmm = folium.FeatureGroup("HMM / Viterbi route")
    for a, ok in hmm[["arc", "ok"]].drop_duplicates().values:
        folium.PolyLine(seg(int(a)), color="#1565c0" if ok else "#d32f2f", weight=3.5).add_to(g_hmm)
    g_gps = folium.FeatureGroup("GPS observations (every 5th fix)")
    for x, y in track[::5, :2]:
        folium.CircleMarker([y, x], radius=1.5, color="black", weight=1, fill=True, opacity=.45).add_to(g_gps)
    for g in (g_net, g_ref, g_clf, g_hmm, g_gps):
        g.add_to(fm)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(fm)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(fm)
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;"
              "border:1px solid #888;font-size:12px;line-height:1.5'><b>Trip " + tid + "</b><br>"
              "<span style='color:green'>&#9644;&#9644;</span> reference route<br>"
              "<span style='color:#1565c0'>&#9644;</span> HMM route, correct arc<br>"
              "<span style='color:#ff9800'>- -</span> classifier-only, correct arc<br>"
              "<span style='color:#d32f2f'>&#9644;</span> wrong arc (solid HMM, dashed classifier)<br>"
              "<span style='color:#616161'>&#9644;</span> road network<br>&#9679; GPS fixes</div>")
    fm.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fm)
    if focus is None:
        fm.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    else:
        fm.fit_bounds([[focus[0], focus[1]], [focus[2], focus[3]]])
    fm.save(path)


def best_window(split, tid, x, win=40):
    """Bounding box of the 40-observation window where HMM fixed the most classifier errors."""
    g = x[x.trip == tid].sort_values("pt")
    fixed = ((~g.clf_ok) & g.hmm_ok).astype(int).values
    if len(g) <= win:
        i0 = 0
    else:
        i0 = int(np.argmax(np.convolve(fixed, np.ones(win), "valid")))
    pts = g.pt.values[i0:i0 + win]
    track = load_trip(split, tid, with_route=False)[0][pts.min():pts.max() + 1]
    pad = 0.0006
    return (track[:, 1].min() - pad, track[:, 0].min() - pad, track[:, 1].max() + pad, track[:, 0].max() + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tuned", choices=["tuned", "untuned"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)
    params = get_params(args.model)
    F = FEATURES_V1

    trips = build_split("train")
    bad = excluded_train_trips()
    c, meta = stack(trips, [t for t in trips if t not in bad])

    # 1. OOF probabilities on training trips
    oof = np.zeros(len(c))
    for a, b in GroupKFold(5).split(c, c["y"], c["trip"]):
        m = lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(c.iloc[a][F].values, c["y"].values[a])
        oof[b] = m.predict_proba(c.iloc[b][F].values)[:, 1]
    clf_oof = sequence_metrics(top1_from_proba(c, oof), meta, "train", continuity=False)[0]["top1"]

    # 2. decoder grid on training OOF
    grid = []
    for beta in GRID_BETA:
        for w in GRID_W:
            t0 = time.perf_counter()
            pred, rs = decode_split(trips, c, oof, dict(beta=beta, w_emit=w))
            r = sequence_metrics(pred, meta, "train", continuity=False)[0]
            grid.append(dict(beta=beta, w_emit=w, beta_over_w=beta / w, train_oof_top1=r["top1"],
                             restarts=rs, seconds=time.perf_counter() - t0))
            print(grid[-1], flush=True)
    for signed in (True,):   # ablation E1b
        pred, rs = decode_split(trips, c, oof, dict(beta=5, w_emit=1.0, signed_back=signed))
        r = sequence_metrics(pred, meta, "train", continuity=False)[0]
        grid.append(dict(beta=5, w_emit=1.0, beta_over_w=5, train_oof_top1=r["top1"], restarts=rs,
                         seconds=np.nan, note="signed_back=True (ablation)"))
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "decoder_grid_train_oof.csv", index=False)
    main_grid = grid[grid.get("note").isna()] if "note" in grid else grid
    # best on training OOF; ties (same beta/w ratio) -> smaller w_emit
    best = main_grid.sort_values(["train_oof_top1", "w_emit"], ascending=[False, True]).iloc[0]
    hmm_cfg = dict(DEFAULT_HMM, beta=float(best.beta), w_emit=float(best.w_emit))
    json.dump(dict(hmm=hmm_cfg, chosen_on="training OOF Top-1 (5-fold GroupKFold)",
                   classifier=args.model, train_oof_classifier_top1=clf_oof,
                   train_oof_hmm_top1=float(best.train_oof_top1)),
              open(OUT / "hmm_config.json", "w"), indent=2)
    print("chosen", hmm_cfg)

    # 3. student-test comparison
    te = build_split("student_test")
    tc, tm = stack(te)
    m = lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1).fit(c[F].values, c["y"].values)
    t0 = time.perf_counter(); p = m.predict_proba(tc[F].values)[:, 1]; t_clf = time.perf_counter() - t0
    top = top1_from_proba(tc, p)
    t0 = time.perf_counter(); hmm, restarts = decode_split(te, tc, p, hmm_cfg); t_hmm = time.perf_counter() - t0
    r_clf, pt_clf = sequence_metrics(top, tm, "student_test")
    r_hmm, pt_hmm = sequence_metrics(hmm, tm, "student_test")
    cm_clf, cm_hmm = onehot_metrics(tc, top), onehot_metrics(tc, hmm)
    comp = pd.DataFrame([
        dict(method="Classifier only (Top-1)", **r_clf, onehot_macro_f1=cm_clf["macro_f1"], seconds=t_clf),
        dict(method="Classifier + HMM/Viterbi", **r_hmm, onehot_macro_f1=cm_hmm["macro_f1"],
             seconds=t_clf + t_hmm, chain_restarts=restarts),
    ])
    comp.to_csv(OUT / "student_test_classifier_vs_hmm.csv", index=False)
    print(comp.T)
    per_trip = pd.DataFrame({"classifier_only": pt_clf, "hmm": pt_hmm})
    per_trip["delta"] = per_trip["hmm"] - per_trip["classifier_only"]
    per_trip.to_csv(OUT / "student_test_per_trip.csv")

    # 4. where HMM helped / hurt (observation level)
    x = tm[["trip", "pt", "true_arc"]].merge(top.rename(columns={"arc": "clf"}), on=["trip", "pt"], how="left") \
        .merge(hmm.rename(columns={"arc": "hmm"}), on=["trip", "pt"], how="left")
    x["clf_ok"], x["hmm_ok"] = x.clf == x.true_arc, x.hmm == x.true_arc
    flips = pd.crosstab(x.clf_ok, x.hmm_ok, rownames=["classifier correct"], colnames=["HMM correct"])
    flips.to_csv(OUT / "student_test_flip_table.csv")
    print(flips)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    per_trip[["classifier_only", "hmm"]].plot.bar(ax=ax[0], rot=45, color=["#ff9800", "#1565c0"])
    ax[0].set_ylim(.6, 1.01); ax[0].set_title("Student-test Top-1 per trip")
    pv = main_grid.pivot(index="beta", columns="w_emit", values="train_oof_top1")
    for w in pv.columns:
        ax[1].plot(pv.index, pv[w], "o-", label=f"w_emit={w}")
    ax[1].axhline(clf_oof, ls="--", c="grey", label="classifier-only (OOF)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("beta (m)"); ax[1].set_ylabel("training OOF Top-1")
    ax[1].set_title("Decoder parameter search (training trips only)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT / "hmm_results.png", dpi=120)

    top_ok = top.merge(tm[["trip", "pt", "true_arc"]], on=["trip", "pt"]).assign(ok=lambda d: d.arc == d.true_arc)
    hmm_ok = hmm.merge(tm[["trip", "pt", "true_arc"]], on=["trip", "pt"]).assign(ok=lambda d: d.arc == d.true_arc)
    for tid in tm.trip.unique():
        trip_map("student_test", tid, top_ok[top_ok.trip == tid], hmm_ok[hmm_ok.trip == tid],
                 OUT / "maps" / f"student_test_{tid}_hmm.html")
    worst = per_trip["classifier_only"].idxmin()   # trip where the classifier alone struggles most
    trip_map("student_test", worst, top_ok[top_ok.trip == worst], hmm_ok[hmm_ok.trip == worst],
             OUT / "maps" / f"student_test_{worst}_hmm_zoom.html", focus=best_window("student_test", worst, x))
    print("maps written")


if __name__ == "__main__":
    main()

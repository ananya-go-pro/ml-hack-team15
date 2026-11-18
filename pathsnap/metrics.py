"""Evaluation metrics. Definitions match model_comparison.ipynb section 4."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from .data import read_arcs

MAX_HOPS = 4
_GRAPH = {}


def get_graph(split, tid, arcs=None):
    key = (split, tid)
    if key not in _GRAPH:
        arcs = read_arcs(split, tid) if arcs is None else arcs
        order = np.argsort(arcs[:, 0], kind="stable")
        start = np.searchsorted(arcs[order, 0], np.arange(arcs[:, :2].max() + 2))
        _GRAPH[key] = (arcs, order, start)
    return _GRAPH[key]


def reachable(a, b, graph, max_hops=MAX_HOPS):
    arcs, order, start = graph
    frontier = {a}
    for _ in range(max_hops):
        nxt = set()
        for x in frontier:
            nd = arcs[x, 1]
            outs = order[start[nd]:start[nd + 1]]
            if b in outs:
                return True
            nxt.update(outs.tolist())
        frontier = nxt
    return False


def route_continuity(seq_df, split):
    """Share of consecutive predictions that are the same arc or reachable within MAX_HOPS directed hops."""
    ok = tot = 0
    for tid, g in seq_df.sort_values(["trip", "pt"]).groupby("trip"):
        a = g["arc"].values
        graph = get_graph(split, tid)
        for x, y in zip(a[:-1], a[1:]):
            tot += 1
            ok += (x == y) or reachable(int(x), int(y), graph)
    return ok / max(tot, 1)


def classifier_metrics(df, proba):
    """Candidate-row metrics (threshold 0.5)."""
    y = df["y"].values
    yhat = (proba >= 0.5).astype(int)
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    return dict(accuracy=accuracy_score(y, yhat),
                precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0),
                macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
                log_loss=log_loss(y, p, labels=[0, 1]))


def top1_from_proba(df, proba):
    tmp = df[["trip", "pt", "arc"]].copy()
    tmp["p"] = proba
    return tmp.loc[tmp.groupby(["trip", "pt"])["p"].idxmax(), ["trip", "pt", "arc"]].reset_index(drop=True)


def sequence_metrics(pred, meta, split, continuity=True):
    """pred: one row per observation with columns trip, pt, arc. Missing observations count as wrong."""
    m = meta[["trip", "pt", "true_arc"]].merge(pred[["trip", "pt", "arc"]], on=["trip", "pt"], how="left")
    m["ok"] = (m["arc"] == m["true_arc"]).astype(float)
    per_trip = m.groupby("trip")["ok"].mean()
    r = dict(top1=m["ok"].mean(), top1_trip_mean=per_trip.mean(), top1_trip_min=per_trip.min())
    # route-level: arc-set precision/recall of the predicted route vs the arcs the reference visits
    pr, rc = [], []
    for tid, g in m.dropna(subset=["arc"]).groupby("trip"):
        ps, ts_ = set(g["arc"].astype(int)), set(g["true_arc"].astype(int))
        pr.append(len(ps & ts_) / max(len(ps), 1))
        rc.append(len(ps & ts_) / max(len(ts_), 1))
    r["route_arc_precision"], r["route_arc_recall"] = float(np.mean(pr)), float(np.mean(rc))
    if continuity:
        r["route_continuity"] = route_continuity(pred.dropna(subset=["arc"]).astype({"arc": int}), split)
    return r, per_trip

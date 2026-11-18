"""Reference labelling, candidate generation and candidate features.

Ported from model_comparison/model_comparison.ipynb so the same code runs in the
experiments, the HMM and (later) the API. With the default config and
FEATURES_V1 it reproduces the notebook's numbers exactly.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import load_trip, project, seg_dist

DEFAULT_CFG = dict(
    R_CAND=50.0,        # candidate search radius (m)
    K_MAX=12,           # max candidates kept per observation (nearest first)
    SAMPLE_EVERY=5,     # use every 5th GPS fix (~5 s at 1 Hz)
    HEAD_H=2,           # half-window (fixes) for heading / speed estimation
    NB_OFF=3,           # offset (fixes) for previous/next context distances
    DENS_STEP=10.0,     # arc densification step for the KD-tree (m)
)

# Feature set used in the 5-model comparison notebook.
FEATURES_V1 = ["dist", "rank", "dist_rel", "n_cand", "head_diff", "head_rel", "speed", "disp", "turn",
               "dt_prev", "t_clip", "t_out", "arc_len", "dist_prev", "dist_next", "ctx_dist", "ctx_rel"]

# Extra features added in experiment E3 (wider temporal context and a longer heading window).
FEATURES_EXTRA = ["head_diff_long", "head_rel_long", "ctx_dist_wide", "ctx_rel_wide",
                  "dist_rel_ctx_rank", "twin_exists"]
FEATURES_V2 = FEATURES_V1 + FEATURES_EXTRA


@dataclass
class TripData:
    tid: str
    pxy: np.ndarray          # projected GPS fixes (N, 2)
    ts: np.ndarray           # timestamps (N,)
    nxy: np.ndarray          # projected nodes
    arcs: np.ndarray         # (n_arcs, 2) from/to node ids
    cand: pd.DataFrame       # one row per (sampled fix, candidate arc)
    meta: pd.DataFrame       # one row per sampled fix
    extra: dict = field(default_factory=dict)


def label_points(pxy, Axy, Bxy, route):
    """Monotone (DP) alignment of every GPS fix to a position on the reference route."""
    N, M = len(pxy), len(route)
    RA, RB = Axy[route], Bxy[route]
    AB = RB - RA
    L2 = np.where((AB ** 2).sum(1) > 0, (AB ** 2).sum(1), 1)
    cost = np.empty((N, M), dtype=np.float32)
    for s in range(0, N, 1000):
        P = pxy[s:s + 1000]
        t = np.clip(((P[:, None, :] - RA[None]) * AB[None]).sum(2) / L2, 0, 1)
        cost[s:s + 1000] = np.linalg.norm(P[:, None, :] - (RA[None] + AB[None] * t[..., None]), axis=2)
    back = np.zeros((N, M), dtype=np.int32)
    idx = np.arange(M)
    prev = cost[0].astype(np.float64)
    for i in range(1, N):
        pm = np.minimum.accumulate(prev)
        back[i] = np.maximum.accumulate(np.where(prev <= pm, idx, 0))   # argmin of prefix
        prev = cost[i] + pm
    j = int(np.argmin(prev))
    pos = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        pos[i] = j
        j = back[i, j]
    return route[pos], cost[np.arange(N), pos], pos


def _bearing(v):
    return np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360


def _angdiff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def build_trip(split, tid, cfg=None, with_labels=True, extra_features=False, raw=None):
    """Build candidate rows for one trip.

    raw: optional (track, nodes, arcs, route) tuple, so the API can pass data it already loaded.
    """
    c = dict(DEFAULT_CFG, **(cfg or {}))
    R_CAND, K_MAX, SAMPLE_EVERY = c["R_CAND"], c["K_MAX"], c["SAMPLE_EVERY"]
    HEAD_H, NB_OFF, DENS_STEP = c["HEAD_H"], c["NB_OFF"], c["DENS_STEP"]

    track, nodes, arcs, route = raw if raw is not None else load_trip(split, tid, with_route=with_labels)
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    pxy = project(track[:, 0], track[:, 1], lon0, lat0)
    nxy = project(nodes[:, 0], nodes[:, 1], lon0, lat0)
    Axy, Bxy = nxy[arcs[:, 0]], nxy[arcs[:, 1]]
    N, ts = len(pxy), track[:, 2]
    if with_labels:
        true_arc, true_d, true_pos = label_points(pxy, Axy, Bxy, route)

    # spatial index over arcs near the trajectory
    lo, hi = pxy.min(0) - R_CAND - 50, pxy.max(0) + R_CAND + 50
    amin, amax = np.minimum(Axy, Bxy), np.maximum(Axy, Bxy)
    keep = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) & (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
    L = np.linalg.norm(Bxy[keep] - Axy[keep], axis=1)
    n = np.maximum(1, np.ceil(L / DENS_STEP).astype(int))
    rep = np.repeat(np.arange(len(keep)), n + 1)
    frac = np.concatenate([np.arange(m + 1) / m for m in n])
    tree = cKDTree(Axy[keep][rep] + (Bxy[keep][rep] - Axy[keep][rep]) * frac[:, None])

    # motion features on the raw 1 Hz sequence
    ar = np.arange(N)
    im, ip = np.clip(ar - HEAD_H, 0, N - 1), np.clip(ar + HEAD_H, 0, N - 1)
    dv = pxy[ip] - pxy[im]
    disp = np.linalg.norm(dv, axis=1)
    speed = np.clip(disp / np.maximum(ts[ip] - ts[im], 1e-3), 0, 70)
    gps_head = _bearing(dv)
    dt_prev = np.clip(np.r_[0, np.diff(ts)], 0, 300)
    turn = np.zeros(N)
    if N > 2:
        turn[1:-1] = _angdiff(_bearing(pxy[2:] - pxy[1:-1]), _bearing(pxy[1:-1] - pxy[:-2]))

    sidx = np.arange(0, N, SAMPLE_EVERY)
    meta = pd.DataFrame({"trip": tid, "pt": sidx, "ts": ts[sidx]})
    if with_labels:
        meta["true_arc"] = true_arc[sidx]
        meta["true_d"] = true_d[sidx]
        meta["true_pos"] = true_pos[sidx]

    hits = tree.query_ball_point(pxy[sidx], r=R_CAND + DENS_STEP / 2 + 1)
    P, A = [], []
    for si, h in zip(sidx, hits):
        if h:
            a = np.unique(keep[rep[h]])
            P.append(np.full(len(a), si))
            A.append(a)
    cols = ["trip", "pt", "arc", "t"] + FEATURES_V1
    if not P:
        return TripData(tid, pxy, ts, nxy, arcs, pd.DataFrame(columns=cols), meta)
    P, A = np.concatenate(P), np.concatenate(A)
    d, t = seg_dist(pxy[P], Axy[A], Bxy[A])
    m = d <= R_CAND
    df = pd.DataFrame({"trip": tid, "pt": P[m], "arc": A[m], "dist": d[m], "t": t[m]}).sort_values(["pt", "dist"])
    df["rank"] = df.groupby("pt").cumcount()
    df = df[df["rank"] < K_MAX].reset_index(drop=True)
    P, A = df["pt"].values, df["arc"].values

    av = Bxy[A] - Axy[A]
    arc_bear = _bearing(av)
    df["arc_len"] = np.linalg.norm(av, axis=1)
    df["head_diff"] = _angdiff(gps_head[P], arc_bear)
    df["speed"], df["disp"], df["dt_prev"], df["turn"] = speed[P], disp[P], dt_prev[P], turn[P]
    df["t_clip"] = df["t"].clip(0, 1)
    df["t_out"] = np.maximum(0, np.maximum(-df["t"], df["t"] - 1)) * df["arc_len"]
    df["dist_prev"], _ = seg_dist(pxy[np.clip(P - NB_OFF, 0, N - 1)], Axy[A], Bxy[A])
    df["dist_next"], _ = seg_dist(pxy[np.clip(P + NB_OFF, 0, N - 1)], Axy[A], Bxy[A])
    df["ctx_dist"] = (df["dist"] + df["dist_prev"] + df["dist_next"]) / 3
    g = df.groupby("pt")
    df["n_cand"] = g["arc"].transform("size")
    df["dist_rel"] = df["dist"] - g["dist"].transform("min")
    df["head_rel"] = df["head_diff"] - g["head_diff"].transform("min")
    df["ctx_rel"] = df["ctx_dist"] - g["ctx_dist"].transform("min")

    if extra_features:
        # Heading over a wider window (+-6 fixes): more stable at low speed / GPS jitter.
        H2 = 3 * HEAD_H
        im2, ip2 = np.clip(ar - H2, 0, N - 1), np.clip(ar + H2, 0, N - 1)
        head_long = _bearing(pxy[ip2] - pxy[im2])
        df["head_diff_long"] = _angdiff(head_long[P], arc_bear)
        df["head_rel_long"] = df["head_diff_long"] - df.groupby("pt")["head_diff_long"].transform("min")
        # Wider context: mean distance of fixes at offsets +-3, +-6, +-9 to the same arc.
        acc = df["dist"].values.copy()
        for k in (1, 2, 3):
            for s in (-1, 1):
                dd, _ = seg_dist(pxy[np.clip(P + s * k * NB_OFF, 0, N - 1)], Axy[A], Bxy[A])
                acc = acc + np.minimum(dd, 3 * R_CAND)
        df["ctx_dist_wide"] = acc / 7
        df["ctx_rel_wide"] = df["ctx_dist_wide"] - df.groupby("pt")["ctx_dist_wide"].transform("min")
        df["dist_rel_ctx_rank"] = df.groupby("pt")["ctx_dist_wide"].rank(method="first") - 1
        # Is the reverse-direction twin of this arc also a candidate? (two-way street ambiguity)
        fr, to = arcs[A, 0], arcs[A, 1]
        pair_rev = set(zip(P.tolist(), to.tolist(), fr.tolist()))
        df["twin_exists"] = np.fromiter((x in pair_rev for x in zip(P.tolist(), fr.tolist(), to.tolist())),
                                        dtype=np.int8, count=len(df))

    if with_labels:
        df["y"] = (df["arc"].values == true_arc[P]).astype(np.int8)
    return TripData(tid, pxy, ts, nxy, arcs, df, meta)

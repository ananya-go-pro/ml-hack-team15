"""HMM map-matching with Viterbi decoding.

Hidden state at observation k : one candidate road arc (plus where the fix projects onto it).
Emission  log P(z_k | arc)    : w_e * log(classifier probability), clipped at P_FLOOR.
Transition log P(arc_k | arc_{k-1}): -|route_distance - straight_line_distance| / beta
    (Newson & Krumm, 2009). route_distance is the network distance travelled from the
    previous projection point to the current one along directed arcs. Arcs that cannot
    be reached within the search cutoff get -inf.
Gaps / breaks: if the time gap or distance between two observations is very large, or no
    state at step k is reachable from any state at k-1, the chain is restarted at k using
    the best previous score (no transition penalty), so decoding never fails.
Observations with no candidates are skipped and get no prediction.
"""
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd

DEFAULT_HMM = dict(
    beta=10.0,          # transition scale (m)
    w_emit=1.0,         # weight on log classifier probability
    p_floor=1e-4,       # probability floor for emissions
    back_tol=15.0,      # backwards movement on the same arc tolerated as GPS jitter (m)
    gap_dt=120.0,       # time gap (s) above which the chain is restarted
    gap_dist=2000.0,    # straight-line gap (m) above which the chain is restarted
    search_extra=400.0, # Dijkstra search radius = 2 * straight-line distance + this (m)
    signed_back=False,  # E1b ablation: use signed (not absolute) distance for backwards moves on an arc
)


class RoadGraph:
    """Directed road graph with cached, distance-bounded Dijkstra from nodes."""

    def __init__(self, nxy, arcs):
        self.nxy, self.arcs = nxy, arcs
        self.alen = np.linalg.norm(nxy[arcs[:, 1]] - nxy[arcs[:, 0]], axis=1)
        self.adj = defaultdict(list)
        for i, (u, v) in enumerate(arcs):
            self.adj[int(u)].append((int(v), float(self.alen[i])))
        self._cache = {}

    def dists_from(self, src, cutoff):
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for v, w in self.adj.get(u, ()):
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._cache[src] = (cutoff, dist)
        return dist


def _route_dist(g, a_prev, t_prev, a_cur, t_cur, cutoff, back_tol, signed_back=False):
    """Network distance matrix between previous and current candidate projection points."""
    arcs, L = g.arcs, g.alen
    R = np.full((len(a_prev), len(a_cur)), np.inf)
    for i, (a, ta) in enumerate(zip(a_prev, t_prev)):
        rem = (1.0 - ta) * L[a]
        dists = g.dists_from(int(arcs[a, 1]), cutoff)
        for j, (b, tb) in enumerate(zip(a_cur, t_cur)):
            if a == b:
                fwd = (tb - ta) * L[a]
                if fwd >= -back_tol:
                    # small backwards moves are treated as GPS jitter. The signed variant
                    # (ablation E1b) was worse on training OOF, see experiments/README.md
                    R[i, j] = fwd if signed_back else abs(fwd)
                    continue
            d = dists.get(int(arcs[b, 0]))
            if d is not None:
                R[i, j] = rem + d + tb * L[b]
    return R


def viterbi_trip(trip, proba, params=None):
    """Decode one trip. trip: TripData; proba: classifier P(y=1) aligned with trip.cand rows.

    Returns DataFrame (trip, pt, arc) with one row per observation that had candidates,
    plus the number of chain restarts.
    """
    p = dict(DEFAULT_HMM, **(params or {}))
    c = trip.cand
    if len(c) == 0:
        return pd.DataFrame(columns=["trip", "pt", "arc"]), 0
    g = trip.extra.get("graph")
    if g is None:
        g = trip.extra["graph"] = RoadGraph(trip.nxy, trip.arcs)

    emit = p["w_emit"] * np.log(np.clip(np.asarray(proba, float), p["p_floor"], 1.0))
    t_clip = c["t"].clip(0, 1).values if "t" in c else c["t_clip"].values
    pts = c["pt"].values
    bounds = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1], True])
    steps = [(pts[s], c["arc"].values[s:e], t_clip[s:e], emit[s:e]) for s, e in zip(bounds[:-1], bounds[1:])]

    deltas, backs = [], []
    restarts = 0
    for k, (pt, arcs_k, t_k, e_k) in enumerate(steps):
        if k == 0:
            deltas.append(e_k.copy()); backs.append(np.zeros(len(arcs_k), int)); continue
        ppt, parcs, pt_, _ = steps[k - 1]
        gc = float(np.linalg.norm(trip.pxy[pt] - trip.pxy[ppt]))
        dt = float(trip.ts[pt] - trip.ts[ppt])
        prev = deltas[-1]
        if dt > p["gap_dt"] or gc > p["gap_dist"]:
            restart = True
        else:
            cutoff = 2 * gc + p["search_extra"]
            R = _route_dist(g, parcs, pt_, arcs_k, t_k, cutoff, p["back_tol"], p["signed_back"])
            trans = -np.abs(R - gc) / p["beta"]          # -inf where unreachable
            score = prev[:, None] + trans
            best = np.argmax(score, axis=0)
            val = score[best, np.arange(len(arcs_k))]
            restart = not np.isfinite(val).any()
        if restart:
            restarts += 1
            best = np.full(len(arcs_k), int(np.argmax(prev)))
            val = np.full(len(arcs_k), prev.max())
        d = val + e_k
        d -= np.nanmax(d[np.isfinite(d)])               # keep numbers bounded
        deltas.append(d); backs.append(best)

    # backtrack
    j = int(np.argmax(deltas[-1]))
    out = np.empty(len(steps), dtype=np.int64)
    for k in range(len(steps) - 1, -1, -1):
        out[k] = steps[k][1][j]
        j = backs[k][j]
    return pd.DataFrame({"trip": trip.tid, "pt": [s[0] for s in steps], "arc": out}), restarts


def decode_split(trips, cand_all, proba_all, params=None):
    """Run Viterbi over every trip present in cand_all (rows must be grouped by trip, sorted by pt)."""
    out, restarts = [], 0
    proba_all = np.asarray(proba_all)
    for tid, idx in cand_all.groupby("trip", sort=False).indices.items():
        tr = trips[tid]
        # cand_all rows for the trip are in the same order as tr.cand
        df, r = viterbi_trip(tr, proba_all[idx], params)
        out.append(df); restarts += r
    return pd.concat(out, ignore_index=True), restarts

"""Server-side evaluation. The ONLY place the reference route is read.

Called after prediction has finished. Only aggregate numbers leave this module; the reference
route, its geometry and per-observation labels are never returned to the client.
Metric definitions are the same as in the offline experiments (pathsnap/metrics.py).
"""
import numpy as np

from pathsnap.candidates import label_points
from pathsnap.metrics import MAX_HOPS, get_graph, reachable
from pathsnap.raw import load_reference_route


def evaluate(trips_dir, trip, result):
    route = load_reference_route(trips_dir, trip.tid, len(trip.arcs))
    Axy, Bxy = trip.nxy[trip.arcs[:, 0]], trip.nxy[trip.arcs[:, 1]]
    true_arc, true_d, _ = label_points(trip.pxy, Axy, Bxy, route)

    obs = result["observations"]                  # sampled observation indices (every 5th fix)
    truth = true_arc[obs]
    out = {}
    for name in ("hmm", "classifier"):
        pred = dict(zip(result[name]["pt"].tolist(), result[name]["arc"].tolist()))
        p = np.array([pred.get(int(i), -1) for i in obs])
        out[f"top1_{name}"] = float((p == truth).mean())      # missing prediction counts as wrong

    # route continuity of the HMM route (same arc or reachable within MAX_HOPS directed hops)
    graph = get_graph("api", f"{trips_dir}/{trip.tid}", arcs=trip.arcs)
    seq = result["hmm"]["arc"].tolist()
    pairs = list(zip(seq[:-1], seq[1:]))
    ok = sum((a == b) or reachable(int(a), int(b), graph, MAX_HOPS) for a, b in pairs)
    out["route_continuity"] = ok / max(len(pairs), 1)

    ps, ts = set(seq), set(truth.tolist())
    out["route_arc_precision"] = len(ps & ts) / max(len(ps), 1)
    out["route_arc_recall"] = len(ps & ts) / max(len(ts), 1)
    cand = trip.cand
    covered = set(zip(cand["pt"].tolist(), cand["arc"].tolist()))
    out["candidate_miss_rate"] = float(np.mean([(int(i), int(a)) not in covered for i, a in zip(obs, truth)]))
    out["observations_evaluated"] = int(len(obs))
    out["correct_hmm"] = int(round(out["top1_hmm"] * len(obs)))
    return out

"""PathSnap FastAPI backend (Laptop 1).

Hosts the tuned LightGBM classifier, the HMM/Viterbi decoder, the road networks of the trips in
--trips-dir and the server-side evaluation.

    python api/main.py --trips-dir /path/to/evaluator_test --host 0.0.0.0 --port 8000

Endpoints
    GET  /health              model + decoder info, number of trips available
    GET  /trips               the trip ids in --trips-dir (with GPS point counts)
    POST /predict {"trip_id"} runs the full pipeline on that trip and returns the predicted route,
                              the GPS observations and server-side accuracy numbers

The reference route of a trip is read only by api/evaluation.py, after prediction. It is never
sent to the client and never passed to the model.
"""
import argparse
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from api.evaluation import evaluate  # noqa: E402
from pathsnap.inference import MapMatcher  # noqa: E402
from pathsnap.raw import count_gps_points, list_trip_dirs, load_inputs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT.parent / "evaluator_test"))

STATE = {}


class PredictRequest(BaseModel):
    trip_id: str


def startup():
    t0 = time.perf_counter()
    STATE["matcher"] = MapMatcher()
    STATE["trips"] = list_trip_dirs(TRIPS_DIR)
    m = STATE["matcher"]
    log.info("Loaded model '%s' (%d features) and HMM config beta=%s w_emit=%s in %.2fs",
             m.name, len(m.features), m.hmm_cfg["beta"], m.hmm_cfg["w_emit"], time.perf_counter() - t0)
    log.info("Trips folder: %s -> %d trips: %s", TRIPS_DIR, len(STATE["trips"]), ", ".join(STATE["trips"]))


@asynccontextmanager
async def lifespan(_app):
    startup()
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health():
    m = STATE["matcher"]
    return dict(status="ok", model=m.name, features=len(m.features), hmm=dict(beta=m.hmm_cfg["beta"],
                w_emit=m.hmm_cfg["w_emit"]), sample_every=m.cfg["SAMPLE_EVERY"], trips=len(STATE["trips"]))


@app.get("/trips")
def trips():
    return [dict(trip_id=t, gps_points=count_gps_points(TRIPS_DIR, t)) for t in STATE["trips"]]


def _segment(nodes, arcs, a):
    u, v = arcs[a]
    return [[round(nodes[u, 1], 7), round(nodes[u, 0], 7)], [round(nodes[v, 1], 7), round(nodes[v, 0], 7)]]


def _route_geometry(nodes, arcs, seq):
    """Predicted arcs in travel order, consecutive duplicates removed."""
    out, last = [], None
    for a in seq:
        if a != last:
            out.append(dict(arc_id=int(a), coords=_segment(nodes, arcs, a)))
            last = a
    return out


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id
    if tid not in STATE["trips"]:          # also blocks path traversal: only listed folder names are accepted
        raise HTTPException(404, f"unknown trip_id {tid!r}")
    client = request.client.host if request.client else "?"
    log.info("=== /predict trip %s requested by %s", tid, client)
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    track, nodes, arcs = load_inputs(TRIPS_DIR, tid)
    t_load = time.perf_counter() - t0
    log.info("[%s] loaded %d GPS fixes, %d nodes, %d arcs (%.2fs)", tid, len(track), len(nodes), len(arcs), t_load)

    trip, res = STATE["matcher"].predict(tid, track, nodes, arcs)
    tm = res["timings"]
    log.info("[%s] %d observations, %d candidate rows (features %.2fs)", tid, len(res["observations"]),
             res["n_candidates"], tm["features_s"])
    log.info("[%s] classifier %.3fs | HMM/Viterbi %.2fs, %d chain restarts", tid, tm["classifier_s"], tm["hmm_s"],
             res["restarts"])

    t0 = time.perf_counter()
    metrics = evaluate(TRIPS_DIR, trip, res)          # reference route used here only, after prediction
    t_eval = time.perf_counter() - t0
    log.info("[%s] server-side evaluation: Top-1 HMM %.4f | classifier-only %.4f | continuity %.4f (%.2fs)",
             tid, metrics["top1_hmm"], metrics["top1_classifier"], metrics["route_continuity"], t_eval)

    obs = res["observations"]
    hmm = res["hmm"]
    total = time.perf_counter() - t_all
    log.info("[%s] done in %.2fs -> 200 OK", tid, total)
    return dict(
        trip_id=tid,
        gps=[[round(track[i, 1], 7), round(track[i, 0], 7)] for i in obs],
        predictions=[dict(obs=int(p), arc_id=int(a)) for p, a in zip(hmm["pt"], hmm["arc"])],
        route=_route_geometry(nodes, arcs, hmm["arc"].tolist()),
        classifier_route=_route_geometry(nodes, arcs, res["classifier"]["arc"].tolist()),
        metrics=metrics,
        stats=dict(gps_fixes=int(len(track)), observations=int(len(obs)), predicted=int(len(hmm)),
                   candidate_rows=res["n_candidates"], chain_restarts=res["restarts"]),
        timings=dict(load_s=t_load, **tm, evaluation_s=t_eval, total_s=total),
        model=STATE["matcher"].name,
    )


def main():
    global TRIPS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips-dir", default=str(TRIPS_DIR), help="folder with one sub-folder per trip")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    TRIPS_DIR = Path(a.trips_dir).resolve()
    if not TRIPS_DIR.is_dir():
        sys.exit(f"trips dir not found: {TRIPS_DIR}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()

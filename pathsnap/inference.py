"""End-to-end prediction for one trip: features -> tuned LightGBM -> HMM/Viterbi.

Uses no reference labels: build_trip is called with with_labels=False and the input tuple has
no route. The same code path produced the offline student-test results.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .candidates import build_trip
from .hmm import viterbi_trip
from .metrics import top1_from_proba

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_DIR / "models" / "tuned_lightgbm.joblib"
DEFAULT_HMM = REPO_DIR / "hmm_decoding" / "outputs" / "hmm_config.json"


class MapMatcher:
    def __init__(self, model_path=DEFAULT_MODEL, hmm_path=DEFAULT_HMM):
        art = joblib.load(model_path)
        self.model, self.features, self.cfg, self.name = art["model"], art["features"], art["config"], art["name"]
        self.hmm_cfg = json.load(open(hmm_path))["hmm"]

    def predict(self, tid, track, nodes, arcs):
        """Returns (trip_data, result dict with per-observation predictions and timings)."""
        t = {}
        t0 = time.perf_counter()
        trip = build_trip(None, tid, self.cfg, with_labels=False, raw=(track, nodes, arcs, None))
        t["features_s"] = time.perf_counter() - t0
        cand = trip.cand
        if len(cand):
            t0 = time.perf_counter()
            proba = self.model.predict_proba(cand[self.features].to_numpy(float))[:, 1]
            t["classifier_s"] = time.perf_counter() - t0
            clf = top1_from_proba(cand, proba)[["pt", "arc"]]
            t0 = time.perf_counter()
            hmm, restarts = viterbi_trip(trip, proba, self.hmm_cfg)
            t["hmm_s"] = time.perf_counter() - t0
            hmm = hmm[["pt", "arc"]]
        else:   # no road within the search radius of any fix
            clf = hmm = pd.DataFrame(columns=["pt", "arc"])
            restarts, t["classifier_s"], t["hmm_s"] = 0, 0.0, 0.0
        return trip, dict(observations=trip.meta["pt"].to_numpy(), hmm=hmm.astype(np.int64),
                          classifier=clf.astype(np.int64), restarts=int(restarts),
                          n_candidates=int(len(cand)), timings=t)

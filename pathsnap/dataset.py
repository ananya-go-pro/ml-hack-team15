"""Build (and cache) candidate tables for whole partitions."""
import json
import pickle

import pandas as pd

from .candidates import DEFAULT_CFG, build_trip
from .data import REPO_DIR, list_trips

CACHE_DIR = REPO_DIR / "cache"
EXCLUDED_JSON = REPO_DIR / "model_comparison" / "outputs" / "excluded_trips.json"


def build_split(split, cfg=None, extra_features=False):
    """Return {tid: TripData} for a partition. Cached in cache/ (git-ignored)."""
    c = dict(DEFAULT_CFG, **(cfg or {}))
    tag = "_".join(f"{k}{v}" for k, v in sorted(c.items())) + ("_x" if extra_features else "")
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"trips_{split}_{tag}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    trips = {tid: build_trip(split, tid, c, with_labels=True, extra_features=extra_features)
             for tid in list_trips(split)}
    with open(path, "wb") as f:
        pickle.dump(trips, f)
    return trips


def stack(trips, tids=None):
    tids = list(trips) if tids is None else tids
    cand = pd.concat([trips[t].cand for t in tids], ignore_index=True)
    meta = pd.concat([trips[t].meta for t in tids], ignore_index=True)
    return cand, meta


def excluded_train_trips():
    """Training trips whose reference route does not match the GPS (decided in the comparison notebook)."""
    return json.load(open(EXCLUDED_JSON))["excluded"]

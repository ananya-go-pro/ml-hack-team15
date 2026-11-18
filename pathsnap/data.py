"""Loading cleaned trips and projecting them to a local metric plane."""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_DIR / "data" / "processed"
EARTH_R = 6371008.8


def list_trips(split):
    return sorted(p.name[:-len("_track.csv")] for p in (PROCESSED_DIR / split).glob("*_track.csv"))


def read_arcs(split, tid):
    return pd.read_csv(PROCESSED_DIR / split / f"{tid}_arcs.csv")[["from_node_id", "to_node_id"]].to_numpy(np.int64)


def load_trip(split, tid, with_route=True):
    d = PROCESSED_DIR / split
    track = pd.read_csv(d / f"{tid}_track.csv")[["longitude", "latitude", "timestamp"]].to_numpy(float)
    nodes = pd.read_csv(d / f"{tid}_nodes.csv")[["longitude", "latitude"]].to_numpy(float)
    arcs = read_arcs(split, tid)
    route = pd.read_csv(d / f"{tid}_route.csv")["edge_id"].to_numpy(np.int64) if with_route else None
    return track, nodes, arcs, route


def project(lon, lat, lon0, lat0):
    """Local equirectangular projection (metres). Error is negligible at city scale."""
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.c_[x, y]


def seg_dist(P, A, B):
    """Row-wise distance from points P to segments AB, plus the unclipped projection parameter t."""
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    t = np.einsum("ij,ij->i", P - A, AB) / np.where(L2 > 0, L2, 1)
    proj = A + AB * np.clip(t, 0, 1)[:, None]
    return np.linalg.norm(P - proj, axis=1), t

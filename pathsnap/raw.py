"""Reading raw, headerless trip folders (<trip>/<trip>.track|.nodes|.arcs|.route).

Cleaning matches preprocessing/preprocess.py exactly, so a raw trip gives the same arrays as
its cleaned CSVs. Prediction inputs and the reference route are loaded by SEPARATE functions:
the prediction pipeline only ever calls `load_inputs`, and `load_reference_route` is only
called by the server-side evaluation after prediction has finished.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names).apply(pd.to_numeric, errors="coerce")


def list_trip_dirs(root):
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def clean_track(track):
    track = track.dropna()
    track = track[track["longitude"].between(-180, 180) & track["latitude"].between(-90, 90)]
    track = track.sort_values("timestamp", kind="stable").drop_duplicates("timestamp").reset_index(drop=True)
    return track


def load_inputs(root, tid):
    """GPS track + road network only. Never touches the .route file."""
    d = Path(root) / tid
    track = clean_track(_read(d / f"{tid}.track", ["longitude", "latitude", "timestamp"]))
    nodes = _read(d / f"{tid}.nodes", ["longitude", "latitude"])
    arcs = _read(d / f"{tid}.arcs", ["from_node_id", "to_node_id"])
    if nodes.isna().any().any() or arcs.isna().any().any():
        raise ValueError(f"{tid}: NaN in nodes/arcs (ids are row indices, rows cannot be dropped)")
    return (track[["longitude", "latitude", "timestamp"]].to_numpy(float),
            nodes.to_numpy(float), arcs.to_numpy(np.int64))


def load_reference_route(root, tid, n_arcs):
    """Reference arc ids. For server-side evaluation only."""
    route = _read(Path(root) / tid / f"{tid}.route", ["edge_id"]).dropna().astype(np.int64)["edge_id"].to_numpy()
    return route[(route >= 0) & (route < n_arcs)]


def count_gps_points(root, tid):
    with open(Path(root) / tid / f"{tid}.track") as f:
        return sum(1 for line in f if line.strip())

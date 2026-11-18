"""
PathSnap data preprocessing.

Reads the raw, headerless trip files for the `train` and `student_test`
partitions and writes cleaned CSVs with named columns to

    data/processed/<partition>/<trip_id>_{track,nodes,arcs,route}.csv
    data/processed/preprocessing_summary.csv

Cleaning steps
--------------
track : coerce to numeric, drop NaN rows, drop out-of-range coordinates,
        sort by timestamp, drop duplicate timestamps
nodes : coerce to numeric (row order is preserved; row index = node_id)
arcs  : coerce to integer (row order is preserved; row index = arc_id)
route : coerce to integer, drop ids outside [0, n_arcs)

nodes/arcs rows are never dropped, because their row index *is* the id.

The raw data folder is located automatically. Set PATHSNAP_DATA to override.
The evaluator-testing partition is never read.

Run from the repository root:  python preprocessing/preprocess.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_DIR / "data" / "processed"
PARTITIONS = ["train", "student_test"]


def find_raw_dir():
    candidates = [
        os.environ.get("PATHSNAP_DATA", ""),
        REPO_DIR / "data" / "raw",
        REPO_DIR / "data" / "raw" / "Student folder",
        REPO_DIR / "Student folder",
        REPO_DIR / "Student folder" / "Student folder",
        REPO_DIR.parent / "Student folder",
    ]
    for c in candidates:
        if c and (Path(c) / "train").is_dir():
            return Path(c)
    raise FileNotFoundError("Raw data not found. Set PATHSNAP_DATA to the folder containing train/ and student_test/.")


def read_raw(path, names):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names)
    return df.apply(pd.to_numeric, errors="coerce")


def clean_trip(trip_folder):
    tid = trip_folder.name
    track = read_raw(trip_folder / f"{tid}.track", ["longitude", "latitude", "timestamp"])
    nodes = read_raw(trip_folder / f"{tid}.nodes", ["longitude", "latitude"])
    arcs = read_raw(trip_folder / f"{tid}.arcs", ["from_node_id", "to_node_id"])
    route = read_raw(trip_folder / f"{tid}.route", ["edge_id"])

    n_raw = len(track)
    track = track.dropna()
    n_nan = n_raw - len(track)
    in_range = track["longitude"].between(-180, 180) & track["latitude"].between(-90, 90)
    n_range = int((~in_range).sum())
    track = track[in_range].sort_values("timestamp", kind="stable")
    n_dup = int(track["timestamp"].duplicated().sum())
    track = track.drop_duplicates("timestamp").reset_index(drop=True)

    if nodes.isna().any().any() or arcs.isna().any().any():
        raise ValueError(f"{tid}: NaN in nodes/arcs (ids are row indices, rows cannot be dropped)")
    arcs = arcs.astype(np.int64)
    bad_arcs = int(((arcs < 0) | (arcs >= len(nodes))).any(axis=1).sum())

    route = route.dropna().astype(np.int64)
    valid_route = route["edge_id"].between(0, len(arcs) - 1)
    n_bad_route = int((~valid_route).sum())
    route = route[valid_route].reset_index(drop=True)

    stats = dict(trip_id=tid, gps_points_raw=n_raw, gps_points=len(track), dropped_nan=n_nan,
                 dropped_out_of_range=n_range, dropped_duplicate_ts=n_dup, nodes=len(nodes), arcs=len(arcs),
                 invalid_arc_refs=bad_arcs, route_segments=len(route), dropped_route_ids=n_bad_route)
    return track, nodes, arcs, route, stats


def main():
    raw_dir = find_raw_dir()
    print("=" * 50)
    print("PATHSNAP DATA PREPROCESSING")
    print("Raw data :", raw_dir)
    print("Output   :", OUTPUT_DIR)
    print("=" * 50)

    summary = []
    for part in PARTITIONS:
        out = OUTPUT_DIR / part
        out.mkdir(parents=True, exist_ok=True)
        trips = sorted(p for p in (raw_dir / part).iterdir() if p.is_dir())
        for trip_folder in trips:
            track, nodes, arcs, route, stats = clean_trip(trip_folder)
            tid = trip_folder.name
            track.to_csv(out / f"{tid}_track.csv", index=False)
            nodes.to_csv(out / f"{tid}_nodes.csv", index=False)
            arcs.to_csv(out / f"{tid}_arcs.csv", index=False)
            route.to_csv(out / f"{tid}_route.csv", index=False)
            summary.append(dict(partition=part, **stats))
        print(f"{part}: {len(trips)} trips processed")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "preprocessing_summary.csv", index=False)

    print("\nPREPROCESSING SUMMARY")
    print(summary_df.groupby("partition")[["gps_points_raw", "gps_points", "dropped_nan", "dropped_out_of_range",
                                           "dropped_duplicate_ts", "invalid_arc_refs", "dropped_route_ids"]].sum())
    print("\nPREPROCESSING COMPLETED!")


if __name__ == "__main__":
    main()

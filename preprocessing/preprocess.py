
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Locate training folder
TRAIN_DIR = DATA_DIR / "Student folder" / "train"

if not TRAIN_DIR.exists():
    TRAIN_DIR = DATA_DIR / "train"

if not TRAIN_DIR.exists():
    raise FileNotFoundError("Training folder not found")

print("=" * 50)
print("PATHSNAP DATA PREPROCESSING")
print("=" * 50)

summary = []

for trip_folder in sorted(TRAIN_DIR.iterdir()):

    if not trip_folder.is_dir():
        continue

    trip_id = trip_folder.name

    track_file = trip_folder / f"{trip_id}.track"
    nodes_file = trip_folder / f"{trip_id}.nodes"
    arcs_file = trip_folder / f"{trip_id}.arcs"
    route_file = trip_folder / f"{trip_id}.route"

    # Read raw files without assuming headers
    track = pd.read_csv(
        track_file, sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        nodes_file, sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        arcs_file, sep=r"\s+", header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        route_file, sep=r"\s+", header=None,
        names=["edge_id"]
    )

    # Convert values to numeric
    track = track.apply(pd.to_numeric, errors="coerce")
    nodes = nodes.apply(pd.to_numeric, errors="coerce")
    arcs = arcs.apply(pd.to_numeric, errors="coerce")
    route = route.apply(pd.to_numeric, errors="coerce")

    # Remove rows with invalid values
    track = track.dropna()
    nodes = nodes.dropna()
    arcs = arcs.dropna()
    route = route.dropna()

    # Export processed files
    track.to_csv(OUTPUT_DIR / f"{trip_id}_track.csv", index=False)
    nodes.to_csv(OUTPUT_DIR / f"{trip_id}_nodes.csv", index=False)
    arcs.to_csv(OUTPUT_DIR / f"{trip_id}_arcs.csv", index=False)
    route.to_csv(OUTPUT_DIR / f"{trip_id}_route.csv", index=False)

    summary.append({
        "trip_id": trip_id,
        "gps_points": len(track),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_segments": len(route)
    })

# Save preprocessing summary
summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_DIR / "preprocessing_summary.csv",
    index=False
)

print("\nPREPROCESSING SUMMARY")
print(summary_df.head(10).to_string(index=False))

print("\nTOTAL TRAINING TRIPS:", len(summary_df))
print("TOTAL GPS POINTS:", summary_df["gps_points"].sum())
print("TOTAL NODES:", summary_df["nodes"].sum())
print("TOTAL ARCS:", summary_df["arcs"].sum())
print("TOTAL ROUTE SEGMENTS:", summary_df["route_segments"].sum())

print("\nProcessed files saved to:", OUTPUT_DIR)
print("PREPROCESSING COMPLETED!")

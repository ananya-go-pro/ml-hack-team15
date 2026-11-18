import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS = 6371000
SAMPLE_SPACING = 100.0


def project_coordinates(lon, lat, ref_lat):
    """Project longitude/latitude into local metre coordinates."""
    x = EARTH_RADIUS * np.radians(lon) * np.cos(np.radians(ref_lat))
    y = EARTH_RADIUS * np.radians(lat)
    return np.column_stack((x, y))


def point_to_segment_distance(point, start, end):
    """Calculate exact distance and projection fraction."""
    vector = end - start
    denominator = np.dot(vector, vector)

    if denominator == 0:
        fraction = 0.0
    else:
        fraction = np.dot(point - start, vector) / denominator
        fraction = np.clip(fraction, 0, 1)

    closest = start + fraction * vector
    distance = np.linalg.norm(point - closest)

    return distance, fraction


def generate_candidates(
    gps,
    nodes,
    arcs,
    max_candidates=8,
    search_radius=150
):
    """Generate road candidates using a densified segment spatial index."""

    ref_lat = gps["latitude"].mean()

    gps_xy = project_coordinates(
        gps["longitude"].to_numpy(),
        gps["latitude"].to_numpy(),
        ref_lat
    )

    node_xy = project_coordinates(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy(),
        ref_lat
    )

    # Preserve original .arcs row positions.
    arc_data = arcs.reset_index(drop=True).copy()
    arc_data["arc_id"] = np.arange(len(arc_data))

    source = arc_data["from_node_id"].to_numpy(dtype=int)
    target = arc_data["to_node_id"].to_numpy(dtype=int)

    valid = (
        (source >= 0) & (source < len(nodes)) &
        (target >= 0) & (target < len(nodes))
    )

    arc_data = arc_data.loc[valid].reset_index(drop=True)
    source = source[valid]
    target = target[valid]

    start_xy = node_xy[source]
    end_xy = node_xy[target]

    vectors = end_xy - start_xy
    lengths = np.linalg.norm(vectors, axis=1)

    # Sample each segment at intervals no greater than 100 metres.
    # Keep a mapping from every sample back to its original arc.
    sample_points = []
    sample_arc_indices = []

    for i, length in enumerate(lengths):
        count = max(1, int(np.ceil(length / SAMPLE_SPACING)))

        fractions = np.linspace(0, 1, count + 1)
        points = start_xy[i] + fractions[:, None] * vectors[i]

        sample_points.append(points)
        sample_arc_indices.extend([i] * len(points))

    sample_points = np.vstack(sample_points)
    sample_arc_indices = np.asarray(sample_arc_indices, dtype=int)

    tree = cKDTree(sample_points)

    # A point on a segment is at most 50m from a sample
    # when the maximum sample spacing is 100m.
    index_radius = search_radius + SAMPLE_SPACING / 2

    results = []

    for observation_id, point in enumerate(gps_xy):

        nearby_samples = tree.query_ball_point(
            point,
            index_radius
        )

        if not nearby_samples:
            continue

        # Multiple samples may belong to the same road segment.
        nearby_arcs = np.unique(
            sample_arc_indices[nearby_samples]
        )

        candidates = []

        for index in nearby_arcs:

            distance, fraction = point_to_segment_distance(
                point,
                start_xy[index],
                end_xy[index]
            )

            # Exact segment distance, not sample distance.
            if distance > search_radius:
                continue

            dx, dy = vectors[index]

            road_bearing = (
                np.degrees(np.arctan2(dx, dy)) + 360
            ) % 360

            arc = arc_data.iloc[index]

            candidates.append({
                "observation_id": observation_id,
                "arc_id": int(arc["arc_id"]),
                "from_node_id": int(source[index]),
                "to_node_id": int(target[index]),
                "distance_to_road_m": float(distance),
                "projection_fraction": float(fraction),
                "road_bearing": float(road_bearing)
            })

        candidates.sort(
            key=lambda item: item["distance_to_road_m"]
        )

        results.extend(candidates[:max_candidates])

    columns = [
        "observation_id",
        "arc_id",
        "from_node_id",
        "to_node_id",
        "distance_to_road_m",
        "projection_fraction",
        "road_bearing"
    ]

    return pd.DataFrame(results, columns=columns)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trip", default="00000001")
    parser.add_argument("--partition", default="train")
    parser.add_argument("--radius", type=float, default=150)
    parser.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args()

    base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    trip_dir = os.path.join(
        base,
        "Student folder",
        args.partition,
        args.trip
    )

    gps = pd.read_csv(
        os.path.join(trip_dir, args.trip + ".track"),
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        os.path.join(trip_dir, args.trip + ".nodes"),
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        os.path.join(trip_dir, args.trip + ".arcs"),
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    candidates = generate_candidates(
        gps,
        nodes,
        arcs,
        max_candidates=args.top_k,
        search_radius=args.radius
    )

    output_dir = os.path.join(base, "results")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        args.trip + "_candidates.csv"
    )

    candidates.to_csv(output_path, index=False)

    matched = candidates["observation_id"].nunique()

    print("\n========== CANDIDATE GENERATION ==========")
    print("GPS observations:", len(gps))
    print("Candidate records:", len(candidates))
    print("Matched observations:", matched)
    print("Unmatched observations:", len(gps) - matched)

    if not candidates.empty:
        print("Average candidates per observation:",
              len(candidates) / len(gps))
        print("Nearest candidate distance:",
              candidates["distance_to_road_m"].min())

    print("Saved:", output_path)


if __name__ == "__main__":
    main()

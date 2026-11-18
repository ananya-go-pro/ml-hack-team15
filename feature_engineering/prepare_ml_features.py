
import os
import argparse
import numpy as np
import pandas as pd


def prepare_features(trip_id, base):

    results = os.path.join(base, "results")

    gps_path = os.path.join(
        results, trip_id + "_gps_features.csv"
    )

    candidate_path = os.path.join(
        results, trip_id + "_candidates.csv"
    )

    gps = pd.read_csv(gps_path)
    candidates = pd.read_csv(candidate_path)

    # Attach GPS movement features to every road candidate.
    merged = candidates.merge(
        gps,
        on="observation_id",
        how="left",
        validate="many_to_one"
    )

    # Circular angular difference between GPS heading
    # and candidate road direction.
    difference = (
        merged["bearing"] - merged["road_bearing"]
    )

    merged["heading_difference"] = (
        (difference + 180) % 360 - 180
    ).abs()

    # Remove raw coordinates from the model feature table.
    # Keep identifiers for tracing predictions.
    feature_columns = [
        "observation_id",
        "arc_id",
        "from_node_id",
        "to_node_id",
        "distance_to_road_m",
        "projection_fraction",
        "road_bearing",
        "speed_mps",
        "distance_m",
        "time_diff",
        "bearing",
        "heading_change",
        "acceleration_mps2",
        "heading_difference"
    ]

    merged = merged[feature_columns]

    output_path = os.path.join(
        results, trip_id + "_ml_features.csv"
    )

    merged.to_csv(output_path, index=False)

    print("\n========== ML FEATURE PREPARATION ==========")
    print("Trip:", trip_id)
    print("Total candidate rows:", len(merged))
    print("Unique GPS observations:",
          merged["observation_id"].nunique())
    print("Unique candidate road segments:",
          merged["arc_id"].nunique())

    print("\nMissing values:")
    print(merged.isna().sum())

    print("\nFeature preview:")
    print(merged.head(10).to_string(index=False))

    print("\nSaved:", output_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trip",
        default="00000001"
    )

    args = parser.parse_args()

    base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    prepare_features(args.trip, base)


if __name__ == "__main__":
    main()


import os
import argparse
import numpy as np
import pandas as pd


def haversine(lon1, lat1, lon2, lat2):
    """Calculate geographical distance in metres."""
    radius = 6371000

    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def calculate_bearing(lon1, lat1, lon2, lat2):
    """Calculate direction of travel in degrees."""
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    bearing = np.degrees(np.arctan2(x, y))

    return (bearing + 360) % 360


def extract_gps_features(track_path):
    """Load GPS trajectory and extract movement features."""

    df = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    # Convert values to numeric and identify invalid observations.
    for col in ["longitude", "latitude", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (
        df[["longitude", "latitude", "timestamp"]].notna().all(axis=1)
        & df["longitude"].between(-180, 180)
        & df["latitude"].between(-90, 90)
        & (df["timestamp"] >= 0)
    )

    df = df.loc[valid].copy().reset_index(drop=True)

    # Preserve original observation order.
    df["observation_id"] = np.arange(len(df))

    # Time difference between consecutive observations.
    df["time_diff"] = df["timestamp"].diff()

    # Distance between consecutive GPS points.
    df["distance_m"] = haversine(
        df["longitude"].shift(),
        df["latitude"].shift(),
        df["longitude"],
        df["latitude"]
    )

    # Direction of movement.
    df["bearing"] = calculate_bearing(
        df["longitude"].shift(),
        df["latitude"].shift(),
        df["longitude"],
        df["latitude"]
    )

    # First observation has no previous point.
    df.loc[0, ["distance_m", "bearing"]] = np.nan

    # Calculate speed only for positive time differences.
    valid_time = df["time_diff"] > 0

    df["speed_mps"] = np.nan
    df.loc[valid_time, "speed_mps"] = (
        df.loc[valid_time, "distance_m"]
        / df.loc[valid_time, "time_diff"]
    )

    # Heading change between consecutive movements.
    angle_diff = df["bearing"].diff()

    df["heading_change"] = (
        (angle_diff + 180) % 360 - 180
    ).abs()

    # Acceleration based on change in estimated speed.
    df["acceleration_mps2"] = (
        df["speed_mps"].diff() / df["time_diff"]
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract GPS trajectory features."
    )

    parser.add_argument(
        "--trip",
        default="00000001",
        help="Trip ID to process"
    )

    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    track_path = os.path.join(
        base,
        "Student folder",
        "train",
        args.trip,
        args.trip + ".track"
    )

    output_dir = os.path.join(base, "results")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(track_path):
        raise FileNotFoundError(track_path)

    features = extract_gps_features(track_path)

    output_path = os.path.join(
        output_dir,
        args.trip + "_gps_features.csv"
    )

    features.to_csv(output_path, index=False)

    print("\n========== GPS FEATURE EXTRACTION ==========")
    print("Trip ID:", args.trip)
    print("Valid observations:", len(features))
    print("Average speed (m/s):", features["speed_mps"].mean())
    print("Maximum speed (m/s):", features["speed_mps"].max())
    print("Average distance (m):", features["distance_m"].mean())
    print("Missing values:\n", features.isna().sum())
    print("\nSaved:", output_path)
    print("\nFeature preview:")
    print(features.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

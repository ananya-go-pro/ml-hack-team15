
import os
import argparse
import pandas as pd
import folium


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", default="00000001")
    parser.add_argument("--partition", default="train")
    args = parser.parse_args()

    base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    trip_dir = os.path.join(
        base, "Student folder", args.partition, args.trip
    )

    gps = pd.read_csv(
        os.path.join(trip_dir, args.trip + ".track"),
        sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        os.path.join(trip_dir, args.trip + ".nodes"),
        sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        os.path.join(trip_dir, args.trip + ".arcs"),
        sep=r"\s+", header=None,
        names=["from_node_id", "to_node_id"]
    )

    candidates = pd.read_csv(
        os.path.join(
            base, "results", args.trip + "_candidates.csv"
        )
    )

    center = [
        gps["latitude"].mean(),
        gps["longitude"].mean()
    ]

    m = folium.Map(
        location=center,
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    # GPS trajectory layer
    gps_layer = folium.FeatureGroup(
        name="Raw GPS Trajectory",
        show=True
    )

    gps_points = gps[
        ["latitude", "longitude"]
    ].values.tolist()

    folium.PolyLine(
        gps_points,
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip="Observed GPS trajectory"
    ).add_to(gps_layer)

    # GPS observation markers (sampled for readability)
    for i in range(0, len(gps), 25):
        row = gps.iloc[i]

        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=3,
            color="blue",
            fill=True,
            popup=f"GPS observation: {i}"
        ).add_to(gps_layer)

    gps_layer.add_to(m)

    # Candidate road segment layer
    road_layer = folium.FeatureGroup(
        name="Candidate Road Segments",
        show=True
    )

    # Show top 3 nearest candidates per observation.
    top_candidates = (
        candidates.sort_values("distance_to_road_m")
        .groupby("observation_id")
        .head(3)
    )

    for row in top_candidates.itertuples(index=False):

        start = nodes.iloc[row.from_node_id]
        end = nodes.iloc[row.to_node_id]

        folium.PolyLine(
            [
                [start.latitude, start.longitude],
                [end.latitude, end.longitude]
            ],
            color="red",
            weight=2,
            opacity=0.5,
            tooltip=(
                f"Arc ID: {row.arc_id} | "
                f"Distance: {row.distance_to_road_m:.2f} m"
            )
        ).add_to(road_layer)

    road_layer.add_to(m)

    # Start and end markers
    folium.Marker(
        gps_points[0],
        tooltip="Trip Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    folium.Marker(
        gps_points[-1],
        tooltip="Trip End",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m.fit_bounds(gps_points)

    output_dir = os.path.join(base, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, args.trip + "_map.html"
    )

    m.save(output_path)

    print("Map generated successfully!")
    print("GPS observations:", len(gps))
    print("Candidate segments displayed:", len(top_candidates))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()


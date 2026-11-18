
import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx


def load_road_network(trip_dir, trip_id):

    nodes_path = os.path.join(
        trip_dir, trip_id + ".nodes"
    )

    arcs_path = os.path.join(
        trip_dir, trip_id + ".arcs"
    )

    nodes = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    return nodes, arcs


def analyse_network(nodes, arcs):

    graph = nx.DiGraph()

    # Add geographical nodes.
    for node_id, row in nodes.iterrows():
        graph.add_node(
            node_id,
            longitude=row["longitude"],
            latitude=row["latitude"]
        )

    # Add valid directed connections.
    valid_arcs = arcs[
        arcs["from_node_id"].between(0, len(nodes) - 1)
        & arcs["to_node_id"].between(0, len(nodes) - 1)
    ]

    for row in valid_arcs.itertuples(index=False):
        graph.add_edge(
            int(row.from_node_id),
            int(row.to_node_id)
        )

    # Network statistics.
    isolated_nodes = list(nx.isolates(graph))

    weak_components = list(
        nx.weakly_connected_components(graph)
    )

    strong_components = list(
        nx.strongly_connected_components(graph)
    )

    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())

    print("\n========== ROAD NETWORK ANALYSIS ==========")

    print("Number of nodes:", len(nodes))
    print("Number of arcs in file:", len(arcs))
    print("Valid directed connections:", len(valid_arcs))
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    print("Isolated nodes:", len(isolated_nodes))
    print("Weakly connected components:", len(weak_components))
    print("Strongly connected components:", len(strong_components))

    print("Average in-degree:", np.mean(list(in_degrees.values())))
    print("Average out-degree:", np.mean(list(out_degrees.values())))

    print("Largest weak component:",
          max(map(len, weak_components), default=0))

    print("Largest strong component:",
          max(map(len, strong_components), default=0))

    return graph


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

    trip_dir = os.path.join(
        base,
        "Student folder",
        "train",
        args.trip
    )

    nodes, arcs = load_road_network(
        trip_dir,
        args.trip
    )

    graph = analyse_network(nodes, arcs)

    output_dir = os.path.join(base, "results")
    os.makedirs(output_dir, exist_ok=True)

    nodes.to_csv(
        os.path.join(output_dir, args.trip + "_nodes.csv"),
        index=False
    )

    arcs.to_csv(
        os.path.join(output_dir, args.trip + "_arcs.csv"),
        index=False
    )

    print("\nRoad-network CSV files exported.")


if __name__ == "__main__":
    main()


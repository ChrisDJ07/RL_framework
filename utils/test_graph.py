import random
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from shapely import wkt as shapely_wkt

try:
    from utils.graph_utils import get_raw_osm_graph, to_training_graph
except ImportError:
    from graph_utils import get_raw_osm_graph, to_training_graph


# Reproducibility
SEED = 40
random.seed(SEED)
np.random.seed(SEED)

# python utils/test_graph.py --prebuilt-graphml data/la_trinidad_hazard_graph.graphml --use-existing-hazards --num-nodes 100 --min-nodes 100 --max-nodes 100

def create_base_graph(num_nodes=60, min_nodes=30, max_nodes=100, force_download=False):
    raw_graph = get_raw_osm_graph(min_nodes=min_nodes, force_download=force_download)
    return to_training_graph(raw_graph, num_nodes=num_nodes, min_nodes=min_nodes, max_nodes=max_nodes)


def create_base_graph_from_source(
    num_nodes=60,
    min_nodes=30,
    max_nodes=100,
    force_download=False,
    prebuilt_graphml_path="",
    use_existing_hazards=False,
    flood_attr="flood_hazard",
    landslide_attr="landslide_hazard",
    travel_time_attr="travel_time_min",
):
    if prebuilt_graphml_path:
        raw_graph = nx.read_graphml(prebuilt_graphml_path)
    else:
        raw_graph = get_raw_osm_graph(min_nodes=min_nodes, force_download=force_download)

    base_graph = to_training_graph(
        raw_graph,
        num_nodes=num_nodes,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        use_existing_hazards=use_existing_hazards,
        flood_attr=flood_attr,
        landslide_attr=landslide_attr,
        travel_time_attr=travel_time_attr,
    )
    return raw_graph, base_graph


def visualize_raw_osm_graph(raw_graph, title="Raw OSM Road Network (La Trinidad Area)"):
    # GraphML stores geometry as WKT strings; convert them back for OSMnx plotting.
    converted = 0
    if raw_graph.is_multigraph():
        edge_iter = raw_graph.edges(keys=True, data=True)
        for _, _, _, data in edge_iter:
            geom = data.get("geometry")
            if isinstance(geom, str):
                try:
                    data["geometry"] = shapely_wkt.loads(geom)
                    converted += 1
                except Exception:
                    data.pop("geometry", None)
    else:
        edge_iter = raw_graph.edges(data=True)
        for _, _, data in edge_iter:
            geom = data.get("geometry")
            if isinstance(geom, str):
                try:
                    data["geometry"] = shapely_wkt.loads(geom)
                    converted += 1
                except Exception:
                    data.pop("geometry", None)

    if converted > 0:
        print(f"Converted {converted} WKT edge geometries for plotting.")

    fig, ax = ox.plot_graph(
        raw_graph,
        node_size=8,
        edge_linewidth=0.8,
        edge_color="#1f2937",
        bgcolor="white",
        show=False,
        close=False,
    )
    ax.set_title(title)
    plt.show()


def visualize_graph(
    G,
    title="Graph Visualization",
    highlight_start=None,
    highlight_deliveries=None,
    node_size=180,
):
    pos = nx.get_node_attributes(G, "pos")
    plt.figure(figsize=(6, 6))

    edge_colors = []
    widths = []
    for _, _, data in G.edges(data=True):
        hazard_intensity = data["flood_score"] + data["landslide_score"]
        hazard_clamped = min(hazard_intensity / 2.0, 1.0)
        if data.get("blocked", False):
            edge_colors.append("black")
            widths.append(3)
        else:
            edge_colors.append((hazard_clamped, 0, 1 - hazard_clamped))
            widths.append(1 + hazard_clamped * 2)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths, alpha=0.95)

    base_nodes = [
        n
        for n in G.nodes()
        if n != highlight_start and (highlight_deliveries is None or n not in highlight_deliveries)
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=base_nodes,
        node_color="none",
        node_size=node_size,
        edgecolors="skyblue",
        linewidths=1.2,
        alpha=0.95,
    )

    if highlight_start is not None:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[highlight_start],
            node_color="green",
            node_size=node_size + 30,
            edgecolors="white",
            linewidths=1.0,
        )
    if highlight_deliveries is not None:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(highlight_deliveries),
            node_color="red",
            node_size=node_size + 20,
            edgecolors="white",
            linewidths=1.0,
        )

    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test/visualize raw and processed RL graph.")
    parser.add_argument("--num-nodes", type=int, default=200)
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--prebuilt-graphml", type=str, default="")
    parser.add_argument("--use-existing-hazards", action="store_true")
    parser.add_argument("--flood-attr", type=str, default="flood_hazard")
    parser.add_argument("--landslide-attr", type=str, default="landslide_hazard")
    parser.add_argument("--travel-time-attr", type=str, default="travel_time_min")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    prebuilt = str(Path(args.prebuilt_graphml)) if args.prebuilt_graphml else ""
    raw_graph, base_graph = create_base_graph_from_source(
        num_nodes=args.num_nodes,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        force_download=args.force_download,
        prebuilt_graphml_path=prebuilt,
        use_existing_hazards=args.use_existing_hazards,
        flood_attr=args.flood_attr,
        landslide_attr=args.landslide_attr,
        travel_time_attr=args.travel_time_attr,
    )

    print("Raw graph stats")
    print(f"Nodes={raw_graph.number_of_nodes()}, Edges={raw_graph.number_of_edges()}")
    print("Processed RL graph stats")
    print(f"Nodes={base_graph.number_of_nodes()}, Edges={base_graph.number_of_edges()}")

    edge_haz = [d.get("flood_score", 0.0) + d.get("landslide_score", 0.0) for _, _, d in base_graph.edges(data=True)]
    if edge_haz:
        print(
            f"Processed hazards: mean={np.mean(edge_haz):.4f}, min={np.min(edge_haz):.4f}, max={np.max(edge_haz):.4f}"
        )

    if not args.no_plot:
        print("Visualizing Raw OSM/Source Graph")
        visualize_raw_osm_graph(raw_graph, title="Raw Source Road Network")
        print("Visualizing Processed RL Graph")
        visualize_graph(base_graph, title="Processed RL Graph (Hazard Scores)")

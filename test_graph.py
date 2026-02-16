import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox

from graph_utils import get_raw_osm_graph, to_training_graph

# Reproducibility
SEED = 40
random.seed(SEED)
np.random.seed(SEED)

def create_base_graph(num_nodes=60, min_nodes=30, max_nodes=100, force_download=False):
    raw_graph = get_raw_osm_graph(min_nodes=min_nodes, force_download=force_download)
    return to_training_graph(raw_graph, num_nodes=num_nodes, min_nodes=min_nodes, max_nodes=max_nodes)


def visualize_raw_osm_graph(raw_graph, title="Raw OSM Road Network (MSU-IIT Area)"):
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

    # Draw Edges
    edge_colors = []
    widths = []

    for u, v, data in G.edges(data=True):
        hazard_intensity = data["flood_score"] + data["landslide_score"]

        # Normalize for color scaling
        hazard_clamped = min(hazard_intensity / 2.0, 1.0)

        if data.get("blocked", False):
            edge_colors.append("black")
            widths.append(3)
        else:
            # Blue (low hazard) → Red (high hazard)
            edge_colors.append((hazard_clamped, 0, 1 - hazard_clamped))
            widths.append(1 + hazard_clamped * 2)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths, alpha=0.95)

    # Draw most nodes as hollow circles to avoid covering edges.
    base_nodes = [
        n for n in G.nodes()
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

    # Keep special nodes filled for visibility.
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

    # Draw Labels
    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis("off")
    plt.show()
    
    
# Visualize the base graph with hazard scores only (no activation)
if __name__ == "__main__":
    raw_graph = get_raw_osm_graph(min_nodes=30, force_download=False)
    base_graph = to_training_graph(raw_graph, num_nodes=60, min_nodes=30, max_nodes=100)

    print("Visualizing Raw OSM Road Network")
    visualize_raw_osm_graph(raw_graph, title="Raw OSM Road Network (MSU-IIT Area)")

    print("Visualizing Base Graph (Hazard Scores Only)")
    visualize_graph(base_graph, title="Base Graph (No Activation)")

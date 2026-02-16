import random
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox


MSU_IIT_CENTER = (8.2280, 124.2452)  # (lat, lon)
DEFAULT_CACHE_PATH = Path("data") / "msu_iit_drive.graphml"


def get_raw_osm_graph(
    cache_path=DEFAULT_CACHE_PATH,
    center_point=MSU_IIT_CENTER,
    min_nodes=30,
    search_dist_m=1200,
    max_attempts=5,
    force_download=False,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_download:
        return ox.load_graphml(filepath=cache_path)

    G_raw = None
    dist = search_dist_m

    for _ in range(max_attempts):
        G_candidate = ox.graph_from_point(
            center_point,
            dist=dist,
            network_type="drive",
            simplify=True,
        )
        if G_candidate.number_of_nodes() >= min_nodes:
            G_raw = G_candidate
            break
        dist += 500

    if G_raw is None:
        raise RuntimeError("Could not fetch a road graph with enough nodes around MSU-IIT.")

    ox.save_graphml(G_raw, filepath=cache_path)
    return G_raw


def sample_edge_hazard_scores(low_hazard_edge_prob=0.8):
    # Most roads are safer; a minority are high-risk segments.
    if np.random.rand() < low_hazard_edge_prob:
        flood_score = np.random.uniform(0.0, 0.2)
        landslide_score = np.random.uniform(0.0, 0.2)
    else:
        flood_score = np.random.uniform(0.35, 1.0)
        landslide_score = np.random.uniform(0.35, 1.0)

    return float(flood_score), float(landslide_score)


def to_training_graph(raw_graph, num_nodes=60, min_nodes=30, max_nodes=100, low_hazard_edge_prob=0.8):
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
        G_work = ox.convert.to_undirected(raw_graph)
    else:
        G_work = ox.utils_graph.get_undirected(raw_graph)

    # Keep only the largest connected component.
    largest_cc = max(nx.connected_components(G_work), key=len)
    G_work = G_work.subgraph(largest_cc).copy()

    target_nodes = int(np.clip(num_nodes, min_nodes, max_nodes))
    if G_work.number_of_nodes() > target_nodes:
        seed_node = random.choice(list(G_work.nodes()))
        bfs_nodes = list(nx.bfs_tree(G_work, seed_node).nodes())[:target_nodes]
        G_work = G_work.subgraph(bfs_nodes).copy()

    if G_work.number_of_nodes() < min_nodes:
        raise RuntimeError(
            f"Road graph too small after trimming: {G_work.number_of_nodes()} nodes (min required {min_nodes})."
        )

    # Relabel OSM node IDs to contiguous indices [0..N-1] for one-hot state/action indexing.
    G_work = nx.convert_node_labels_to_integers(G_work, ordering="default")

    G = nx.Graph()
    for node, data in G_work.nodes(data=True):
        x = float(data.get("x", 0.0))  # longitude
        y = float(data.get("y", 0.0))  # latitude
        G.add_node(node, pos=np.array([x, y], dtype=float))

    for u, v, data in G_work.edges(data=True):
        length_m = float(data.get("length", 1.0))
        # Approx. travel time in minutes with nominal 30 km/h urban speed.
        base_time = (length_m / 8.33) / 60.0
        flood_score, landslide_score = sample_edge_hazard_scores(low_hazard_edge_prob=low_hazard_edge_prob)
        G.add_edge(
            u,
            v,
            length=length_m,
            base_time=max(base_time, 0.01),
            flood_score=flood_score,
            landslide_score=landslide_score,
        )

    return G

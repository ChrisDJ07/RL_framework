import random
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox


MSU_IIT_CENTER = (8.2280, 124.2452)  # (lat, lon)
DEFAULT_CACHE_PATH = Path("data") / "msu_iit_drive.graphml"

# Paper-grounded hazard class -> numeric score mapping.
FLOOD_CLASS_TO_SCORE = {
    "low": 0.2,       # 0-0.5m
    "moderate": 0.6,  # 0.5-1.5m
    "high": 1.0,      # >1.5m
}
LANDSLIDE_CLASS_TO_SCORE = {
    "very_low": 0.1,
    "low": 0.5,
    "moderate": 0.8,
    "high": 1.0,
}


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


def _sample_class(score_map, class_probs):
    classes = list(score_map.keys())
    probs = np.array([class_probs[c] for c in classes], dtype=float)
    probs = probs / probs.sum()
    chosen = np.random.choice(classes, p=probs)
    return chosen, float(score_map[chosen])


def sample_edge_hazard_scores(low_hazard_edge_prob=0.8):
    """Sample discrete hazard classes and mapped numeric scores per edge.

    `low_hazard_edge_prob` keeps backward compatibility with previous behavior:
    it biases sampling toward the lowest hazard class.
    """
    low_p = float(np.clip(low_hazard_edge_prob, 0.0, 1.0))
    rem_p = 1.0 - low_p

    flood_probs = {
        "low": low_p,
        "moderate": rem_p * 0.6,
        "high": rem_p * 0.4,
    }
    landslide_probs = {
        "very_low": low_p,
        "low": rem_p * 0.5,
        "moderate": rem_p * 0.3,
        "high": rem_p * 0.2,
    }

    flood_class, flood_score = _sample_class(FLOOD_CLASS_TO_SCORE, flood_probs)
    landslide_class, landslide_score = _sample_class(LANDSLIDE_CLASS_TO_SCORE, landslide_probs)
    return flood_score, landslide_score, flood_class, landslide_class


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
        flood_score, landslide_score, flood_class, landslide_class = sample_edge_hazard_scores(
            low_hazard_edge_prob=low_hazard_edge_prob
        )
        G.add_edge(
            u,
            v,
            length=length_m,
            base_time=max(base_time, 0.01),
            flood_score=flood_score,
            landslide_score=landslide_score,
            flood_class=flood_class,
            landslide_class=landslide_class,
        )

    return G

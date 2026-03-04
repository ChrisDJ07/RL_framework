"""
Utility functions for graph manipulation and hazard scoring. 
Includes OSM graph fetching, conversion to training format, and edge hazard sampling. 
Designed for use in hazard-aware routing experiments.
"""

import random
from pathlib import Path

import networkx as nx
import numpy as np
try:
    import osmnx as ox
except ImportError:
    ox = None


MSU_IIT_CENTER = (8.2280, 124.2452)  # Approximate center of MSU-IIT campus (latitude, longitude).
DEFAULT_CACHE_PATH = Path("data") / "msu_iit_drive.graphml"  # Local cache for raw OSM graph.

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
    if ox is None:
        raise ImportError(
            "osmnx is required to download OSM data. Install osmnx or use a prebuilt GraphML path."
        )

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
    """Sample discrete hazard classes and mapped numeric scores per edge."""
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


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_undirected_fallback(raw_graph):
    """Convert arbitrary directed/multi graph to undirected simple graph."""
    G_u = nx.Graph()
    for n, d in raw_graph.nodes(data=True):
        G_u.add_node(n, **dict(d))

    if raw_graph.is_multigraph():
        edge_iter = raw_graph.edges(keys=True, data=True)
        for u, v, _, d in edge_iter:
            payload = dict(d)
            cand_len = _safe_float(payload.get("length", float("inf")), float("inf"))
            if G_u.has_edge(u, v):
                prev_len = _safe_float(G_u[u][v].get("length", float("inf")), float("inf"))
                if cand_len >= prev_len:
                    continue
            G_u.add_edge(u, v, **payload)
    else:
        for u, v, d in raw_graph.edges(data=True):
            payload = dict(d)
            cand_len = _safe_float(payload.get("length", float("inf")), float("inf"))
            if G_u.has_edge(u, v):
                prev_len = _safe_float(G_u[u][v].get("length", float("inf")), float("inf"))
                if cand_len >= prev_len:
                    continue
            G_u.add_edge(u, v, **payload)

    return G_u


def to_training_graph(
    raw_graph,
    num_nodes=60,
    min_nodes=30,
    max_nodes=100,
    low_hazard_edge_prob=0.8,
    use_existing_hazards=False,
    flood_attr="flood_hazard",
    landslide_attr="landslide_hazard",
    travel_time_attr="travel_time_min",
):
    try:
        if ox is not None and hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
            G_work = ox.convert.to_undirected(raw_graph)
        elif ox is not None:
            G_work = ox.utils_graph.get_undirected(raw_graph)
        else:
            G_work = _to_undirected_fallback(raw_graph)
    except Exception:
        G_work = _to_undirected_fallback(raw_graph)

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

    # Relabel OSM IDs to contiguous indices for easier state/action indexing.
    G_work = nx.convert_node_labels_to_integers(G_work, ordering="default")

    G = nx.Graph()
    for node, data in G_work.nodes(data=True):
        x = _safe_float(data.get("x", 0.0), 0.0)  # longitude
        y = _safe_float(data.get("y", 0.0), 0.0)  # latitude
        G.add_node(node, pos=np.array([x, y], dtype=float))

    for u, v, data in G_work.edges(data=True):
        length_m = max(_safe_float(data.get("length", 1.0), 1.0), 1e-3)

        if use_existing_hazards:
            flood_score = float(np.clip(_safe_float(data.get(flood_attr, 0.0), 0.0), 0.0, 1.0))
            landslide_score = float(np.clip(_safe_float(data.get(landslide_attr, 0.0), 0.0), 0.0, 1.0))
            base_time_raw = _safe_float(data.get(travel_time_attr, np.nan), np.nan)
            # Fallback to nominal speed (30 km/h) if travel time is missing.
            base_time = base_time_raw if np.isfinite(base_time_raw) and base_time_raw > 0 else (length_m / 8.33) / 60.0
            flood_class = str(data.get("flood_class", "from_source"))
            landslide_class = str(data.get("landslide_class", "from_source"))
        else:
            # Approx. base travel time in minutes using nominal 30 km/h.
            base_time = (length_m / 8.33) / 60.0
            flood_score, landslide_score, flood_class, landslide_class = sample_edge_hazard_scores(
                low_hazard_edge_prob=low_hazard_edge_prob
            )

        edge_payload = dict(
            length=length_m,
            base_time=max(float(base_time), 0.01),
            flood_score=flood_score,
            landslide_score=landslide_score,
            flood_class=flood_class,
            landslide_class=landslide_class,
        )

        # If multiple parallel edges collapse into one undirected edge, keep the faster one.
        if G.has_edge(u, v):
            if edge_payload["base_time"] >= G[u][v].get("base_time", float("inf")):
                continue
        G.add_edge(u, v, **edge_payload)

    return G

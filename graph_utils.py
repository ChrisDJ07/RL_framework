import random
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox


MSU_IIT_CENTER = (8.2280, 124.2452)  # Approximate center of MSU-IIT campus (latitude, longitude).
DEFAULT_CACHE_PATH = Path("data") / "msu_iit_drive.graphml" # Local cache for the raw OSM graph to avoid repeated downloads during development.

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
    search_dist_m=1200, # Initial search radius in meters
    max_attempts=5, # Max attempts to fetch a graph with enough nodes by expanding the search radius.
    force_download=False, # If True, ignore cached graph and fetch fresh data from OSM (useful if you want to update the graph or if the cache is corrupted).
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the cache directory exists.

    if cache_path.exists() and not force_download: # Load from cache if available and not forcing a download.
        return ox.load_graphml(filepath=cache_path)

    G_raw = None
    dist = search_dist_m

    # Attempt to fetch a graph with enough nodes, expanding the search radius if needed.
    for _ in range(max_attempts):
        G_candidate = ox.graph_from_point(
            center_point,
            dist=dist,
            network_type="drive",
            simplify=True, # Keep the graph simplified (merge consecutive nodes on straight roads) to reduce complexity while preserving connectivity and geometry.
        )
        if G_candidate.number_of_nodes() >= min_nodes: # Check if the fetched graph has enough nodes to work with.
            G_raw = G_candidate
            break
        dist += 500

    if G_raw is None:
        raise RuntimeError("Could not fetch a road graph with enough nodes around MSU-IIT.")

    ox.save_graphml(G_raw, filepath=cache_path) # Cache the raw graph for future use to avoid repeated downloads during development.
    return G_raw


# Sample hazard classes for an edge based on specified probabilities, and return the chosen class along with its numeric score.
def _sample_class(score_map, class_probs):
    classes = list(score_map.keys()) # Get the list of hazard classes from the score mapping.
    probs = np.array([class_probs[c] for c in classes], dtype=float) # Get the corresponding probabilities for each class in the same order.
    probs = probs / probs.sum() # Normalize probabilities to ensure they sum to 1, in case of any rounding issues.
    chosen = np.random.choice(classes, p=probs) # Sample a class based on the specified probabilities.
    return chosen, float(score_map[chosen]) # Return the chosen class and its corresponding numeric score as a float.


def sample_edge_hazard_scores(low_hazard_edge_prob=0.8):
    """Sample discrete hazard classes and mapped numeric scores per edge.

    `low_hazard_edge_prob` keeps backward compatibility with previous behavior:
    it biases sampling toward the lowest hazard class.
    """
    low_p = float(np.clip(low_hazard_edge_prob, 0.0, 1.0)) # Probability of sampling the lowest hazard class (e.g., "low" for flood, "very_low" for landslide).
    rem_p = 1.0 - low_p # Remaining probability to be distributed among the higher hazard classes.

    # Distribute the remaining probability among the higher hazard classes according to a simple heuristic (e.g., 60% moderate, 40% high for flood; 50% low, 30% moderate, 20% high for landslide).
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

    # Sample classes and scores for both flood and landslide hazards using the defined probabilities and score mappings.
    flood_class, flood_score = _sample_class(FLOOD_CLASS_TO_SCORE, flood_probs)
    landslide_class, landslide_score = _sample_class(LANDSLIDE_CLASS_TO_SCORE, landslide_probs)
    return flood_score, landslide_score, flood_class, landslide_class


# Convert the raw OSM graph into a training graph with node features (positions) and edge features (length, base travel time, hazard scores and classes). This includes:
# - Keeping only the largest connected component to ensure the graph is fully navigable.
# - Trimming the graph to a target number of nodes using BFS from a random seed node, while ensuring it doesn't go below a minimum node count.
# - Relabeling nodes to contiguous integer indices for easier indexing in state/action representations.
# - Adding edge attributes for length, base travel time (estimated from length), and sampled hazard scores and classes for flood and landslide.
def to_training_graph(raw_graph, num_nodes=60, min_nodes=30, max_nodes=100, low_hazard_edge_prob=0.8):
    
    # Convert to undirected graph for connectivity analysis and pathfinding, as OSM data may contain one-way streets but we want to ensure the graph is fully navigable in both directions for training. 
    # Depending on the version of OSMnx, the method to convert to undirected may differ, so we check for the appropriate function and fall back to a compatible method if needed.
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
        G_work = ox.convert.to_undirected(raw_graph)
    else:
        G_work = ox.utils_graph.get_undirected(raw_graph)

    # Keep only the largest connected component.
    # This ensures that the resulting graph is fully navigable and doesn't contain isolated nodes or small disconnected subgraphs that could complicate training and evaluation.
    largest_cc = max(nx.connected_components(G_work), key=len) # Find the largest connected component in the undirected graph.
    G_work = G_work.subgraph(largest_cc).copy() # Create a subgraph containing only the nodes in the largest connected component and make a copy to ensure we have a mutable graph for further processing.

    # Trim the graph to a target number of nodes using BFS from a random seed node, while ensuring it doesn't go below a minimum node count. This helps to control the size of the graph for training while maintaining connectivity and a realistic structure.
    target_nodes = int(np.clip(num_nodes, min_nodes, max_nodes)) # Ensure the target number of nodes is within the specified min and max bounds.
    if G_work.number_of_nodes() > target_nodes:
        seed_node = random.choice(list(G_work.nodes())) # Randomly select a seed node from the graph to start BFS. This introduces some variability in the resulting subgraph across different runs, which can be beneficial for training.
        bfs_nodes = list(nx.bfs_tree(G_work, seed_node).nodes())[:target_nodes]
        G_work = G_work.subgraph(bfs_nodes).copy()

    if G_work.number_of_nodes() < min_nodes:
        raise RuntimeError(
            f"Road graph too small after trimming: {G_work.number_of_nodes()} nodes (min required {min_nodes})."
        )

    # Relabel OSM node IDs to contiguous indices [0..N-1] for one-hot state/action indexing.
    # OSM node IDs can be large and non-contiguous, which can complicate indexing in state and action representations. Relabeling to contiguous integer indices simplifies this and ensures that node features and edge features can be easily accessed using array indexing.
    G_work = nx.convert_node_labels_to_integers(G_work, ordering="default")

    G = nx.Graph() # Create a new graph to store the processed training graph with the desired node and edge attributes.
    
    # Add node attributes (positions) and edge attributes (length, base travel time, hazard scores and classes) to the new graph. This includes:
    # - Node attributes: 'pos' containing the (longitude, latitude) coordinates as a numpy array for each node.
    # - Edge attributes: 'length' (in meters), 'base_time' (estimated travel time in minutes based on length), 'flood_score', 'landslide_score' (numeric hazard scores), and 'flood_class', 'landslide_class' (discrete hazard classes).
    
    for node, data in G_work.nodes(data=True):
        x = float(data.get("x", 0.0))  # longitude
        y = float(data.get("y", 0.0))  # latitude
        G.add_node(node, pos=np.array([x, y], dtype=float)) # Add the node to the new graph with its position as a numpy array for easier handling in training.

    for u, v, data in G_work.edges(data=True):
        length_m = float(data.get("length", 1.0))
        # Approx. travel time in minutes with nominal 30 km/h urban speed.
        base_time = (length_m / 8.33) / 60.0 # Convert length to travel time in minutes (length in meters divided by speed in m/s, then converted to minutes).
        flood_score, landslide_score, flood_class, landslide_class = sample_edge_hazard_scores(
            low_hazard_edge_prob=low_hazard_edge_prob
        )
        G.add_edge(
            u, # Source node index.
            v, # Target node index.
            length=length_m,
            base_time=max(base_time, 0.01),
            flood_score=flood_score,
            landslide_score=landslide_score,
            flood_class=flood_class,
            landslide_class=landslide_class,
        )

    return G

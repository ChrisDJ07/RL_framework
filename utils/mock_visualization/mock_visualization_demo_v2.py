"""
mock_visualization_demo.py
==========================

Self-contained, barebones demo for model-to-visualization wiring.

Why this file exists
--------------------
A simple example that demonstrates, end-to-end, how a trained
RL checkpoint can be queried with route inputs (depot, delivery stops,
rain intensity, route type) and transformed into a visualization-friendly
response payload.

This file intentionally keeps everything in one place:
1) Inline config (based on stage_200)
2) Inline mock request
3) Model + graph loading
4) Inference run
5) JSON response formatting

Important clarification
-----------------------
The graph is NOT "stored inside the neural network weights". The .pt checkpoint
stores model parameters. In the training pipeline, checkpoints also include
"base_graph_node_link", which is a serialized graph snapshot.

So at inference, graph loading can happen in two ways:
- Preferred: load graph snapshot from checkpoint payload
- Fallback: rebuild graph from config fields (graph section)

Config fields needed for inference:
---------------------------------------------
Required (this demo uses them):
- seed
- graph
- environment
- hazard
- reward
- model

Not required for pure inference:
- replay
- training
- evaluation
- paths
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch

import rl_routing_wCUDA_wCheckP as rl


# ============================================================================
# Inline config (stage_200-based, inference-relevant fields only)
# ============================================================================
DEMO_CONFIG_STAGE_200 = {
    "seed": 40,
    "graph": {
        "num_nodes": 200,
        "min_nodes": 200,
        "max_nodes": 200,
        "force_download": False,
        "prebuilt_graphml_path": "data/staged_subgraphs/selected_subgraph_n200.graphml",
        "use_existing_hazards": True,
        "flood_attr": "flood_hazard",
        "landslide_attr": "landslide_hazard",
        "travel_time_attr": "travel_time_min",
    },
    "environment": {
        "num_deliveries": 2,
        "min_max_steps": 220,
        "max_steps_multiplier": 0.7,
        "episode_time_scale": 8.0,
    },
    "hazard": {
        "rain_levels": {
            "RI1": {
                "speed_mult": 1.0,
                "flood_block_threshold": 1.1,
                "flood_block_prob": 0.0,
                "landslide_block_threshold": 1.1,
                "landslide_block_prob": 0.0,
            },
            "RI2": {
                "speed_mult": 0.9,
                "flood_block_threshold": 1.0,
                "flood_block_prob": 0.3,
                "landslide_block_threshold": 0.8,
                "landslide_block_prob": 0.05,
            },
            "RI3": {
                "speed_mult": 0.85,
                "flood_block_threshold": 0.6,
                "flood_block_prob": 0.6,
                "landslide_block_threshold": 0.8,
                "landslide_block_prob": 0.15,
            },
            # Included for model state-dimension compatibility.
            # Even if inactive, these keys affect rain one-hot dimensionality.
            "RI4": {
                "speed_mult": 0.4,
                "flood_block_threshold": 0.6,
                "flood_block_prob": 0.9,
                "landslide_block_threshold": 0.5,
                "landslide_block_prob": 0.3,
            },
            "RI5": {
                "speed_mult": 0.2,
                "flood_block_threshold": 0.2,
                "flood_block_prob": 1.0,
                "landslide_block_threshold": 0.5,
                "landslide_block_prob": 1.0,
            },
        },
        "active_rain_keys": ["RI1"],
        "flood_time_weight": 0.0,
        "landslide_time_weight": 0.0,
        "high_risk_flood_threshold": 1.1,
        "high_risk_landslide_threshold": 1.1,
        "max_neighbor_slots": 4,
        "neighbor_feature_dim": 7,
    },
    "reward": {
        "delivery": 60.0,
        "mission_success": 120.0,
        "k_progress": 10.0,
        "hazard_lambda": 0.0,
        "w_flood": 0.0,
        "w_landslide": 0.0,
        "eta_time": 0.2,
        "step_cost": 0.05,
        "penalty_timeout": -120.0,
        "penalty_blockage": -120.0,
        "penalty_incomplete_per_delivery": -20.0,
    },
    "model": {
        "hidden_sizes": [128, 64],
        "node_embedding_dim": 32,
    },
}

# typically present in training config but unnecessary for inference-only use.
TRAINING_ONLY_CONFIG_SECTIONS = ["replay", "training", "evaluation", "paths"]


# ============================================================================
# Inline mock request payload
# ============================================================================
MOCK_REQUEST = {
    "id": str(uuid.uuid4()),
    "routeType": "balanced",   # safe | balanced | fast (informational in this demo)
    "rainIntensity": "RI1",    # RI1..RI5
    "depot": {
        "id": "depot",
        "location": {"lat": 16.4885651, "lng": 120.5945807},
        "label": "Mock Depot",
    },
    "deliveryStops": [
        {
            "id": "stop-1",
            "location": {"lat": 16.4723677, "lng": 120.5996959},
            "label": "Mock Stop 1",
        },
        {
            "id": "stop-2",
            "location": {"lat": 16.4743090, "lng": 120.6114882},
            "label": "Mock Stop 2",
        },
    ],
}


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distance in meters between two lon/lat points."""
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def find_nearest_node(
    node_pos: Dict[int, np.ndarray],
    lat: float,
    lng: float,
    excluded: Optional[Set[int]] = None,
) -> int:
    """
    Map a lat/lng query point to nearest graph node.

    `excluded` prevents duplicate mapping (e.g., depot and stop mapping to same node).
    """
    excluded = excluded or set()
    best_node = None
    best_dist = float("inf")
    for n, pos in node_pos.items():
        if n in excluded:
            continue
        lon_n, lat_n = float(pos[0]), float(pos[1])
        d = haversine_m(lng, lat, lon_n, lat_n)
        if d < best_dist:
            best_dist = d
            best_node = n
    if best_node is None:
        raise RuntimeError("Could not map coordinates to a graph node.")
    return int(best_node)


def node_to_latlng(node_pos: Dict[int, np.ndarray], node_id: int) -> Dict[str, float]:
    """Convert node id back to {lat, lng}."""
    lon, lat = node_pos[int(node_id)]
    return {"lat": float(lat), "lng": float(lon)}


def load_env_and_model(cfg: Dict, checkpoint_path: Path) -> Tuple[rl.HazardRoutingEnv, rl.DQN, str]:
    """
    Initialize runtime config, environment, and model from checkpoint.

    Returns:
    - env
    - model
    - graph_source: checkpoint_snapshot | config_graph_build
    """
    rl.set_seed(cfg["seed"])
    rl.apply_runtime_config(cfg)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint must contain key 'model_state_dict'.")

    graph_source = "config_graph_build"
    if "base_graph_node_link" in checkpoint:
        # Best-case: exact graph snapshot used during training is embedded.
        base_graph = nx.node_link_graph(checkpoint["base_graph_node_link"], edges="edges")
        graph_source = "checkpoint_snapshot"
    else:
        # Fallback: reconstruct graph from config fields.
        graph_cfg = cfg["graph"]
        base_graph = rl.create_base_graph(
            num_nodes=int(graph_cfg["num_nodes"]),
            min_nodes=int(graph_cfg["min_nodes"]),
            max_nodes=int(graph_cfg["max_nodes"]),
            force_download=bool(graph_cfg.get("force_download", False)),
            prebuilt_graphml_path=str(graph_cfg.get("prebuilt_graphml_path", "") or ""),
            use_existing_hazards=bool(graph_cfg.get("use_existing_hazards", False)),
            flood_attr=str(graph_cfg.get("flood_attr", "flood_hazard")),
            landslide_attr=str(graph_cfg.get("landslide_attr", "landslide_hazard")),
            travel_time_attr=str(graph_cfg.get("travel_time_attr", "travel_time_min")),
        )

    env = rl.HazardRoutingEnv(
        base_graph,
        num_deliveries=int(cfg["environment"]["num_deliveries"]),
        env_cfg=cfg["environment"],
        reward_cfg=cfg["reward"],
    )

    model_cfg = cfg["model"]
    model = rl.DQN(
        env.state_dim,
        env.action_dim,
        num_nodes=env.num_nodes,
        num_delivery_slots=env.num_deliveries,
        hidden_sizes=tuple(model_cfg["hidden_sizes"]),
        node_embedding_dim=int(model_cfg.get("node_embedding_dim", 16)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return env, model, graph_source


def map_request_to_nodes(env: rl.HazardRoutingEnv, request: Dict) -> Tuple[int, List[int]]:
    """Map depot + stop coordinates to nearest unique node ids."""
    used = set()
    depot_loc = request["depot"]["location"]
    depot_node = find_nearest_node(
        env.node_pos,
        lat=float(depot_loc["lat"]),
        lng=float(depot_loc["lng"]),
        excluded=used,
    )
    used.add(depot_node)

    stop_nodes: List[int] = []
    for stop in request["deliveryStops"]:
        loc = stop["location"]
        node = find_nearest_node(
            env.node_pos,
            lat=float(loc["lat"]),
            lng=float(loc["lng"]),
            excluded=used,
        )
        used.add(node)
        stop_nodes.append(node)

    if not stop_nodes:
        raise ValueError("Request must include at least one delivery stop.")
    return depot_node, stop_nodes


def initialize_env_for_request(
    env: rl.HazardRoutingEnv,
    rain_intensity: str,
    depot_node: int,
    stop_nodes: Iterable[int],
) -> None:
    """
    Override environment episode state based on external request.

    This mimics what a REST endpoint would do after parsing request payload.
    """
    if rain_intensity not in rl.RAIN_KEYS:
        raise ValueError(f"Unsupported rain_intensity '{rain_intensity}'. Valid: {rl.RAIN_KEYS}")

    env.G = rl.activate_hazards(env.base_graph, rain_intensity)
    env.rain_onehot = np.zeros(env.rain_dim, dtype=float)
    env.rain_onehot[rl.RAIN_KEYS.index(rain_intensity)] = 1.0

    stop_nodes = [int(x) for x in stop_nodes]
    env.num_deliveries = len(stop_nodes)
    env.current_node = int(depot_node)
    env.delivery_nodes = set(stop_nodes)
    env.completed = set()
    env.total_time = 0.0
    env.total_hazard = 0.0
    env.steps = 0


def run_inference(env: rl.HazardRoutingEnv, model: rl.DQN, epsilon: float = 0.0) -> Dict:
    """
    Single-episode policy rollout for the requested depot/stops/rain settings.

    epsilon=0.0 means deterministic greedy inference.
    """
    state = env._get_state()
    done = False
    total_reward = 0.0
    route_nodes = [int(env.current_node)]
    route_edges = []
    reason = None

    while not done:
        mask = env.get_action_mask()
        action = rl.select_action(model, state, mask, epsilon=epsilon)
        if action is None:
            total_reward += env.failure_penalty("blockage")
            reason = "no_valid_action"
            break

        prev_node = int(env.current_node)
        next_state, reward, done, info = env.step(action)
        next_node = int(env.current_node)

        edge = env.G[prev_node][next_node]
        route_edges.append(
            {
                "u": prev_node,
                "v": next_node,
                "length_m": float(edge.get("length", 0.0)),
                "travel_time_min": float(edge.get("travel_time", 0.0) or 0.0),
                "flood_score": float(edge.get("flood_score", 0.0)),
                "landslide_score": float(edge.get("landslide_score", 0.0)),
                "blocked": bool(edge.get("blocked", False)),
            }
        )

        route_nodes.append(next_node)
        total_reward += float(reward)
        state = next_state
        if done:
            reason = info.get("termination_reason")

    success = len(env.completed) == len(env.delivery_nodes)
    if reason is None:
        reason = "success" if success else "unknown"

    return {
        "success": bool(success),
        "terminationReason": reason,
        "totalReward": float(total_reward),
        "steps": int(len(route_edges)),
        "routeNodes": route_nodes,
        "routeEdges": route_edges,
    }


def to_route_response(
    request: Dict,
    env: rl.HazardRoutingEnv,
    inference: Dict,
    graph_source: str,
) -> Dict:
    """
    Convert rollout output into frontend/API-friendly route response.

    Shape is compatible with your visualization-side routing types.
    """
    segments = []
    total_dist = 0.0
    total_time_s = 0.0
    hz_scores = []

    for i, e in enumerate(inference["routeEdges"], start=1):
        u, v = int(e["u"]), int(e["v"])
        coords = [node_to_latlng(env.node_pos, u), node_to_latlng(env.node_pos, v)]
        dist_m = float(e["length_m"])
        tt_s = float(e["travel_time_min"]) * 60.0
        hz = float(e["flood_score"] + e["landslide_score"])
        total_dist += dist_m
        total_time_s += tt_s
        hz_scores.append(hz)
        segments.append(
            {
                "id": f"segment-{i}",
                "coordinates": coords,
                "distanceMeters": dist_m,
                "travelTimeSeconds": tt_s,
                "hazardScore": hz,
            }
        )

    average_hazard = float(np.mean(hz_scores)) if hz_scores else 0.0

    depot = request["depot"]
    delivery_stops = [
        {
            "id": depot.get("id", "depot"),
            "location": depot["location"],
            "sequence": 1,
            "label": depot.get("label", "Depot"),
        }
    ]
    for idx, stop in enumerate(request["deliveryStops"], start=2):
        delivery_stops.append(
            {
                "id": stop.get("id", f"stop-{idx - 1}"),
                "location": stop["location"],
                "sequence": idx,
                "label": stop.get("label", f"Stop {idx - 1}"),
            }
        )

    return {
        "id": request["id"],
        "type": request["routeType"],  # For now all route types use same model.
        "rainIntensity": request["rainIntensity"],
        "graphSource": graph_source,
        "success": inference["success"],
        "terminationReason": inference["terminationReason"],
        "segments": segments,
        "depot": depot["location"],
        "deliveryStops": delivery_stops,
        "totalDistanceMeters": total_dist,
        "totalTravelTimeSeconds": total_time_s,
        "averageHazardScore": average_hazard,
        "debug": {
            "steps": inference["steps"],
            "routeNodes": inference["routeNodes"],
            "completedDeliveries": len(env.completed),
            "requiredDeliveries": len(env.delivery_nodes),
            "totalReward": inference["totalReward"],
            "inferenceConfigKeysUsed": list(DEMO_CONFIG_STAGE_200.keys()),
            "trainingOnlyConfigSectionsNotRequired": TRAINING_ONLY_CONFIG_SECTIONS,
        },
    }


def parse_args():
    """Small CLI for easy experimentation."""
    parser = argparse.ArgumentParser(description="Self-contained RL visualization integration demo.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/staged_training/stage_200/best_model.pt",
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--rain-intensity",
        type=str,
        default="",
        help="Optional override for request rainIntensity (e.g. RI1, RI2, RI3).",
    )
    parser.add_argument(
        "--route-type",
        type=str,
        default="",
        help="Optional override for request routeType (safe|balanced|fast).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Inference epsilon. Use 0.0 for deterministic policy.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output path for saving response JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)

    cfg = copy.deepcopy(DEMO_CONFIG_STAGE_200)
    request = copy.deepcopy(MOCK_REQUEST)
    if args.rain_intensity:
        request["rainIntensity"] = args.rain_intensity
    if args.route_type:
        request["routeType"] = args.route_type

    env, model, graph_source = load_env_and_model(cfg=cfg, checkpoint_path=checkpoint_path)
    depot_node, stop_nodes = map_request_to_nodes(env, request)
    initialize_env_for_request(
        env=env,
        rain_intensity=request["rainIntensity"],
        depot_node=depot_node,
        stop_nodes=stop_nodes,
    )

    inference = run_inference(env=env, model=model, epsilon=float(args.epsilon))
    response = to_route_response(request=request, env=env, inference=inference, graph_source=graph_source)

    print(json.dumps(response, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        print(f"\nSaved response JSON to: {out_path}")


if __name__ == "__main__":
    main()

# python mock_visualization/mock_visualization_demo.py --checkpoint-path checkpoints/staged_training/stage_200/best_model.pt --epsilon 0.0
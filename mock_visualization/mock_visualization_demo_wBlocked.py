"""
Barebones end-to-end demo: load RL checkpoint + config, build/load graph,
accept mock route request (depot, delivery stops, rain intensity, route type),
run policy inference, and print a JSON response.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch

import rl_routing_wCUDA_wCheckP as rl


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
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
    lon, lat = node_pos[int(node_id)]
    return {"lat": float(lat), "lng": float(lon)}


def build_auto_mock_request(env: rl.HazardRoutingEnv, rain_intensity: str, route_type: str) -> Dict:
    # Deterministic, simple sample from existing graph nodes.
    nodes = sorted(list(env.base_graph.nodes()))
    if len(nodes) < 3:
        raise RuntimeError("Need at least 3 nodes for depot + 2 deliveries.")

    depot_node = nodes[0]
    stop_nodes = [nodes[len(nodes) // 3], nodes[(2 * len(nodes)) // 3]]

    return {
        "id": str(uuid.uuid4()),
        "routeType": route_type,
        "rainIntensity": rain_intensity,
        "depot": {
            "id": "depot",
            "location": node_to_latlng(env.node_pos, depot_node),
            "label": "Mock Depot",
        },
        "deliveryStops": [
            {
                "id": f"stop-{i + 1}",
                "location": node_to_latlng(env.node_pos, node_id),
                "label": f"Mock Stop {i + 1}",
            }
            for i, node_id in enumerate(stop_nodes)
        ],
    }


def load_request_from_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env_and_model(config_path: Path, checkpoint_path: Path):
    cfg = rl.load_config(str(config_path))
    rl.set_seed(cfg["seed"])
    rl.apply_runtime_config(cfg)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint must contain a 'model_state_dict' payload.")

    graph_source = "config_graph_build"
    if "base_graph_node_link" in checkpoint:
        base_graph = nx.node_link_graph(checkpoint["base_graph_node_link"], edges="edges")
        graph_source = "checkpoint_snapshot"
    else:
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

    env_cfg = cfg["environment"]
    reward_cfg = cfg["reward"]
    model_cfg = cfg["model"]

    env = rl.HazardRoutingEnv(
        base_graph,
        num_deliveries=int(env_cfg["num_deliveries"]),
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
    )

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
):
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


def run_inference(
    env: rl.HazardRoutingEnv,
    model: rl.DQN,
    epsilon: float = 0.0,
) -> Dict:
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


def to_route_response(request: Dict, env: rl.HazardRoutingEnv, inference: Dict, graph_source: str) -> Dict:
    segments = []
    total_dist = 0.0
    total_time_s = 0.0
    hz_scores = []

    for i, e in enumerate(inference["routeEdges"], start=1):
        u, v = int(e["u"]), int(e["v"])
        coords = [
            node_to_latlng(env.node_pos, u),
            node_to_latlng(env.node_pos, v),
        ]
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

    # Blocked edges for the current rain realization (useful for map overlays).
    blocked_edges = []
    for i, (u, v, data) in enumerate(env.G.edges(data=True), start=1):
        if not bool(data.get("blocked", False)):
            continue
        blocked_edges.append(
            {
                "id": f"blocked-{i}",
                "u": int(u),
                "v": int(v),
                "coordinates": [
                    node_to_latlng(env.node_pos, int(u)),
                    node_to_latlng(env.node_pos, int(v)),
                ],
                "lengthMeters": float(data.get("length", 0.0)),
                "floodScore": float(data.get("flood_score", 0.0)),
                "landslideScore": float(data.get("landslide_score", 0.0)),
                "edgeState": str(data.get("edge_state", "blocked")),
            }
        )

    depot = request["depot"]
    delivery_stops = [{"id": depot.get("id", "depot"), "location": depot["location"], "sequence": 1, "label": depot.get("label", "Depot")}]
    for idx, stop in enumerate(request["deliveryStops"], start=2):
        delivery_stops.append(
            {
                "id": stop.get("id", f"stop-{idx-1}"),
                "location": stop["location"],
                "sequence": idx,
                "label": stop.get("label", f"Stop {idx-1}"),
            }
        )

    return {
        "id": request["id"],
        "type": request["routeType"],  # For now routeType does not switch model/profile.
        "rainIntensity": request["rainIntensity"],
        "graphSource": graph_source,
        "success": inference["success"],
        "terminationReason": inference["terminationReason"],
        "segments": segments,
        "blockedEdges": blocked_edges,
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
            "blockedEdgeCount": len(blocked_edges),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Barebones RL-to-visualization integration demo.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to training config JSON.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument(
        "--request-json",
        type=str,
        default="",
        help="Optional request JSON file. If omitted, auto-generates a mock request from graph nodes.",
    )
    parser.add_argument("--rain-intensity", type=str, default="RI1", help="Used when auto-generating request.")
    parser.add_argument("--route-type", type=str, default="balanced", help="safe|balanced|fast (currently informational).")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Inference epsilon; use 0.0 for deterministic policy.")
    parser.add_argument("--output-json", type=str, default="", help="Optional output JSON file path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config_path)
    checkpoint_path = Path(args.checkpoint_path)

    env, model, graph_source = load_env_and_model(config_path=config_path, checkpoint_path=checkpoint_path)

    if args.request_json:
        request = load_request_from_json(Path(args.request_json))
    else:
        request = build_auto_mock_request(
            env=env,
            rain_intensity=args.rain_intensity,
            route_type=args.route_type,
        )

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


# python mock_visualization_demo.py --config-path mock_visualization/sample_config_200.json --checkpoint-path checkpoints/staged_training/stage_200/best_model.pt --request-json mock_visualization/sample_request.json --output-json mock_visualization/sample_response.json

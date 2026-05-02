from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class LoadedEnv:
    module: Any
    cfg: dict
    base_graph: nx.Graph
    env: Any


@dataclass
class LoadedModelEnv(LoadedEnv):
    model: Any
    checkpoint: dict


def load_trainer_module():
    candidate_modules = [
        "rl_routing_FINAL",
        "rl_routing_wCUDA_wCheckP_latest",
        "rl_routing_wCUDA_wCheckP",
        "rl_routing",
        "rl_routing_wCUDA",
        "rl_routing_curriculum",
        "mock_rl_routing",
    ]
    last_error = None
    for module_name in candidate_modules:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
    raise RuntimeError("Could not import a supported trainer module.") from last_error


def load_env_from_config(
    config_path: str,
    seed: int | None = None,
    num_deliveries_override: int | None = None,
) -> LoadedEnv:
    module = load_trainer_module()
    cfg = module.load_config(config_path)
    resolved_seed = seed if seed is not None else cfg.get("seed")
    if resolved_seed is not None:
        module.set_seed(int(resolved_seed))
    module.apply_runtime_config(cfg)

    graph_cfg = cfg["graph"]
    env_cfg = dict(cfg["environment"])
    reward_cfg = cfg["reward"]
    if num_deliveries_override is not None:
        env_cfg["num_deliveries"] = int(num_deliveries_override)

    base_graph = module.create_base_graph(
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
    env = module.HazardRoutingEnv(
        base_graph,
        num_deliveries=int(env_cfg["num_deliveries"]),
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
    )
    return LoadedEnv(module=module, cfg=cfg, base_graph=base_graph, env=env)


def load_model_env(
    config_path: str,
    checkpoint_path: str,
    seed: int | None = None,
    num_deliveries_override: int | None = None,
) -> LoadedModelEnv:
    loaded = load_env_from_config(
        config_path=config_path,
        seed=seed,
        num_deliveries_override=num_deliveries_override,
    )
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model_cfg = loaded.cfg["model"]
    env = loaded.env

    model = loaded.module.DQN(
        env.state_dim,
        env.action_dim,
        num_nodes=env.num_nodes,
        num_delivery_slots=env.num_deliveries,
        hidden_sizes=tuple(model_cfg["hidden_sizes"]),
        node_embedding_dim=int(model_cfg.get("node_embedding_dim", 16)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedModelEnv(
        module=loaded.module,
        cfg=loaded.cfg,
        base_graph=loaded.base_graph,
        env=loaded.env,
        model=model,
        checkpoint=checkpoint,
    )


def choose_episode_nodes_random(env) -> tuple[int, list[int]]:
    import random

    start_node = random.randint(0, env.num_nodes - 1)
    all_nodes = list(env.G.nodes())
    all_nodes.remove(start_node)
    delivery_nodes = random.sample(all_nodes, env.num_deliveries)
    return int(start_node), [int(node) for node in delivery_nodes]


def choose_episode_nodes_feasible(env) -> tuple[int, list[int]] | None:
    import random

    candidate_components = [
        component for component in env._passable_components() if len(component) >= (env.num_deliveries + 1)
    ]
    if not candidate_components:
        return None

    weights = [len(component) for component in candidate_components]
    chosen_component = random.choices(candidate_components, weights=weights, k=1)[0]
    sampled = random.sample(chosen_component, env.num_deliveries + 1)
    return int(sampled[0]), [int(node) for node in sampled[1:]]


def initialize_env_from_episode(env, module, rain_key: str, start_node: int, delivery_nodes: list[int]):
    env.G = module.activate_hazards(env.base_graph, rain_key)
    rain_idx = module.RAIN_KEYS.index(rain_key)
    env.rain_onehot = module.np.zeros(env.rain_dim, dtype=float)
    env.rain_onehot[rain_idx] = 1.0
    env.current_node = int(start_node)
    env.delivery_nodes = set(int(node) for node in delivery_nodes)
    env.completed = set()
    env.total_time = 0.0
    env.total_hazard = 0.0
    env.steps = 0
    env.prev_node = None
    env.visit_counts = {env.current_node: 1}
    return env._get_state()


def episode_blocked_edges(env) -> list[list[int]]:
    edges = []
    for u, v, data in env.G.edges(data=True):
        if data.get("blocked", False):
            edges.append([int(u), int(v)])
    edges.sort()
    return edges


def rollout_policy_episode(env, module, model, epsilon: float, stop_on_infeasible: bool = True) -> dict:
    start_time = time.perf_counter()
    route_nodes = [int(env.current_node)]
    traversed_edges = []
    hazard_exposure = 0.0
    hazard_score_sum = 0.0
    total_distance = 0.0
    total_reward = 0.0

    if stop_on_infeasible and not env.is_current_episode_feasible():
        elapsed = time.perf_counter() - start_time
        return {
            "success": False,
            "termination_reason": "infeasible_at_reset",
            "timeout": False,
            "steps": 0,
            "total_time": 0.0,
            "total_distance": 0.0,
            "total_reward": float(env.failure_penalty("blockage")),
            "total_hazard_raw": 0.0,
            "hazard_exposure": 0.0,
            "hazard_score_sum": 0.0,
            "route_nodes": route_nodes,
            "traversed_edges": traversed_edges,
            "runtime_sec": elapsed,
        }

    state = env._get_state()
    done = False
    reason = None
    while not done:
        mask = env.get_action_mask()
        action = module.select_action(model, state, mask, epsilon)
        if action is None:
            total_reward += env.failure_penalty("blockage")
            reason = "no_valid_action"
            break

        prev_node = int(env.current_node)
        next_state, reward, done, info = env.step(action)
        curr_node = int(env.current_node)
        edge_data = env.G[prev_node][curr_node]
        flood_score = float(edge_data.get("flood_score", 0.0))
        landslide_score = float(edge_data.get("landslide_score", 0.0))
        edge_length = float(edge_data.get("length", 0.0))
        weighted_hazard = env.w_flood * flood_score + env.w_landslide * landslide_score

        traversed_edges.append([prev_node, curr_node])
        route_nodes.append(curr_node)
        total_distance += edge_length
        hazard_score_sum += weighted_hazard
        hazard_exposure += weighted_hazard * edge_length
        total_reward += float(reward)
        state = next_state
        if done:
            reason = info.get("termination_reason")

    success = len(env.completed) == len(env.delivery_nodes)
    elapsed = time.perf_counter() - start_time
    return {
        "success": bool(success),
        "termination_reason": reason or ("success" if success else "unknown"),
        "timeout": bool(reason in {"timeout", "step_guard_timeout"}),
        "steps": int(env.steps),
        "total_time": float(env.total_time),
        "total_distance": float(total_distance),
        "total_reward": float(total_reward),
        "total_hazard_raw": float(env.total_hazard),
        "hazard_exposure": float(hazard_exposure),
        "hazard_score_sum": float(hazard_score_sum),
        "route_nodes": route_nodes,
        "traversed_edges": traversed_edges,
        "runtime_sec": float(elapsed),
    }


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

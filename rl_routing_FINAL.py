"""
Hazard-aware RL routing prototype on a compact OSM road graph.

This script is config-driven. Edit `configs/experiment_config.json` to run ablations
without touching code.
"""

import json
import math
import random
import argparse
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.graph_utils import get_raw_osm_graph, to_training_graph


# =========================
# Defaults
# =========================
DEFAULT_RAIN_LEVELS = {
    "RI1": {
        "speed_mult": 0.94,
        "flood_block_threshold": 1.0,
        "flood_block_prob": 0.10,
        "landslide_block_threshold": None,
        "landslide_block_prob": 0.00,
    },
    "RI2": {
        "speed_mult": 0.90,
        "flood_block_threshold": 1.0,
        "flood_block_prob": 0.30,
        "landslide_block_threshold": 0.8,
        "landslide_block_prob": 0.05,
    },
    "RI3": {
        "speed_mult": 0.85,
        "flood_block_threshold": 0.6,
        "flood_block_prob": 0.60,
        "landslide_block_threshold": 0.8,
        "landslide_block_prob": 0.15,
    },
    "RI4": {
        "speed_mult": 0.40,
        "flood_block_threshold": 0.6,
        "flood_block_prob": 0.90,
        "landslide_block_threshold": 0.5,
        "landslide_block_prob": 0.30,
    },
    "RI5": {
        "speed_mult": 0.20,
        "flood_block_threshold": 0.2,
        "flood_block_prob": 1.00,
        "landslide_block_threshold": 0.5,
        "landslide_block_prob": 1.00,
    },
}

DEFAULT_CONFIG = {
    "seed": 40,
    "graph": {
        "num_nodes": 15,
        "min_nodes": 12,
        "max_nodes": 20,
        "force_download": False,
        "prebuilt_graphml_path": "",
        "use_existing_hazards": False,
        "flood_attr": "flood_hazard",
        "landslide_attr": "landslide_hazard",
        "travel_time_attr": "travel_time_min",
    },
    "environment": {
        "num_deliveries": 2,
        "min_max_steps": 50,
        "max_steps_multiplier": 2.0,
        "episode_time_scale": 6.0,
    },
    "hazard": {
        "rain_levels": DEFAULT_RAIN_LEVELS,
        "active_rain_keys": ["RI1", "RI2", "RI3"],
        "flood_time_weight": 0.5,
        "landslide_time_weight": 0.5,
        "high_risk_flood_threshold": 0.6,
        "high_risk_landslide_threshold": 0.5,
        "max_neighbor_slots": 4,
        "neighbor_feature_dim": 7,
    },
    "reward": {
        "delivery": 50.0,
        "mission_success": 100.0,
        "k_progress": 0.1,
        "hazard_lambda": 10.0,
        "w_flood": 0.6,
        "w_landslide": 0.4,
        "eta_time": 0.2,
        "step_cost": 0.2,
        "revisit_penalty": 0.0,
        "backtrack_penalty": 0.0,
        "penalty_timeout": -100.0,
        "penalty_blockage": -100.0,
        "penalty_incomplete_per_delivery": -20.0,
    },
    "model": {
        "hidden_sizes": [64, 64],
        "node_embedding_dim": 16,
    },
    "replay": {
        "capacity": 10000,
    },
    "training": {
        "num_episodes": 1500,
        "gamma": 0.99,
        "lr": 3e-4,
        "device": "auto", # "auto", "cpu", "cuda"
        "batch_size": 32,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "epsilon_schedule": "multiplicative",
        "epsilon_exp_decay_rate": 0.005,
        "target_update_every_steps": 500,
        "log_every": 20,
        "eval_every": 20,
        "use_pretrained_model": False,
        "pretrained_model_path": "",
        "resume_optimizer": False,
        "reroll_infeasible_episodes": False,
        "sample_feasible_nodes_directly": False,
        "max_feasibility_resamples": 20,
        "resume_training": False,
        "resume_checkpoint_path": "",
        "save_last_every_episodes": 200,
        "early_stopping": {
            "enabled": False,
            "patience_evals": 20,
            "min_evals": 8,
            "min_episodes": 0,
            "restore_best_weights_for_final_eval": True,
        },
    },
    "evaluation": {
        "episodes": 300,
        "epsilon_greedy": 0.0,
        "epsilon_noisy": 0.05,
        "skip_infeasible_at_reset": False,
    },
    "paths": {
        "checkpoints_dir": "checkpoints",
        "runs_dir": "results/runs/",
        "run_log_file": "last_run.txt",
    },
}

CONFIG_PATH_DEFAULT = "configs/experiment_config.json"


# Runtime globals configured from config file.
SEED = DEFAULT_CONFIG["seed"]
RAIN_LEVELS = deepcopy(DEFAULT_CONFIG["hazard"]["rain_levels"])
RAIN_KEYS = list(RAIN_LEVELS.keys())
ACTIVE_RAIN_KEYS = list(DEFAULT_CONFIG["hazard"]["active_rain_keys"])
FLOOD_TIME_WEIGHT = float(DEFAULT_CONFIG["hazard"]["flood_time_weight"])
LANDSLIDE_TIME_WEIGHT = float(DEFAULT_CONFIG["hazard"]["landslide_time_weight"])
HIGH_RISK_FLOOD_THRESHOLD = float(DEFAULT_CONFIG["hazard"]["high_risk_flood_threshold"])
HIGH_RISK_LANDSLIDE_THRESHOLD = float(DEFAULT_CONFIG["hazard"]["high_risk_landslide_threshold"])
MAX_NEIGHBOR_SLOTS = int(DEFAULT_CONFIG["hazard"]["max_neighbor_slots"])
NEIGHBOR_FEATURE_DIM = int(DEFAULT_CONFIG["hazard"]["neighbor_feature_dim"])


def set_seed(seed):
    global SEED
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def resolve_device(train_cfg):
    requested = str(train_cfg.get("device", "auto")).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        print(f"Warning: requested device '{requested}' but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    raise ValueError(f"Unsupported training.device value: {requested}. Use one of: auto, cpu, cuda, cuda:0, ...")


def deep_update(base, override):
    """Recursively merge `override` into `base`."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path=CONFIG_PATH_DEFAULT):
    """Load config from JSON; create default file if missing."""
    cfg = deepcopy(DEFAULT_CONFIG)
    path = Path(config_path)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"Created default config at: {path}")
        return cfg

    with path.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    return deep_update(cfg, user_cfg)


def apply_runtime_config(cfg):
    """Apply config values that are used via module-level globals."""
    global RAIN_LEVELS
    global RAIN_KEYS
    global ACTIVE_RAIN_KEYS
    global FLOOD_TIME_WEIGHT
    global LANDSLIDE_TIME_WEIGHT
    global HIGH_RISK_FLOOD_THRESHOLD
    global HIGH_RISK_LANDSLIDE_THRESHOLD
    global MAX_NEIGHBOR_SLOTS
    global NEIGHBOR_FEATURE_DIM

    hazard_cfg = cfg["hazard"]
    RAIN_LEVELS = deepcopy(hazard_cfg["rain_levels"])
    RAIN_KEYS = list(RAIN_LEVELS.keys())

    active = [k for k in hazard_cfg.get("active_rain_keys", RAIN_KEYS) if k in RAIN_LEVELS]
    ACTIVE_RAIN_KEYS = active if active else list(RAIN_KEYS)

    FLOOD_TIME_WEIGHT = float(hazard_cfg["flood_time_weight"])
    LANDSLIDE_TIME_WEIGHT = float(hazard_cfg["landslide_time_weight"])
    HIGH_RISK_FLOOD_THRESHOLD = float(hazard_cfg["high_risk_flood_threshold"])
    HIGH_RISK_LANDSLIDE_THRESHOLD = float(hazard_cfg["high_risk_landslide_threshold"])
    MAX_NEIGHBOR_SLOTS = int(hazard_cfg["max_neighbor_slots"])
    NEIGHBOR_FEATURE_DIM = int(hazard_cfg["neighbor_feature_dim"])


# =========================
# Graph Construction
# =========================
def create_base_graph(
    num_nodes=30,
    min_nodes=30,
    max_nodes=40,
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

    return to_training_graph(
        raw_graph,
        num_nodes=num_nodes,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        use_existing_hazards=use_existing_hazards,
        flood_attr=flood_attr,
        landslide_attr=landslide_attr,
        travel_time_attr=travel_time_attr,
    )


# =========================
# Hazard Activation
# =========================
def _haversine_distance_m(pos_a, pos_b):
    lon1, lat1 = float(pos_a[0]), float(pos_a[1])
    lon2, lat2 = float(pos_b[0]), float(pos_b[1])
    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _bearing_radians(pos_a, pos_b):
    lon1, lat1 = math.radians(float(pos_a[0])), math.radians(float(pos_a[1]))
    lon2, lat2 = math.radians(float(pos_b[0])), math.radians(float(pos_b[1]))
    d_lam = lon2 - lon1
    y = math.sin(d_lam) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lam)
    return math.atan2(y, x)


def _sample_block(hazard_value, threshold, prob):
    if threshold is None or hazard_value < threshold:
        return False
    return np.random.rand() < prob


def activate_hazards(G_base, rain_key):
    rain = RAIN_LEVELS[rain_key]
    G = G_base.copy()

    for _, _, data in G.edges(data=True):
        hf = data["flood_score"]
        hl = data["landslide_score"]

        flood_blocked = _sample_block(hf, rain["flood_block_threshold"], rain["flood_block_prob"])
        landslide_blocked = _sample_block(hl, rain["landslide_block_threshold"], rain["landslide_block_prob"])
        blocked = flood_blocked or landslide_blocked

        if blocked:
            data["blocked"] = True
            data["travel_time"] = None
            data["edge_state"] = "blocked"
            data["flood_triggered_block"] = flood_blocked
            data["landslide_triggered_block"] = landslide_blocked
            continue

        speed_mult = max(rain["speed_mult"], 1e-6)
        time_mult = 1.0 / speed_mult
        hazard_factor = 1.0 + FLOOD_TIME_WEIGHT * hf + LANDSLIDE_TIME_WEIGHT * hl
        data["blocked"] = False
        data["travel_time"] = data["base_time"] * time_mult * hazard_factor
        data["edge_state"] = (
            "passable_high_risk"
            if (hf >= HIGH_RISK_FLOOD_THRESHOLD or hl >= HIGH_RISK_LANDSLIDE_THRESHOLD)
            else "passable_with_delay"
        )
        data["flood_triggered_block"] = False
        data["landslide_triggered_block"] = False

    return G


# =========================
# Environment
# =========================
class HazardRoutingEnv:
    def __init__(self, base_graph, num_deliveries=2, env_cfg=None, reward_cfg=None):
        env_cfg = env_cfg or {}
        reward_cfg = reward_cfg or {}

        self.base_graph = base_graph
        self.num_nodes = base_graph.number_of_nodes()
        self.num_deliveries = min(int(num_deliveries), self.num_nodes - 1)

        min_max_steps = int(env_cfg.get("min_max_steps", 50))
        step_multiplier = float(env_cfg.get("max_steps_multiplier", 2.0))
        self.max_steps = max(min_max_steps, int(self.num_nodes * step_multiplier))
        self.episode_time_scale = float(env_cfg.get("episode_time_scale", 6.0))

        self.rain_dim = len(RAIN_KEYS)
        self.max_neighbor_slots = MAX_NEIGHBOR_SLOTS
        # Goal-aware neighbor features require at least 7 slots per neighbor.
        self.neighbor_feature_dim = max(NEIGHBOR_FEATURE_DIM, 7)

        edge_data = list(base_graph.edges(data=True))
        avg_base_time = np.mean([d["base_time"] for _, _, d in edge_data]) if edge_data else 1.0
        avg_hazard = np.mean([d["flood_score"] + d["landslide_score"] for _, _, d in edge_data]) if edge_data else 1.0
        max_edge_length = np.max([d.get("length", 1.0) for _, _, d in edge_data]) if edge_data else 1.0
        max_base_time = np.max([d.get("base_time", 1.0) for _, _, d in edge_data]) if edge_data else 1.0
        self.avg_base_time = max(float(avg_base_time), 1e-6)
        self.avg_edge_hazard = max(float(avg_hazard), 1e-6)
        self.max_edge_length = max(float(max_edge_length), 1e-6)
        self.max_base_time = max(float(max_base_time), 1e-6)

        # Reward coefficients
        self.reward_delivery = float(reward_cfg.get("delivery", 50.0))
        self.reward_mission_success = float(reward_cfg.get("mission_success", 100.0))
        self.k_progress = float(reward_cfg.get("k_progress", 0.1))
        self.hazard_lambda = float(reward_cfg.get("hazard_lambda", 10.0))
        self.w_flood = float(reward_cfg.get("w_flood", 0.6))
        self.w_landslide = float(reward_cfg.get("w_landslide", 0.4))
        self.eta_time = float(reward_cfg.get("eta_time", 0.2))
        self.step_cost = float(reward_cfg.get("step_cost", 0.2))
        self.revisit_penalty = float(reward_cfg.get("revisit_penalty", 0.0))
        self.backtrack_penalty = float(reward_cfg.get("backtrack_penalty", 0.0))
        self.penalty_timeout = float(reward_cfg.get("penalty_timeout", -100.0))
        self.penalty_blockage = float(reward_cfg.get("penalty_blockage", -100.0))
        self.penalty_incomplete_per_delivery = float(reward_cfg.get("penalty_incomplete_per_delivery", -20.0))

        self.max_elapsed_time = max(1e-6, self.max_steps * self.max_base_time * self.episode_time_scale)
        self.max_episode_time = self.max_elapsed_time

        self.node_pos = {n: np.array(base_graph.nodes[n]["pos"], dtype=float) for n in base_graph.nodes()}
        self.shortest_len = dict(nx.all_pairs_dijkstra_path_length(base_graph, weight="length"))
        self.max_shortest_len = max(
            (dist for src_map in self.shortest_len.values() for dist in src_map.values()),
            default=1.0,
        )
        self.max_shortest_len = max(float(self.max_shortest_len), 1e-6)

        self.state_dim = (
            3
            + 4
            + self.max_neighbor_slots * self.neighbor_feature_dim
            + self.rain_dim
        )
        self.action_dim = self.max_neighbor_slots
        self._action_slots = []

    def _nearest_unvisited_shortest(self, node, unvisited_nodes):
        if not unvisited_nodes:
            return 0.0
        dist_map = self.shortest_len.get(node, {})
        return min(dist_map.get(d, self.max_shortest_len) for d in unvisited_nodes)

    def _incomplete_penalty(self):
        remaining = len([d for d in self.delivery_nodes if d not in self.completed])
        return self.penalty_incomplete_per_delivery * remaining

    def failure_penalty(self, reason):
        base = self.penalty_blockage if reason == "blockage" else self.penalty_timeout
        return base + self._incomplete_penalty()

    def is_current_episode_feasible(self):
        """Check whether every delivery is reachable from the current node on passable edges."""
        if not self.delivery_nodes:
            return True
        if self.current_node not in self.G:
            return False

        reachable = set()
        stack = [self.current_node]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            for nbr in self.G.neighbors(node):
                if self.G[node][nbr].get("blocked", False):
                    continue
                if nbr not in reachable:
                    stack.append(nbr)

        return all(node in reachable for node in self.delivery_nodes)

    def _passable_components(self):
        passable = nx.Graph()
        passable.add_nodes_from(self.G.nodes())
        passable.add_edges_from(
            (u, v) for u, v, data in self.G.edges(data=True) if not data.get("blocked", False)
        )
        return [list(component) for component in nx.connected_components(passable)]

    def _assign_random_episode_nodes(self):
        self.current_node = random.randint(0, self.num_nodes - 1)
        all_nodes = list(self.G.nodes())
        all_nodes.remove(self.current_node)
        self.delivery_nodes = set(random.sample(all_nodes, self.num_deliveries))

    def _assign_feasible_episode_nodes(self):
        candidate_components = [
            component for component in self._passable_components() if len(component) >= (self.num_deliveries + 1)
        ]
        if not candidate_components:
            return False

        weights = [len(component) for component in candidate_components]
        chosen_component = random.choices(candidate_components, weights=weights, k=1)[0]
        sampled = random.sample(chosen_component, self.num_deliveries + 1)
        self.current_node = int(sampled[0])
        self.delivery_nodes = set(int(node) for node in sampled[1:])
        return True

    def reset(self, sample_feasible_nodes=False):
        rain_key = random.choice(ACTIVE_RAIN_KEYS)
        self.G = activate_hazards(self.base_graph, rain_key)
        rain_idx = RAIN_KEYS.index(rain_key)
        self.rain_onehot = np.zeros(self.rain_dim, dtype=float)
        self.rain_onehot[rain_idx] = 1.0

        assigned = False
        if sample_feasible_nodes:
            assigned = self._assign_feasible_episode_nodes()
        if not assigned:
            self._assign_random_episode_nodes()
        self.completed = set()

        self.total_time = 0.0
        self.total_hazard = 0.0
        self.steps = 0
        self.prev_node = None
        self.visit_counts = {self.current_node: 1}
        return self._get_state()

    def _build_unvisited_delivery_state(self):
        unvisited = [d for d in self.delivery_nodes if d not in self.completed]
        unvisited_idx = np.zeros(self.num_deliveries, dtype=np.int64)
        unvisited_mask = np.zeros(self.num_deliveries, dtype=np.float32)
        for i, node_id in enumerate(unvisited[: self.num_deliveries]):
            unvisited_idx[i] = int(node_id)
            unvisited_mask[i] = 1.0
        return unvisited, unvisited_idx, unvisited_mask

    def _build_target_features(self, unvisited):
        n_remaining_norm = len(unvisited) / max(self.num_deliveries, 1)
        n_completed_norm = len(self.completed) / max(self.num_deliveries, 1)
        elapsed_norm = min(self.total_time / self.max_episode_time, 1.0)

        cur_pos = self.node_pos[self.current_node]
        nearest_euclid = 0.0
        nearest_shortest = 0.0
        bearing = 0.0
        farthest_dist = 0.0

        if unvisited:
            dist_euclid = {d: _haversine_distance_m(cur_pos, self.node_pos[d]) for d in unvisited}
            nearest_node = min(dist_euclid, key=dist_euclid.get)
            nearest_euclid = dist_euclid[nearest_node]
            farthest_dist = max(dist_euclid.values())
            nearest_shortest = self.shortest_len.get(self.current_node, {}).get(nearest_node, self.max_shortest_len)
            bearing = _bearing_radians(cur_pos, self.node_pos[nearest_node])

        return np.array(
            [
                n_remaining_norm,
                n_completed_norm,
                elapsed_norm,
                min(nearest_euclid / self.max_shortest_len, 1.0),
                min(nearest_shortest / self.max_shortest_len, 1.0),
                bearing / math.pi,
                min(farthest_dist / self.max_shortest_len, 1.0),
            ],
            dtype=float,
        )

    def _get_action_slots(self):
        neighbors = sorted(
            list(self.G.neighbors(self.current_node)),
            key=lambda nbr: self.G[self.current_node][nbr].get("length", 0.0),
        )
        return neighbors[: self.max_neighbor_slots]

    def _build_neighbor_features(self, action_slots, unvisited):
        features = []
        current_nearest = self._nearest_unvisited_shortest(self.current_node, unvisited)

        for nbr in action_slots:
            edge = self.G[self.current_node][nbr]
            flood_score = edge.get("flood_score", 0.0)
            landslide_score = edge.get("landslide_score", 0.0)
            length_norm = min(edge.get("length", 0.0) / self.max_edge_length, 1.0)
            travel_time = edge.get("travel_time", None)
            travel_time_norm = 0.0 if travel_time is None else min(travel_time / self.max_episode_time, 1.0)
            feasible = 0.0 if edge.get("blocked", False) else 1.0

            # Goal-aware neighbor context for action ranking.
            next_nearest = self._nearest_unvisited_shortest(nbr, unvisited)
            next_nearest_norm = min(next_nearest / self.max_shortest_len, 1.0)
            progress_delta_norm = float(
                np.clip((current_nearest - next_nearest) / self.max_shortest_len, -1.0, 1.0)
            )

            features.extend(
                [
                    flood_score,
                    landslide_score,
                    length_norm,
                    travel_time_norm,
                    feasible,
                    next_nearest_norm,
                    progress_delta_norm,
                ]
            )

        expected = self.max_neighbor_slots * self.neighbor_feature_dim
        if len(features) < expected:
            features.extend([0.0] * (expected - len(features)))
        return np.array(features, dtype=float)

    def _get_state(self):
        unvisited, unvisited_idx, unvisited_mask = self._build_unvisited_delivery_state()
        target_feats = self._build_target_features(unvisited)
        action_slots = self._get_action_slots()
        neighbor_feats = self._build_neighbor_features(action_slots, unvisited)
        state_vec = np.concatenate([target_feats, neighbor_feats, self.rain_onehot])
        return {
            "state_vec": torch.tensor(state_vec, dtype=torch.float32),
            "current_idx": torch.tensor(int(self.current_node), dtype=torch.long),
            "unvisited_idx": torch.tensor(unvisited_idx, dtype=torch.long),
            "unvisited_mask": torch.tensor(unvisited_mask, dtype=torch.float32),
        }

    def get_action_mask(self):
        self._action_slots = self._get_action_slots()
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for slot_idx, nbr in enumerate(self._action_slots):
            if not self.G[self.current_node][nbr].get("blocked", False):
                mask[slot_idx] = 1.0
        return torch.tensor(mask, dtype=torch.float32)

    def step(self, action):
        self.steps += 1
        mask = self.get_action_mask()
        if action is None or action < 0 or action >= self.action_dim or mask[action] == 0:
            reward = self.failure_penalty("blockage")
            return self._get_state(), reward, True, {"termination_reason": "invalid_action"}

        unvisited_before = [d for d in self.delivery_nodes if d not in self.completed]
        d_before = self._nearest_unvisited_shortest(self.current_node, unvisited_before)

        prev_node = self.current_node
        next_node = self._action_slots[action]
        edge = self.G[self.current_node][next_node]
        travel_time = edge["travel_time"]
        hf = edge["flood_score"]
        hl = edge["landslide_score"]
        revisiting_node = self.visit_counts.get(next_node, 0) > 0
        is_backtrack = self.prev_node is not None and next_node == self.prev_node

        self.total_time += travel_time
        self.total_hazard += (hf + hl)
        self.current_node = next_node
        self.prev_node = prev_node
        self.visit_counts[next_node] = self.visit_counts.get(next_node, 0) + 1

        delivery_reward = 0.0
        if next_node in self.delivery_nodes and next_node not in self.completed:
            self.completed.add(next_node)
            delivery_reward = self.reward_delivery

        unvisited_after = [d for d in self.delivery_nodes if d not in self.completed]
        d_after = self._nearest_unvisited_shortest(self.current_node, unvisited_after)

        progress_reward = self.k_progress * ((d_before - d_after) / self.max_shortest_len)
        hazard_penalty = -self.hazard_lambda * (self.w_flood * hf + self.w_landslide * hl)
        time_penalty = -self.eta_time * travel_time
        step_penalty = -self.step_cost
        revisit_penalty = -self.revisit_penalty if revisiting_node else 0.0
        backtrack_penalty = -self.backtrack_penalty if is_backtrack else 0.0
        reward = (
            delivery_reward
            + progress_reward
            + hazard_penalty
            + time_penalty
            + step_penalty
            + revisit_penalty
            + backtrack_penalty
        )

        done = False
        reason = None

        if len(self.completed) == len(self.delivery_nodes):
            reward += self.reward_mission_success
            done = True
            reason = "success"

        if (not done) and (self.total_time > self.max_elapsed_time):
            reward += self.failure_penalty("timeout")
            done = True
            reason = "timeout"

        if (not done) and (self.steps >= self.max_steps):
            reward += self.failure_penalty("timeout")
            done = True
            reason = "step_guard_timeout"

        return self._get_state(), reward, done, {"termination_reason": reason}


# =========================
# Model + Replay
# =========================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_nodes, num_delivery_slots, hidden_sizes=(64, 64), node_embedding_dim=16):
        super().__init__()
        self.node_embedding = nn.Embedding(int(num_nodes), int(node_embedding_dim))
        self.num_delivery_slots = int(num_delivery_slots)
        embed_input_dim = int(node_embedding_dim) * 2  # current node + pooled unvisited delivery embedding
        layers = []
        prev = state_dim + embed_input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state_vec, current_idx, unvisited_idx, unvisited_mask):
        cur_emb = self.node_embedding(current_idx)
        unvisited_emb = self.node_embedding(unvisited_idx)
        mask = unvisited_mask.unsqueeze(-1)
        pooled_unvisited = (unvisited_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        model_in = torch.cat([state_vec, cur_emb, pooled_unvisited], dim=1)
        return self.net(model_in)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        (
            state_vecs,
            current_idxs,
            unvisited_idxs,
            unvisited_masks,
            actions,
            rewards,
            next_state_vecs,
            next_current_idxs,
            next_unvisited_idxs,
            next_unvisited_masks,
            dones,
            next_masks,
        ) = zip(*batch)
        return (
            torch.stack(state_vecs),
            torch.tensor(current_idxs, dtype=torch.long),
            torch.stack(unvisited_idxs),
            torch.stack(unvisited_masks),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_state_vecs),
            torch.tensor(next_current_idxs, dtype=torch.long),
            torch.stack(next_unvisited_idxs),
            torch.stack(next_unvisited_masks),
            torch.tensor(dones, dtype=torch.float32),
            torch.stack(next_masks),
        )

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        return {
            "capacity": int(self.capacity),
            "items": list(self.buffer),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError("ReplayBuffer state must be a dict.")
        capacity = int(state.get("capacity", self.capacity))
        items = list(state.get("items", []))
        self.capacity = capacity
        self.buffer = deque(items, maxlen=self.capacity)


def _capture_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_rng_state(state):
    if not isinstance(state, dict):
        return
    py_state = state.get("python_random_state")
    if py_state is not None:
        random.setstate(py_state)
    np_state = state.get("numpy_random_state")
    if np_state is not None:
        np.random.set_state(np_state)
    torch_state = state.get("torch_random_state")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)
    cuda_state_all = state.get("torch_cuda_random_state_all")
    if cuda_state_all is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state_all)
        except Exception:
            pass


def select_action(model, state, mask, epsilon):
    model_device = next(model.parameters()).device
    mask = mask.to(model_device)
    valid_actions = torch.where(mask == 1)[0]
    if valid_actions.numel() == 0:
        return None
    if random.random() < epsilon:
        return int(random.choice(valid_actions.tolist()))
    with torch.no_grad():
        q_values = model(
            state["state_vec"].unsqueeze(0).to(model_device),
            state["current_idx"].unsqueeze(0).to(model_device),
            state["unvisited_idx"].unsqueeze(0).to(model_device),
            state["unvisited_mask"].unsqueeze(0).to(model_device),
        ).squeeze(0).clone()
        q_values[mask == 0] = -1e9
        return int(torch.argmax(q_values).item())


def update_epsilon(
    epsilon,
    epsilon_start,
    epsilon_min,
    epsilon_decay,
    epsilon_schedule="multiplicative",
    epsilon_exp_decay_rate=None,
    step_index=None,
):
    schedule = str(epsilon_schedule).strip().lower()
    eps_min = float(epsilon_min)

    if schedule == "multiplicative":
        return max(eps_min, float(epsilon) * float(epsilon_decay))

    if schedule == "exp":
        if step_index is None:
            raise ValueError("step_index is required when epsilon_schedule='exp'.")
        if epsilon_exp_decay_rate is None:
            decay = min(max(float(epsilon_decay), 1e-12), 0.999999)
            rate = -math.log(decay)
        else:
            rate = float(epsilon_exp_decay_rate)
        step = max(0, int(step_index))
        return eps_min + (float(epsilon_start) - eps_min) * math.exp(-rate * step)

    raise ValueError(f"Unsupported epsilon_schedule: {epsilon_schedule}")


def evaluate_policy(
    model,
    env,
    num_episodes=100,
    epsilon=0.0,
    skip_infeasible_at_reset=False,
    return_reason_counts=False,
    return_metrics=False,
):
    model.eval()
    rewards = []
    successes = 0
    reason_counts = {}
    episode_steps = []
    episode_times = []
    episode_hazards = []
    success_steps = []
    success_times = []
    success_hazards = []

    for _ in range(num_episodes):
        state = env.reset()
        if skip_infeasible_at_reset and not env.is_current_episode_feasible():
            total_reward = env.failure_penalty("blockage")
            rewards.append(total_reward)
            episode_steps.append(float(env.steps))
            episode_times.append(float(env.total_time))
            episode_hazards.append(float(env.total_hazard))
            reason_counts["infeasible_at_reset"] = reason_counts.get("infeasible_at_reset", 0) + 1
            continue

        done = False
        total_reward = 0.0
        reason = None

        while not done:
            mask = env.get_action_mask()
            action = select_action(model, state, mask, epsilon)
            if action is None:
                total_reward += env.failure_penalty("blockage")
                reason = "no_valid_action"
                break

            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                reason = info.get("termination_reason")

        rewards.append(total_reward)
        success = len(env.completed) == len(env.delivery_nodes)
        episode_steps.append(float(env.steps))
        episode_times.append(float(env.total_time))
        episode_hazards.append(float(env.total_hazard))
        if success:
            successes += 1
            success_steps.append(float(env.steps))
            success_times.append(float(env.total_time))
            success_hazards.append(float(env.total_hazard))
        if reason is None:
            reason = "success" if success else "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    model.train()
    mean_reward = float(np.mean(rewards))
    success_rate = successes / max(num_episodes, 1)
    metrics = {
        "mean_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
        "mean_time": float(np.mean(episode_times)) if episode_times else 0.0,
        "mean_hazard": float(np.mean(episode_hazards)) if episode_hazards else 0.0,
        "mean_steps_success": float(np.mean(success_steps)) if success_steps else None,
        "mean_time_success": float(np.mean(success_times)) if success_times else None,
        "mean_hazard_success": float(np.mean(success_hazards)) if success_hazards else None,
    }
    if return_reason_counts and return_metrics:
        return mean_reward, success_rate, reason_counts, metrics
    if return_reason_counts:
        return mean_reward, success_rate, reason_counts
    if return_metrics:
        return mean_reward, success_rate, metrics
    return mean_reward, success_rate


def format_reason_counts(reason_counts, total_episodes):
    if not reason_counts:
        return "none"
    total = max(int(total_episodes), 1)
    ordered = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k}:{v} ({(v / total) * 100:.1f}%)" for k, v in ordered)


def format_eval_metrics(metrics):
    if not metrics:
        return "none"

    def _fmt(value):
        if value is None:
            return "n/a"
        return f"{float(value):.2f}"

    return (
        f"AvgSteps:{_fmt(metrics.get('mean_steps'))}, "
        f"AvgTime:{_fmt(metrics.get('mean_time'))}, "
        f"AvgHazard:{_fmt(metrics.get('mean_hazard'))}, "
        f"SuccSteps:{_fmt(metrics.get('mean_steps_success'))}, "
        f"SuccTime:{_fmt(metrics.get('mean_time_success'))}, "
        f"SuccHazard:{_fmt(metrics.get('mean_hazard_success'))}"
    )


def reset_episode_with_feasibility(
    env,
    reroll_infeasible=False,
    max_resamples=20,
    sample_feasible_nodes_directly=False,
):
    """Reset the environment, optionally sampling directly from feasible nodes or rerolling."""
    attempts = 0
    while True:
        state = env.reset(sample_feasible_nodes=sample_feasible_nodes_directly)
        feasible = env.is_current_episode_feasible()
        if feasible or not reroll_infeasible:
            return state, feasible, attempts
        attempts += 1
        if attempts >= max_resamples:
            return state, feasible, attempts


# =========================
# Training
# =========================
def train(config_path=CONFIG_PATH_DEFAULT, config_overrides=None, force_resume=False):
    cfg = load_config(config_path)
    if config_overrides:
        cfg = deep_update(cfg, config_overrides)
    if force_resume:
        cfg = deep_update(
            cfg,
            {
                "training": {
                    "resume_training": True,
                }
            },
        )

    set_seed(cfg["seed"])
    apply_runtime_config(cfg)

    graph_cfg = cfg["graph"]
    env_cfg = cfg["environment"]
    reward_cfg = cfg["reward"]
    model_cfg = cfg["model"]
    replay_cfg = cfg["replay"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    paths_cfg = cfg["paths"]
    device = resolve_device(train_cfg)

    checkpoints_dir = Path(paths_cfg["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_dir / "best_model.pt"
    last_model_path = checkpoints_dir / "last_model.pt"

    runs_dir = Path(paths_cfg.get("runs_dir", "results"))
    run_log_file = str(paths_cfg.get("run_log_file", "last_run.txt"))
    run_log_path = Path(run_log_file)
    if not run_log_path.is_absolute() and run_log_path.parent == Path("."):
        run_log_path = runs_dir / run_log_path
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    resume_training = bool(train_cfg.get("resume_training", False))
    resume_checkpoint_path = str(train_cfg.get("resume_checkpoint_path", "") or "").strip()
    run_log_mode = "a" if resume_training else "w"
    run_log_fp = run_log_path.open(run_log_mode, encoding="utf-8")

    def log(msg):
        print(msg)
        run_log_fp.write(f"{msg}\n")
        run_log_fp.flush()

    def _wall_clock():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _fmt_minutes(seconds):
        try:
            return f"{float(seconds) / 60.0:.2f}m"
        except (TypeError, ValueError):
            return "n/a"

    train_time_offset_seconds = 0.0
    best_model_elapsed_seconds = 0.0
    session_perf_start = time.perf_counter()

    def current_elapsed_seconds():
        return float(train_time_offset_seconds + (time.perf_counter() - session_perf_start))

    def _load_model_state_with_transfer(model, incoming_state, allow_partial_node_embedding=False):
        """
        Load a checkpoint state dict into `model` while tolerating selected shape changes.

        If `allow_partial_node_embedding=True`, `node_embedding.weight` is warm-started by
        copying overlapping rows when row count changes but embedding dim matches.
        """
        if not isinstance(incoming_state, dict):
            raise TypeError("incoming_state must be a state_dict-like dict.")

        model_state = model.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        partial_transfer = []

        for key, src_val in incoming_state.items():
            if key not in model_state:
                continue
            dst_val = model_state[key]
            if not torch.is_tensor(src_val):
                continue

            if tuple(src_val.shape) == tuple(dst_val.shape):
                filtered_state[key] = src_val
                continue

            if (
                allow_partial_node_embedding
                and key == "node_embedding.weight"
                and src_val.ndim == 2
                and dst_val.ndim == 2
                and int(src_val.shape[1]) == int(dst_val.shape[1])
            ):
                rows_to_copy = min(int(src_val.shape[0]), int(dst_val.shape[0]))
                merged = dst_val.detach().clone()
                merged[:rows_to_copy] = src_val[:rows_to_copy].to(dtype=dst_val.dtype)
                filtered_state[key] = merged
                partial_transfer.append(
                    f"{key}: copied {rows_to_copy}/{dst_val.shape[0]} rows from checkpoint ({src_val.shape[0]} rows)"
                )
                continue

            skipped_mismatch.append(
                f"{key}: checkpoint{tuple(src_val.shape)} != model{tuple(dst_val.shape)}"
            )

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        return missing_keys, unexpected_keys, skipped_mismatch, partial_transfer

    try:
        config_path_resolved = str(Path(config_path).resolve())
        configured_step_cost = float(reward_cfg.get("step_cost", 0.2))
        if configured_step_cost < 0.0:
            raise ValueError(f"reward.step_cost must be >= 0.0, got {configured_step_cost}")
        configured_revisit_penalty = float(reward_cfg.get("revisit_penalty", 0.0))
        configured_backtrack_penalty = float(reward_cfg.get("backtrack_penalty", 0.0))
        if configured_revisit_penalty < 0.0:
            raise ValueError(
                f"reward.revisit_penalty must be >= 0.0, got {configured_revisit_penalty}"
            )
        if configured_backtrack_penalty < 0.0:
            raise ValueError(
                f"reward.backtrack_penalty must be >= 0.0, got {configured_backtrack_penalty}"
            )

        use_pretrained = bool(train_cfg.get("use_pretrained_model", False))
        pretrained_model_path = str(train_cfg.get("pretrained_model_path", "") or "").strip()
        resume_optimizer = bool(train_cfg.get("resume_optimizer", False))
        reroll_infeasible_episodes = bool(train_cfg.get("reroll_infeasible_episodes", False))
        sample_feasible_nodes_directly = bool(train_cfg.get("sample_feasible_nodes_directly", False))
        max_feasibility_resamples = max(0, int(train_cfg.get("max_feasibility_resamples", 20)))
        save_last_every = int(train_cfg.get("save_last_every_episodes", 200))
        if save_last_every < 0:
            raise ValueError("training.save_last_every_episodes must be >= 0")

        skip_infeasible_at_reset_eval = bool(eval_cfg.get("skip_infeasible_at_reset", False))

        if resume_training and use_pretrained:
            log("WARNING: training.resume_training=True overrides use_pretrained_model; resume checkpoint will be used.")

        if run_log_mode == "a":
            log("\n=== Resumed Training Session ===")

        log(f"Session started | {_wall_clock()}")

        log(
            "Run config | "
            f"config_path={config_path_resolved} | "
            f"seed={cfg['seed']} | "
            f"num_deliveries={env_cfg.get('num_deliveries')} | "
            f"step_cost={configured_step_cost} | "
            f"revisit_penalty={configured_revisit_penalty} | "
            f"backtrack_penalty={configured_backtrack_penalty} | "
            f"reroll_infeasible_episodes={reroll_infeasible_episodes} | "
            f"sample_feasible_nodes_directly={sample_feasible_nodes_directly} | "
            f"max_feasibility_resamples={max_feasibility_resamples} | "
            f"skip_infeasible_at_reset_eval={skip_infeasible_at_reset_eval} | "
            f"resume_training={resume_training} | "
            f"resume_checkpoint_path={resume_checkpoint_path if resume_checkpoint_path else str(last_model_path)} | "
            f"use_pretrained_model={use_pretrained} | "
            f"pretrained_model_path={pretrained_model_path if pretrained_model_path else '<none>'} | "
            f"resume_optimizer={resume_optimizer}"
        )

        base_graph = create_base_graph(
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
        env = HazardRoutingEnv(
            base_graph,
            num_deliveries=int(env_cfg["num_deliveries"]),
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
        )
        base_graph_node_link = nx.node_link_data(base_graph, edges="edges")

        online = DQN(
            env.state_dim,
            env.action_dim,
            num_nodes=env.num_nodes,
            num_delivery_slots=env.num_deliveries,
            hidden_sizes=tuple(model_cfg["hidden_sizes"]),
            node_embedding_dim=int(model_cfg.get("node_embedding_dim", 16)),
        )
        target = DQN(
            env.state_dim,
            env.action_dim,
            num_nodes=env.num_nodes,
            num_delivery_slots=env.num_deliveries,
            hidden_sizes=tuple(model_cfg["hidden_sizes"]),
            node_embedding_dim=int(model_cfg.get("node_embedding_dim", 16)),
        )
        target.load_state_dict(online.state_dict())
        online.to(device)
        target.to(device)

        optimizer = optim.Adam(online.parameters(), lr=float(train_cfg["lr"]))
        buffer = ReplayBuffer(capacity=int(replay_cfg["capacity"]))

        num_episodes = int(train_cfg["num_episodes"])
        gamma = float(train_cfg["gamma"])
        epsilon_start = float(train_cfg["epsilon_start"])
        epsilon = epsilon_start
        epsilon_min = float(train_cfg["epsilon_min"])
        epsilon_decay = float(train_cfg["epsilon_decay"])
        epsilon_schedule = str(train_cfg.get("epsilon_schedule", "multiplicative"))
        epsilon_exp_decay_rate = train_cfg.get("epsilon_exp_decay_rate", None)
        if epsilon_exp_decay_rate is not None:
            epsilon_exp_decay_rate = float(epsilon_exp_decay_rate)
        epsilon_step = 0
        batch_size = int(train_cfg["batch_size"])
        target_update_every_steps = int(train_cfg["target_update_every_steps"])
        log_every = int(train_cfg["log_every"])
        eval_every = int(train_cfg["eval_every"])

        eval_episodes = int(eval_cfg["episodes"])
        eval_eps0 = float(eval_cfg["epsilon_greedy"])
        eval_eps_noise = float(eval_cfg["epsilon_noisy"])
        early_stop_cfg = train_cfg.get("early_stopping", {})
        early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
        early_stop_patience_evals = int(early_stop_cfg.get("patience_evals", 20))
        early_stop_min_evals = int(early_stop_cfg.get("min_evals", 8))
        early_stop_min_episodes = int(early_stop_cfg.get("min_episodes", 0))
        early_stop_restore_best = bool(early_stop_cfg.get("restore_best_weights_for_final_eval", True))

        start_episode = 0
        train_steps = 0
        reward_history = []
        success_history = []
        best_eval_success = -1.0
        best_eval_reward = -1e9
        best_episode = 0
        last_completed_episode = 0
        eval_count = 0
        evals_since_improvement = 0
        early_stopped = False

        def _to_float(value, default):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _make_checkpoint_payload(episode_completed):
            return {
                "episode": int(episode_completed),
                "model_state_dict": online.state_dict(),
                "target_state_dict": target.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "replay_buffer_state": buffer.state_dict(),
                "rng_state": _capture_rng_state(),
                "epsilon": float(epsilon),
                "epsilon_step": int(epsilon_step),
                "train_steps": int(train_steps),
                "best_episode": int(best_episode),
                "best_eval_success": float(best_eval_success),
                "best_eval_reward": float(best_eval_reward),
                "eval_success_rate_primary": float(best_eval_success),
                "eval_mean_reward_primary": float(best_eval_reward),
                "cumulative_train_seconds": float(current_elapsed_seconds()),
                "best_model_elapsed_seconds": float(best_model_elapsed_seconds),
                "reward_history_tail": [float(x) for x in reward_history[-max(log_every, 1):]],
                "success_history_tail": [float(x) for x in success_history[-max(log_every, 1):]],
                "graph_num_nodes": env.num_nodes,
                "action_dim": env.action_dim,
                "num_deliveries": env.num_deliveries,
                "seed": SEED,
                "model_variant": "spatial_neighbor_head",
                "config_path": str(config_path),
                "config_path_resolved": config_path_resolved,
                "base_graph_node_link": base_graph_node_link,
                "graph_node_ids": [int(n) for n in base_graph.nodes()],
            }

        if resume_training:
            ckpt_path = Path(resume_checkpoint_path) if resume_checkpoint_path else last_model_path
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Resume requested but checkpoint not found: {ckpt_path}. "
                    "Set training.resume_checkpoint_path or ensure last_model.pt exists."
                )

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
            else:
                model_state = checkpoint

            missing_keys, unexpected_keys, skipped_mismatch, partial_transfer = _load_model_state_with_transfer(
                online, model_state, allow_partial_node_embedding=False
            )
            if skipped_mismatch:
                raise RuntimeError(
                    "Resume checkpoint is incompatible with current model shape. "
                    f"First mismatch: {skipped_mismatch[0]}"
                )
            if partial_transfer:
                log(f"Resume partial transfer: {partial_transfer[0]}")
            target_state = checkpoint.get("target_state_dict") if isinstance(checkpoint, dict) else None
            if isinstance(target_state, dict):
                _, _, target_skipped, _ = _load_model_state_with_transfer(
                    target, target_state, allow_partial_node_embedding=False
                )
                if target_skipped:
                    target.load_state_dict(online.state_dict())
                    log("WARNING: target_state_dict incompatible on resume; cloned online weights into target.")
            else:
                target.load_state_dict(online.state_dict())

            if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            replay_state = checkpoint.get("replay_buffer_state") if isinstance(checkpoint, dict) else None
            if replay_state is not None:
                try:
                    buffer.load_state_dict(replay_state)
                except Exception as exc:
                    log(f"WARNING: failed to restore replay buffer state on resume: {exc}")

            rng_state = checkpoint.get("rng_state") if isinstance(checkpoint, dict) else None
            if rng_state is not None:
                try:
                    _restore_rng_state(rng_state)
                except Exception as exc:
                    log(f"WARNING: failed to restore RNG state on resume: {exc}")

            start_episode = max(0, int(checkpoint.get("episode", 0))) if isinstance(checkpoint, dict) else 0
            start_episode = min(start_episode, num_episodes)
            last_completed_episode = start_episode
            epsilon = _to_float(checkpoint.get("epsilon", epsilon_start), epsilon_start) if isinstance(checkpoint, dict) else epsilon_start
            epsilon_step = int(checkpoint.get("epsilon_step", start_episode)) if isinstance(checkpoint, dict) else start_episode
            train_steps = int(checkpoint.get("train_steps", 0)) if isinstance(checkpoint, dict) else 0
            train_time_offset_seconds = _to_float(
                checkpoint.get("cumulative_train_seconds", 0.0),
                0.0,
            ) if isinstance(checkpoint, dict) else 0.0
            best_model_elapsed_seconds = _to_float(
                checkpoint.get("best_model_elapsed_seconds", 0.0),
                0.0,
            ) if isinstance(checkpoint, dict) else 0.0
            session_perf_start = time.perf_counter()

            best_eval_success = _to_float(
                checkpoint.get("best_eval_success", checkpoint.get("eval_success_rate_primary", best_eval_success)),
                best_eval_success,
            ) if isinstance(checkpoint, dict) else best_eval_success
            best_eval_reward = _to_float(
                checkpoint.get("best_eval_reward", checkpoint.get("eval_mean_reward_primary", best_eval_reward)),
                best_eval_reward,
            ) if isinstance(checkpoint, dict) else best_eval_reward
            best_episode = int(checkpoint.get("best_episode", checkpoint.get("episode", 0))) if isinstance(checkpoint, dict) else 0

            reward_history = list(checkpoint.get("reward_history_tail", [])) if isinstance(checkpoint, dict) else []
            success_history = list(checkpoint.get("success_history_tail", [])) if isinstance(checkpoint, dict) else []

            checkpoint_config_path = str(checkpoint.get("config_path", "") or "").strip() if isinstance(checkpoint, dict) else ""

            log(
                f"Resumed checkpoint: {ckpt_path} | start_episode={start_episode} | "
                f"epsilon={epsilon:.4f} | train_steps={train_steps} | replay_size={len(buffer)} | "
                f"ElapsedSoFar={_fmt_minutes(train_time_offset_seconds)} | "
                f"BestModelTime={_fmt_minutes(best_model_elapsed_seconds)} | "
                f"MissingKeys: {len(missing_keys)}, UnexpectedKeys: {len(unexpected_keys)}"
            )
            if checkpoint_config_path:
                log(f"Resume checkpoint config_path: {checkpoint_config_path}")
                checkpoint_config_path_resolved = str(Path(checkpoint_config_path).resolve())
                if checkpoint_config_path_resolved != config_path_resolved:
                    log(
                        "WARNING: current run config_path differs from resume checkpoint config_path. "
                        "This is valid for transfer/fine-tuning, but verify this is intentional."
                    )

        elif use_pretrained:
            if not pretrained_model_path:
                raise ValueError("training.use_pretrained_model=True but training.pretrained_model_path is empty.")

            ckpt_path = Path(pretrained_model_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
            else:
                model_state = checkpoint
            checkpoint_config_path = ""
            if isinstance(checkpoint, dict):
                checkpoint_config_path = str(checkpoint.get("config_path", "") or "").strip()

            (
                missing_keys,
                unexpected_keys,
                skipped_mismatch,
                partial_transfer,
            ) = _load_model_state_with_transfer(
                online, model_state, allow_partial_node_embedding=True
            )
            target.load_state_dict(online.state_dict())
            log(
                f"Loaded pretrained model: {ckpt_path} | "
                f"MissingKeys: {len(missing_keys)}, UnexpectedKeys: {len(unexpected_keys)}"
            )
            if partial_transfer:
                for msg in partial_transfer:
                    log(f"Pretrained partial transfer: {msg}")
            if skipped_mismatch:
                log(
                    "WARNING: skipped incompatible pretrained tensors "
                    f"(count={len(skipped_mismatch)}). First: {skipped_mismatch[0]}"
                )
            if checkpoint_config_path:
                log(f"Pretrained checkpoint config_path: {checkpoint_config_path}")
                checkpoint_config_path_resolved = str(Path(checkpoint_config_path).resolve())
                if checkpoint_config_path_resolved != config_path_resolved:
                    log(
                        "WARNING: current run config_path differs from checkpoint config_path. "
                        "This is valid for transfer/fine-tuning, but verify this is intentional."
                    )

            ckpt_graph_nodes = int(checkpoint.get("graph_num_nodes", -1)) if isinstance(checkpoint, dict) else -1
            optimizer_compatible = (ckpt_graph_nodes == env.num_nodes) and (len(skipped_mismatch) == 0)
            if resume_optimizer and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
                if optimizer_compatible:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    log("Loaded optimizer state from checkpoint (resume_optimizer=True).")
                else:
                    log(
                        "Skipped optimizer state load: checkpoint/model are not shape-compatible "
                        f"(checkpoint graph nodes={ckpt_graph_nodes}, current graph nodes={env.num_nodes})."
                    )

        log(
            f"Graph stats | Nodes: {env.num_nodes}, Edges: {base_graph.number_of_edges()}, "
            f"AvgBaseTime: {env.avg_base_time:.4f}, AvgHazard: {env.avg_edge_hazard:.4f}, "
            f"StateDim: {env.state_dim}, ActionDim: {env.action_dim}, Tmax(min): {env.max_elapsed_time:.2f}, "
            f"Deliveries: {env.num_deliveries}, Device: {device}"
        )
        interrupted = False
        infeasible_rerolls_total = 0
        try:
            for episode in range(start_episode, num_episodes):
                state, _, infeasible_resamples = reset_episode_with_feasibility(
                    env,
                    reroll_infeasible=reroll_infeasible_episodes,
                    max_resamples=max_feasibility_resamples,
                    sample_feasible_nodes_directly=sample_feasible_nodes_directly,
                )
                infeasible_rerolls_total += infeasible_resamples
                done = False
                total_reward = 0.0

                while not done:
                    mask = env.get_action_mask()
                    action = select_action(online, state, mask, epsilon)
                    if action is None:
                        total_reward += env.failure_penalty("blockage")
                        break

                    next_state, reward, done, _ = env.step(action)
                    next_mask = env.get_action_mask() if not done else torch.zeros(env.action_dim, dtype=torch.float32)

                    buffer.store(
                        (
                            state["state_vec"],
                            state["current_idx"],
                            state["unvisited_idx"],
                            state["unvisited_mask"],
                            action,
                            reward,
                            next_state["state_vec"],
                            next_state["current_idx"],
                            next_state["unvisited_idx"],
                            next_state["unvisited_mask"],
                            done,
                            next_mask,
                        )
                    )
                    state = next_state
                    total_reward += reward

                    if len(buffer) >= batch_size:
                        (
                            state_vecs,
                            current_idxs,
                            unvisited_idxs,
                            unvisited_masks,
                            actions,
                            rewards,
                            next_state_vecs,
                            next_current_idxs,
                            next_unvisited_idxs,
                            next_unvisited_masks,
                            dones,
                            next_masks,
                        ) = buffer.sample(batch_size)
                        state_vecs = state_vecs.to(device)
                        current_idxs = current_idxs.to(device)
                        unvisited_idxs = unvisited_idxs.to(device)
                        unvisited_masks = unvisited_masks.to(device)
                        actions = actions.to(device)
                        rewards = rewards.to(device)
                        next_state_vecs = next_state_vecs.to(device)
                        next_current_idxs = next_current_idxs.to(device)
                        next_unvisited_idxs = next_unvisited_idxs.to(device)
                        next_unvisited_masks = next_unvisited_masks.to(device)
                        dones = dones.to(device)
                        next_masks = next_masks.to(device)

                        q_values = online(state_vecs, current_idxs, unvisited_idxs, unvisited_masks)
                        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                        with torch.no_grad():
                            next_online_q = online(
                                next_state_vecs,
                                next_current_idxs,
                                next_unvisited_idxs,
                                next_unvisited_masks,
                            )
                            next_online_q[next_masks == 0] = -1e9
                            next_actions = next_online_q.argmax(dim=1)
                            next_q = target(
                                next_state_vecs,
                                next_current_idxs,
                                next_unvisited_idxs,
                                next_unvisited_masks,
                            ).gather(1, next_actions.unsqueeze(1)).squeeze()
                            has_valid_next = (next_masks.sum(dim=1) > 0).float()
                            target_q = rewards + gamma * next_q * (1 - dones) * has_valid_next

                        loss = nn.MSELoss()(q_selected, target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_steps += 1
                        if train_steps % target_update_every_steps == 0:
                            target.load_state_dict(online.state_dict())

                reward_history.append(total_reward)
                success_history.append(1 if len(env.completed) == len(env.delivery_nodes) else 0)
                epsilon_step += 1
                epsilon = update_epsilon(
                    epsilon,
                    epsilon_start=epsilon_start,
                    epsilon_min=epsilon_min,
                    epsilon_decay=epsilon_decay,
                    epsilon_schedule=epsilon_schedule,
                    epsilon_exp_decay_rate=epsilon_exp_decay_rate,
                    step_index=epsilon_step,
                )
                last_completed_episode = episode + 1

                if (episode + 1) % log_every == 0:
                    avg_reward = float(np.mean(reward_history[-log_every:]))
                    success_rate = float(np.mean(success_history[-log_every:]))
                    log(
                        f"Episode {episode + 1} | "
                        f"Elapsed: {_fmt_minutes(current_elapsed_seconds())} | "
                        f"LastReward: {total_reward:.2f}, "
                        f"AvgReward({log_every}): {avg_reward:.2f}, "
                        f"SuccessRate({log_every}): {success_rate:.2%}, "
                        f"Epsilon: {epsilon:.3f}, "
                        f"InfeasibleRerolls(total): {infeasible_rerolls_total}"
                    )

                if (episode + 1) % eval_every == 0:
                    eval_count += 1
                    eval_clock = _wall_clock()
                    elapsed_now = current_elapsed_seconds()
                    eval_reward_eps0, eval_success_eps0, eval_reasons_eps0, eval_metrics_eps0 = evaluate_policy(
                        online,
                        env,
                        num_episodes=eval_episodes,
                        epsilon=eval_eps0,
                        skip_infeasible_at_reset=skip_infeasible_at_reset_eval,
                        return_reason_counts=True,
                        return_metrics=True,
                    )
                    eval_reward_epsn, eval_success_epsn, eval_reasons_epsn, eval_metrics_epsn = evaluate_policy(
                        online,
                        env,
                        num_episodes=eval_episodes,
                        epsilon=eval_eps_noise,
                        skip_infeasible_at_reset=skip_infeasible_at_reset_eval,
                        return_reason_counts=True,
                        return_metrics=True,
                    )
                    log(
                        f"[Eval @ Episode {episode + 1} | {eval_clock} | Elapsed: {_fmt_minutes(elapsed_now)}] "
                        f"eps={eval_eps0:.2f} -> MeanReward: {eval_reward_eps0:.2f}, SuccessRate: {eval_success_eps0:.2%} | "
                        f"eps={eval_eps_noise:.2f} -> MeanReward: {eval_reward_epsn:.2f}, SuccessRate: {eval_success_epsn:.2%}"
                    )
                    log(
                        f"[EvalReasons @ Episode {episode + 1} | {eval_clock} | Elapsed: {_fmt_minutes(elapsed_now)}] "
                        f"eps={eval_eps0:.2f} -> {format_reason_counts(eval_reasons_eps0, eval_episodes)} | "
                        f"eps={eval_eps_noise:.2f} -> {format_reason_counts(eval_reasons_epsn, eval_episodes)}"
                    )
                    log(
                        f"[EvalStats @ Episode {episode + 1} | {eval_clock} | Elapsed: {_fmt_minutes(elapsed_now)}] "
                        f"eps={eval_eps0:.2f} -> {format_eval_metrics(eval_metrics_eps0)} | "
                        f"eps={eval_eps_noise:.2f} -> {format_eval_metrics(eval_metrics_epsn)}"
                    )

                    improved = (eval_success_eps0 > best_eval_success) or (
                        eval_success_eps0 == best_eval_success and eval_reward_eps0 > best_eval_reward
                    )
                    if improved:
                        best_eval_success = eval_success_eps0
                        best_eval_reward = eval_reward_eps0
                        best_episode = episode + 1
                        best_model_elapsed_seconds = elapsed_now
                        evals_since_improvement = 0
                        torch.save(_make_checkpoint_payload(best_episode), best_model_path)
                        log(
                            f"Saved best checkpoint: {best_model_path} "
                            f"(episode {best_episode}, success={best_eval_success:.2%}, reward={best_eval_reward:.2f}, "
                            f"elapsed={_fmt_minutes(best_model_elapsed_seconds)})"
                        )
                    else:
                        evals_since_improvement += 1

                    if (
                        early_stop_enabled
                        and eval_count >= early_stop_min_evals
                        and (episode + 1) >= early_stop_min_episodes
                        and evals_since_improvement >= early_stop_patience_evals
                    ):
                        early_stopped = True
                        log(
                            "Early stopping triggered: "
                            f"no primary-eval improvement for {evals_since_improvement} evals "
                            f"after episode {episode + 1}. "
                            f"Best episode={best_episode}, best success={best_eval_success:.2%}, "
                            f"best reward={best_eval_reward:.2f}."
                        )
                        break

                if save_last_every > 0 and ((episode + 1) % save_last_every == 0):
                    torch.save(_make_checkpoint_payload(episode + 1), last_model_path)
                    log(f"Saved resumable checkpoint: {last_model_path} (episode {episode + 1})")

        except KeyboardInterrupt:
            interrupted = True
            log("Training interrupted by user. Saving resumable checkpoint...")

        if interrupted:
            torch.save(_make_checkpoint_payload(last_completed_episode), last_model_path)
            log(
                f"Saved resumable checkpoint: {last_model_path} (episode {last_completed_episode}, "
                f"epsilon={epsilon:.3f}, elapsed={_fmt_minutes(current_elapsed_seconds())})."
            )
            log("Resume by setting training.resume_training=true.")
            return

        if early_stopped and early_stop_restore_best and best_episode > 0 and Path(best_model_path).exists():
            best_payload = torch.load(best_model_path, map_location=device, weights_only=False)
            online.load_state_dict(best_payload["model_state_dict"])
            target.load_state_dict(online.state_dict())
            log(
                "Restored best checkpoint weights before final save/eval "
                f"from episode {best_episode} due to early stopping."
            )

        final_completed_episode = last_completed_episode if early_stopped else num_episodes
        torch.save(_make_checkpoint_payload(final_completed_episode), last_model_path)
        log(f"Saved last checkpoint: {last_model_path} | Elapsed: {_fmt_minutes(current_elapsed_seconds())}")

        final_reward_eps0, final_success_eps0, final_reasons_eps0, final_metrics_eps0 = evaluate_policy(
            online,
            env,
            num_episodes=eval_episodes,
            epsilon=eval_eps0,
            skip_infeasible_at_reset=skip_infeasible_at_reset_eval,
            return_reason_counts=True,
            return_metrics=True,
        )
        final_reward_epsn, final_success_epsn, final_reasons_epsn, final_metrics_epsn = evaluate_policy(
            online,
            env,
            num_episodes=eval_episodes,
            epsilon=eval_eps_noise,
            skip_infeasible_at_reset=skip_infeasible_at_reset_eval,
            return_reason_counts=True,
            return_metrics=True,
        )
        total_runtime_seconds = current_elapsed_seconds()
        final_eval_clock = _wall_clock()
        log(
            f"Evaluation over {eval_episodes} episodes | {final_eval_clock} | Elapsed: {_fmt_minutes(total_runtime_seconds)} | "
            f"eps={eval_eps0:.2f} MeanReward: {final_reward_eps0:.2f}, SuccessRate: {final_success_eps0:.2%} | "
            f"eps={eval_eps_noise:.2f} MeanReward: {final_reward_epsn:.2f}, SuccessRate: {final_success_epsn:.2%}"
        )
        log(
            f"Evaluation termination reasons | "
            f"eps={eval_eps0:.2f}: {format_reason_counts(final_reasons_eps0, eval_episodes)} | "
            f"eps={eval_eps_noise:.2f}: {format_reason_counts(final_reasons_epsn, eval_episodes)}"
        )
        log(
            f"Evaluation route stats | "
            f"eps={eval_eps0:.2f}: {format_eval_metrics(final_metrics_eps0)} | "
            f"eps={eval_eps_noise:.2f}: {format_eval_metrics(final_metrics_epsn)}"
        )
        log(
            f"Best checkpoint summary | Episode: {best_episode}, "
            f"PrimaryEval MeanReward: {best_eval_reward:.2f}, SuccessRate: {best_eval_success:.2%}"
        )
        log(
            f"Time summary | TotalRuntime: {_fmt_minutes(total_runtime_seconds)}, "
            f"BestModelTime: {_fmt_minutes(best_model_elapsed_seconds)}, "
            f"FinalEpisode: {final_completed_episode}, EvalCount: {eval_count}, EarlyStopped: {early_stopped}"
        )
        log(f"Run log saved to: {run_log_path}")
    finally:
        run_log_fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hazard-aware RL routing model.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="train",
        choices=["train", "resume"],
        help="Use 'resume' to continue from the run's resumable checkpoint without editing the config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH_DEFAULT,
        help="Path to config JSON (default: configs/experiment_config.json).",
    )
    args = parser.parse_args()
    train(config_path=args.config, force_resume=(args.mode == "resume"))

# python rl_routing_wCUDA_wCheckP.py --config configs/no_hazard_training/no_hazard_config.json
# python rl_routing_wCUDA_wCheckP.py --config configs/no_hazard_training/no_hazard_config_control.json

""" 
1. First run
"training": {
  "resume_training": false,
  "resume_checkpoint_path": "",
  "save_last_every_episodes": 200
}

2. Resume later
"training": {
  "resume_training": true,
  "resume_checkpoint_path": ""
}

3. Resume from specific checkpoint:
"training": {
  "resume_training": true,
  "resume_checkpoint_path": "checkpoints/staged_training/stage_100/last_model.pt"
}
"""

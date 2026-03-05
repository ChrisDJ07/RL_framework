"""
Hazard-aware RL routing prototype on a compact OSM road graph.

This script is config-driven. Edit `configs/experiment_config.json` to run ablations
without touching code.
"""

import json
import math
import random
import argparse
from collections import deque
from copy import deepcopy
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
    },
    "evaluation": {
        "episodes": 300,
        "epsilon_greedy": 0.0,
        "epsilon_noisy": 0.05,
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

    def reset(self):
        rain_key = random.choice(ACTIVE_RAIN_KEYS)
        self.G = activate_hazards(self.base_graph, rain_key)
        rain_idx = RAIN_KEYS.index(rain_key)
        self.rain_onehot = np.zeros(self.rain_dim, dtype=float)
        self.rain_onehot[rain_idx] = 1.0

        self.current_node = random.randint(0, self.num_nodes - 1)
        all_nodes = list(self.G.nodes())
        all_nodes.remove(self.current_node)
        self.delivery_nodes = set(random.sample(all_nodes, self.num_deliveries))
        self.completed = set()

        self.total_time = 0.0
        self.total_hazard = 0.0
        self.steps = 0
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

        next_node = self._action_slots[action]
        edge = self.G[self.current_node][next_node]
        travel_time = edge["travel_time"]
        hf = edge["flood_score"]
        hl = edge["landslide_score"]

        self.total_time += travel_time
        self.total_hazard += (hf + hl)
        self.current_node = next_node

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
        reward = delivery_reward + progress_reward + hazard_penalty + time_penalty + step_penalty

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
        self.buffer = deque(maxlen=int(capacity))

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


def select_action(model, state, mask, epsilon):
    valid_actions = torch.where(mask == 1)[0]
    if valid_actions.numel() == 0:
        return None
    if random.random() < epsilon:
        return random.choice(valid_actions).item()
    with torch.no_grad():
        q_values = model(
            state["state_vec"].unsqueeze(0),
            state["current_idx"].unsqueeze(0),
            state["unvisited_idx"].unsqueeze(0),
            state["unvisited_mask"].unsqueeze(0),
        ).squeeze(0).clone()
        q_values[mask == 0] = -1e9
        return torch.argmax(q_values).item()


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


def evaluate_policy(model, env, num_episodes=100, epsilon=0.0):
    model.eval()
    rewards = []
    successes = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            mask = env.get_action_mask()
            action = select_action(model, state, mask, epsilon)
            if action is None:
                total_reward += env.failure_penalty("blockage")
                break

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if len(env.completed) == len(env.delivery_nodes):
            successes += 1

    model.train()
    return float(np.mean(rewards)), successes / max(num_episodes, 1)


# =========================
# Training
# =========================
def train(config_path=CONFIG_PATH_DEFAULT, config_overrides=None):
    cfg = load_config(config_path)
    if config_overrides:
        cfg = deep_update(cfg, config_overrides)

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

    runs_dir = Path(paths_cfg.get("runs_dir", "results"))
    run_log_file = str(paths_cfg.get("run_log_file", "last_run.txt"))
    run_log_path = Path(run_log_file)
    if not run_log_path.is_absolute() and run_log_path.parent == Path("."):
        run_log_path = runs_dir / run_log_path
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    run_log_fp = run_log_path.open("w", encoding="utf-8")

    def log(msg):
        print(msg)
        run_log_fp.write(f"{msg}\n")
        run_log_fp.flush()

    try:
        config_path_resolved = str(Path(config_path).resolve())
        configured_step_cost = float(reward_cfg.get("step_cost", 0.2))
        if configured_step_cost < 0.0:
            raise ValueError(f"reward.step_cost must be >= 0.0, got {configured_step_cost}")

        use_pretrained = bool(train_cfg.get("use_pretrained_model", False))
        pretrained_model_path = str(train_cfg.get("pretrained_model_path", "") or "").strip()
        resume_optimizer = bool(train_cfg.get("resume_optimizer", False))
        log(
            "Run config | "
            f"config_path={config_path_resolved} | "
            f"seed={cfg['seed']} | "
            f"num_deliveries={env_cfg.get('num_deliveries')} | "
            f"step_cost={configured_step_cost} | "
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

        optimizer = optim.Adam(online.parameters(), lr=float(train_cfg["lr"]))
        buffer = ReplayBuffer(capacity=int(replay_cfg["capacity"]))

        if use_pretrained:
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

            missing_keys, unexpected_keys = online.load_state_dict(model_state, strict=False)
            target.load_state_dict(online.state_dict())
            log(
                f"Loaded pretrained model: {ckpt_path} | "
                f"MissingKeys: {len(missing_keys)}, UnexpectedKeys: {len(unexpected_keys)}"
            )
            if checkpoint_config_path:
                log(f"Pretrained checkpoint config_path: {checkpoint_config_path}")
                checkpoint_config_path_resolved = str(Path(checkpoint_config_path).resolve())
                if checkpoint_config_path_resolved != config_path_resolved:
                    log(
                        "WARNING: current run config_path differs from checkpoint config_path. "
                        "This is valid for transfer/fine-tuning, but verify this is intentional."
                    )

            if resume_optimizer and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                log("Loaded optimizer state from checkpoint (resume_optimizer=True).")

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

        train_steps = 0
        reward_history = []
        success_history = []
        best_eval_success = -1.0
        best_eval_reward = -1e9
        best_episode = 0

        checkpoints_dir = Path(paths_cfg["checkpoints_dir"])
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoints_dir / "best_model.pt"
        last_model_path = checkpoints_dir / "last_model.pt"

        log(
            f"Graph stats | Nodes: {env.num_nodes}, Edges: {base_graph.number_of_edges()}, "
            f"AvgBaseTime: {env.avg_base_time:.4f}, AvgHazard: {env.avg_edge_hazard:.4f}, "
            f"StateDim: {env.state_dim}, ActionDim: {env.action_dim}, Tmax(min): {env.max_elapsed_time:.2f}, "
            f"Deliveries: {env.num_deliveries}"
        )

        for episode in range(num_episodes):
            state = env.reset()
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

            if (episode + 1) % log_every == 0:
                avg_reward = float(np.mean(reward_history[-log_every:]))
                success_rate = float(np.mean(success_history[-log_every:]))
                log(
                    f"Episode {episode + 1}, "
                    f"LastReward: {total_reward:.2f}, "
                    f"AvgReward({log_every}): {avg_reward:.2f}, "
                    f"SuccessRate({log_every}): {success_rate:.2%}, "
                    f"Epsilon: {epsilon:.3f}"
                )

            if (episode + 1) % eval_every == 0:
                eval_reward_eps0, eval_success_eps0 = evaluate_policy(
                    online, env, num_episodes=eval_episodes, epsilon=eval_eps0
                )
                eval_reward_epsn, eval_success_epsn = evaluate_policy(
                    online, env, num_episodes=eval_episodes, epsilon=eval_eps_noise
                )
                log(
                    f"[Eval @ Episode {episode + 1}] "
                    f"eps={eval_eps0:.2f} -> MeanReward: {eval_reward_eps0:.2f}, SuccessRate: {eval_success_eps0:.2%} | "
                    f"eps={eval_eps_noise:.2f} -> MeanReward: {eval_reward_epsn:.2f}, SuccessRate: {eval_success_epsn:.2%}"
                )

                if (eval_success_eps0 > best_eval_success) or (
                    eval_success_eps0 == best_eval_success and eval_reward_eps0 > best_eval_reward
                ):
                    best_eval_success = eval_success_eps0
                    best_eval_reward = eval_reward_eps0
                    best_episode = episode + 1
                    torch.save(
                        {
                            "episode": best_episode,
                            "model_state_dict": online.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "eval_success_rate_primary": best_eval_success,
                            "eval_mean_reward_primary": best_eval_reward,
                            "graph_num_nodes": env.num_nodes,
                            "action_dim": env.action_dim,
                            "num_deliveries": env.num_deliveries,
                            "seed": SEED,
                            "model_variant": "spatial_neighbor_head",
                            "config_path": str(config_path),
                            "config_path_resolved": config_path_resolved,
                            "base_graph_node_link": base_graph_node_link,
                        },
                        best_model_path,
                    )
                    log(
                        f"Saved best checkpoint: {best_model_path} "
                        f"(episode {best_episode}, success={best_eval_success:.2%}, reward={best_eval_reward:.2f})"
                    )

        torch.save(
            {
                "episode": num_episodes,
                "model_state_dict": online.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "graph_num_nodes": env.num_nodes,
                "action_dim": env.action_dim,
                "num_deliveries": env.num_deliveries,
                "seed": SEED,
                "model_variant": "spatial_neighbor_head",
                "config_path": str(config_path),
                "config_path_resolved": config_path_resolved,
                "base_graph_node_link": base_graph_node_link,
            },
            last_model_path,
        )
        log(f"Saved last checkpoint: {last_model_path}")

        final_reward_eps0, final_success_eps0 = evaluate_policy(
            online, env, num_episodes=eval_episodes, epsilon=eval_eps0
        )
        final_reward_epsn, final_success_epsn = evaluate_policy(
            online, env, num_episodes=eval_episodes, epsilon=eval_eps_noise
        )
        log(
            f"Evaluation over {eval_episodes} episodes | "
            f"eps={eval_eps0:.2f} MeanReward: {final_reward_eps0:.2f}, SuccessRate: {final_success_eps0:.2%} | "
            f"eps={eval_eps_noise:.2f} MeanReward: {final_reward_epsn:.2f}, SuccessRate: {final_success_epsn:.2%}"
        )
        log(
            f"Best checkpoint summary | Episode: {best_episode}, "
            f"PrimaryEval MeanReward: {best_eval_reward:.2f}, SuccessRate: {best_eval_success:.2%}"
        )
        log(f"Run log saved to: {run_log_path}")
    finally:
        run_log_fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hazard-aware RL routing model.")
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH_DEFAULT,
        help="Path to config JSON (default: configs/experiment_config.json).",
    )
    args = parser.parse_args()
    train(config_path=args.config)

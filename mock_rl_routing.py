"""
Hazard-aware RL routing prototype on a compact OSM road graph.

Main components:
1) Base graph construction from cached OSM data.
2) Episode hazard activation (rainfall RI1..RI5, currently sampling RI1..RI3).
3) Routing environment with masked actions and shaped reward.
4) DQN training/evaluation with replay buffer and target network.
"""

import math
import random
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from graph_utils import get_raw_osm_graph, to_training_graph

# =========================
# Reproducibility
# =========================
SEED = 40
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# Graph Construction
# =========================
def create_base_graph(num_nodes=30, min_nodes=30, max_nodes=40, force_download=False):
    """Create the static graph used for all episode hazard realizations."""
    raw_graph = get_raw_osm_graph(min_nodes=min_nodes, force_download=force_download)
    return to_training_graph(
        raw_graph,
        num_nodes=num_nodes,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
    )


# =========================
# Hazard Activation
# =========================
RAIN_LEVELS = {
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
RAIN_KEYS = list(RAIN_LEVELS.keys())
# Feasibility mode: only sample lighter scenarios for now.
ACTIVE_RAIN_KEYS = ["RI1", "RI2", "RI3"]

# travel_time = base_time * (1/speed_mult) * (1 + f*Hf + l*Hl)
FLOOD_TIME_WEIGHT = 0.5
LANDSLIDE_TIME_WEIGHT = 0.5

HIGH_RISK_FLOOD_THRESHOLD = 0.6
HIGH_RISK_LANDSLIDE_THRESHOLD = 0.5

MAX_NEIGHBOR_SLOTS = 4
NEIGHBOR_FEATURE_DIM = 5


def _haversine_distance_m(pos_a, pos_b):
    """Great-circle distance in meters between [lon, lat] points."""
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
    """Initial bearing from pos_a to pos_b in radians in [-pi, pi]."""
    lon1, lat1 = math.radians(float(pos_a[0])), math.radians(float(pos_a[1]))
    lon2, lat2 = math.radians(float(pos_b[0])), math.radians(float(pos_b[1]))
    d_lam = lon2 - lon1
    y = math.sin(d_lam) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lam)
    return math.atan2(y, x)


def _sample_block(hazard_value, threshold, prob):
    """Sample whether one mechanism blocks an edge."""
    if threshold is None or hazard_value < threshold:
        return False
    return np.random.rand() < prob


def activate_hazards(G_base, rain_key):
    """Activate edge states for a single episode."""
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
    """Routing environment with fixed-per-episode hazard realization."""

    def __init__(self, base_graph, num_deliveries=2):
        self.base_graph = base_graph
        self.num_nodes = base_graph.number_of_nodes()
        self.num_deliveries = min(num_deliveries, self.num_nodes - 1)
        self.max_steps = max(50, self.num_nodes * 2)
        self.rain_dim = len(RAIN_KEYS)
        self.max_neighbor_slots = MAX_NEIGHBOR_SLOTS
        self.neighbor_feature_dim = NEIGHBOR_FEATURE_DIM

        edge_data = list(base_graph.edges(data=True))
        avg_base_time = np.mean([d["base_time"] for _, _, d in edge_data]) if edge_data else 1.0
        avg_hazard = np.mean([d["flood_score"] + d["landslide_score"] for _, _, d in edge_data]) if edge_data else 1.0
        max_edge_length = np.max([d.get("length", 1.0) for _, _, d in edge_data]) if edge_data else 1.0
        max_base_time = np.max([d.get("base_time", 1.0) for _, _, d in edge_data]) if edge_data else 1.0
        self.avg_base_time = max(float(avg_base_time), 1e-6)
        self.avg_edge_hazard = max(float(avg_hazard), 1e-6)
        self.max_edge_length = max(float(max_edge_length), 1e-6)
        self.max_base_time = max(float(max_base_time), 1e-6)

        # Paper-aligned reward coefficients.
        self.reward_delivery = 50.0
        self.reward_mission_success = 100.0
        self.k_progress = 0.1
        self.hazard_lambda = 10.0
        self.w_flood = 0.6
        self.w_landslide = 0.4
        self.eta_time = 0.2
        self.penalty_timeout = -100.0
        self.penalty_blockage = -100.0
        self.penalty_incomplete_per_delivery = -20.0

        self.max_elapsed_time = max(1e-6, self.max_steps * self.max_base_time * 6.0)
        self.max_episode_time = self.max_elapsed_time

        # Precomputed geometric context.
        self.node_pos = {n: np.array(base_graph.nodes[n]["pos"], dtype=float) for n in base_graph.nodes()}
        self.shortest_len = dict(nx.all_pairs_dijkstra_path_length(base_graph, weight="length"))
        self.max_shortest_len = max(
            (dist for src_map in self.shortest_len.values() for dist in src_map.values()),
            default=1.0,
        )
        self.max_shortest_len = max(float(self.max_shortest_len), 1e-6)

        # 2N one-hot + 3 progress + 4 spatial + (4*5) neighbor + rain one-hot.
        self.state_dim = (
            2 * self.num_nodes
            + 3
            + 4
            + self.max_neighbor_slots * self.neighbor_feature_dim
            + self.rain_dim
        )

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

    def _build_delivery_vec(self):
        delivery_vec = np.zeros(self.num_nodes)
        for d in self.delivery_nodes:
            if d not in self.completed:
                delivery_vec[d] = 1.0
        return delivery_vec

    def _build_target_features(self, unvisited):
        n_remaining = len(unvisited)
        n_completed = len(self.completed)
        n_remaining_norm = n_remaining / max(self.num_deliveries, 1)
        n_completed_norm = n_completed / max(self.num_deliveries, 1)
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

    def _build_neighbor_features(self):
        features = []
        neighbors = sorted(
            list(self.G.neighbors(self.current_node)),
            key=lambda nbr: self.G[self.current_node][nbr].get("length", 0.0),
        )
        for nbr in neighbors[: self.max_neighbor_slots]:
            edge = self.G[self.current_node][nbr]
            flood_score = edge.get("flood_score", 0.0)
            landslide_score = edge.get("landslide_score", 0.0)
            length_norm = min(edge.get("length", 0.0) / self.max_edge_length, 1.0)
            travel_time = edge.get("travel_time", None)
            travel_time_norm = 0.0 if travel_time is None else min(travel_time / self.max_episode_time, 1.0)
            feasible = 0.0 if edge.get("blocked", False) else 1.0
            features.extend([flood_score, landslide_score, length_norm, travel_time_norm, feasible])

        expected = self.max_neighbor_slots * self.neighbor_feature_dim
        if len(features) < expected:
            features.extend([0.0] * (expected - len(features)))
        return np.array(features, dtype=float)

    def _get_state(self):
        node_onehot = np.zeros(self.num_nodes)
        node_onehot[self.current_node] = 1.0
        delivery_vec = self._build_delivery_vec()
        unvisited = [d for d in self.delivery_nodes if d not in self.completed]
        target_feats = self._build_target_features(unvisited)
        neighbor_feats = self._build_neighbor_features()
        state = np.concatenate([node_onehot, delivery_vec, target_feats, neighbor_feats, self.rain_onehot])
        return torch.tensor(state, dtype=torch.float32)

    def get_action_mask(self):
        mask = np.zeros(self.num_nodes)
        for nbr in self.G.neighbors(self.current_node):
            if not self.G[self.current_node][nbr].get("blocked", False):
                mask[nbr] = 1.0
        return torch.tensor(mask, dtype=torch.float32)

    def step(self, action):
        self.steps += 1

        mask = self.get_action_mask()
        if mask[action] == 0:
            reward = self.failure_penalty("blockage")
            return self._get_state(), reward, True, {"termination_reason": "invalid_action"}

        unvisited_before = [d for d in self.delivery_nodes if d not in self.completed]
        d_before = self._nearest_unvisited_shortest(self.current_node, unvisited_before)

        edge = self.G[self.current_node][action]
        travel_time = edge["travel_time"]
        hf = edge["flood_score"]
        hl = edge["landslide_score"]

        self.total_time += travel_time
        self.total_hazard += (hf + hl)
        self.current_node = action

        delivery_reward = 0.0
        if action in self.delivery_nodes and action not in self.completed:
            self.completed.add(action)
            delivery_reward = self.reward_delivery

        unvisited_after = [d for d in self.delivery_nodes if d not in self.completed]
        d_after = self._nearest_unvisited_shortest(self.current_node, unvisited_after)

        progress_reward = self.k_progress * ((d_before - d_after) / self.max_shortest_len)
        hazard_penalty = -self.hazard_lambda * (self.w_flood * hf + self.w_landslide * hl)
        time_penalty = -self.eta_time * travel_time
        reward = delivery_reward + progress_reward + hazard_penalty + time_penalty

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
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
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
        q_values = model(state).clone()
        q_values[mask == 0] = -1e9
        return torch.argmax(q_values).item()


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
def train(
    num_episodes=1500,
    log_every=20,
    target_update_every_steps=500,
    eval_episodes=300,
    eval_every=20,
    graph_num_nodes=15,
    graph_min_nodes=12,
    graph_max_nodes=20,
):
    base_graph = create_base_graph(
        num_nodes=graph_num_nodes,
        min_nodes=graph_min_nodes,
        max_nodes=graph_max_nodes,
    )
    env = HazardRoutingEnv(base_graph)

    online = DQN(env.state_dim, env.num_nodes)
    target = DQN(env.state_dim, env.num_nodes)
    target.load_state_dict(online.state_dict())

    optimizer = optim.Adam(online.parameters(), lr=3e-4)
    buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    batch_size = 32
    train_steps = 0

    print(
        f"Graph stats | Nodes: {env.num_nodes}, Edges: {base_graph.number_of_edges()}, "
        f"AvgBaseTime: {env.avg_base_time:.4f}, AvgHazard: {env.avg_edge_hazard:.4f}, "
        f"StateDim: {env.state_dim}, Tmax(min): {env.max_elapsed_time:.2f}"
    )

    reward_history = []
    success_history = []
    best_eval_success = -1.0
    best_eval_reward = -1e9
    best_episode = 0

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_dir / "best_model.pt"
    last_model_path = checkpoints_dir / "last_model.pt"

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
            next_mask = env.get_action_mask() if not done else torch.zeros(env.num_nodes, dtype=torch.float32)

            buffer.store((state, action, reward, next_state, done, next_mask))
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones, next_masks = buffer.sample(batch_size)

                q_values = online(states)
                q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_online_q = online(next_states)
                    next_online_q[next_masks == 0] = -1e9
                    next_actions = next_online_q.argmax(dim=1)
                    next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
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
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % log_every == 0:
            avg_reward = float(np.mean(reward_history[-log_every:]))
            success_rate = float(np.mean(success_history[-log_every:]))
            print(
                f"Episode {episode + 1}, "
                f"LastReward: {total_reward:.2f}, "
                f"AvgReward({log_every}): {avg_reward:.2f}, "
                f"SuccessRate({log_every}): {success_rate:.2%}, "
                f"Epsilon: {epsilon:.3f}"
            )

        if (episode + 1) % eval_every == 0:
            eval_reward_eps0, eval_success_eps0 = evaluate_policy(online, env, num_episodes=eval_episodes, epsilon=0.0)
            eval_reward_eps005, eval_success_eps005 = evaluate_policy(
                online, env, num_episodes=eval_episodes, epsilon=0.05
            )
            print(
                f"[Eval @ Episode {episode + 1}] "
                f"eps=0.0 -> MeanReward: {eval_reward_eps0:.2f}, SuccessRate: {eval_success_eps0:.2%} | "
                f"eps=0.05 -> MeanReward: {eval_reward_eps005:.2f}, SuccessRate: {eval_success_eps005:.2%}"
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
                        "eval_success_rate_eps0": best_eval_success,
                        "eval_mean_reward_eps0": best_eval_reward,
                        "graph_num_nodes": env.num_nodes,
                        "seed": SEED,
                    },
                    best_model_path,
                )
                print(
                    f"Saved best checkpoint: {best_model_path} "
                    f"(episode {best_episode}, success={best_eval_success:.2%}, reward={best_eval_reward:.2f})"
                )

    torch.save(
        {
            "episode": num_episodes,
            "model_state_dict": online.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "graph_num_nodes": env.num_nodes,
            "seed": SEED,
        },
        last_model_path,
    )
    print(f"Saved last checkpoint: {last_model_path}")

    final_reward_eps0, final_success_eps0 = evaluate_policy(online, env, num_episodes=eval_episodes, epsilon=0.0)
    final_reward_eps005, final_success_eps005 = evaluate_policy(online, env, num_episodes=eval_episodes, epsilon=0.05)
    print(
        f"Evaluation over {eval_episodes} episodes | "
        f"eps=0.0 MeanReward: {final_reward_eps0:.2f}, SuccessRate: {final_success_eps0:.2%} | "
        f"eps=0.05 MeanReward: {final_reward_eps005:.2f}, SuccessRate: {final_success_eps005:.2%}"
    )
    print(
        f"Best checkpoint summary | Episode: {best_episode}, "
        f"eps=0.0 MeanReward: {best_eval_reward:.2f}, SuccessRate: {best_eval_success:.2%}"
    )


if __name__ == "__main__":
    train()

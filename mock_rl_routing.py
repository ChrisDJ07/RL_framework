import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

from graph_utils import get_raw_osm_graph, to_training_graph

# Reproducibility
SEED = 40
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# OSM-Based Graph (Iligan City, around MSU-IIT)
def create_base_graph(num_nodes=15, min_nodes=12, max_nodes=20, force_download=False):
    raw_graph = get_raw_osm_graph(min_nodes=min_nodes, force_download=force_download)
    return to_training_graph(raw_graph, num_nodes=num_nodes, min_nodes=min_nodes, max_nodes=max_nodes)

def visualize_graph(G, title="Graph Visualization", highlight_start=None, highlight_deliveries=None):
    pos = nx.get_node_attributes(G, "pos")

    plt.figure(figsize=(6, 6))

    # Draw Nodes
    node_colors = []

    for node in G.nodes():
        if highlight_start is not None and node == highlight_start:
            node_colors.append("green")
        elif highlight_deliveries is not None and node in highlight_deliveries:
            node_colors.append("red")
        else:
            node_colors.append("skyblue")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)

    # Draw Edges
    edge_colors = []
    widths = []

    for u, v, data in G.edges(data=True):
        hazard_intensity = data["flood_score"] + data["landslide_score"]

        # Normalize for color scaling
        hazard_clamped = min(hazard_intensity / 2.0, 1.0)

        if data.get("blocked", False):
            edge_colors.append("black")
            widths.append(3)
        else:
            # Blue (low hazard) → Red (high hazard)
            edge_colors.append((hazard_clamped, 0, 1 - hazard_clamped))
            widths.append(1 + hazard_clamped * 2)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths)

    # Draw Labels
    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis("off")
    plt.show()


# Hazard Activation Model
RAIN_LEVELS = {
    "RI1": {"block_mult": 0.1, "slow_mult": 1.1},
    "RI2": {"block_mult": 0.2, "slow_mult": 1.2},
    "RI3": {"block_mult": 0.3, "slow_mult": 1.3},
}
RAIN_KEYS = list(RAIN_LEVELS.keys())

def activate_hazards(G_base, rain_key):
    rain = RAIN_LEVELS[rain_key]
    G = G_base.copy()

    for u, v, data in G.edges(data=True):
        hazard_intensity = data["flood_score"] + data["landslide_score"]
        p_block = min(hazard_intensity * rain["block_mult"], 1.0)

        if np.random.rand() < p_block:
            data["blocked"] = True
        else:
            data["blocked"] = False
            data["travel_time"] = data["base_time"] * rain["slow_mult"]

    return G


# Environment Definition
class HazardRoutingEnv:
    def __init__(self, base_graph, num_deliveries=2):
        self.base_graph = base_graph
        self.num_nodes = base_graph.number_of_nodes()
        self.num_deliveries = min(num_deliveries, self.num_nodes - 1)
        self.max_steps = max(50, self.num_nodes * 2)
        self.rain_dim = len(RAIN_KEYS)

        edge_data = list(base_graph.edges(data=True))
        avg_base_time = np.mean([d["base_time"] for _, _, d in edge_data]) if edge_data else 1.0
        avg_hazard = np.mean([d["flood_score"] + d["landslide_score"] for _, _, d in edge_data]) if edge_data else 1.0
        self.avg_base_time = max(float(avg_base_time), 1e-6)
        self.avg_edge_hazard = max(float(avg_hazard), 1e-6)
        self.time_weight = 1.0
        self.hazard_weight = 0.5

    def reset(self):
        rain_key = random.choice(RAIN_KEYS)
        self.G = activate_hazards(self.base_graph, rain_key)
        rain_idx = RAIN_KEYS.index(rain_key)
        self.rain_onehot = np.zeros(self.rain_dim, dtype=float)
        self.rain_onehot[rain_idx] = 1.0

        self.current_node = random.randint(0, self.num_nodes - 1)

        all_nodes = list(self.G.nodes())
        all_nodes.remove(self.current_node)
        self.delivery_nodes = set(random.sample(all_nodes, self.num_deliveries))
        self.completed = set()

        self.total_time = 0
        self.total_hazard = 0
        self.steps = 0

        return self._get_state()


    def _get_state(self):
        node_onehot = np.zeros(self.num_nodes)
        node_onehot[self.current_node] = 1

        delivery_vec = np.zeros(self.num_nodes)
        for d in self.delivery_nodes:
            if d not in self.completed:
                delivery_vec[d] = 1

        state = np.concatenate([node_onehot, delivery_vec, self.rain_onehot])
        return torch.tensor(state, dtype=torch.float32)

    # Action Masking
    def get_action_mask(self):
        mask = np.zeros(self.num_nodes)

        for neighbor in self.G.neighbors(self.current_node):
            edge_data = self.G[self.current_node][neighbor]
            if not edge_data.get("blocked", False):
                mask[neighbor] = 1

        return torch.tensor(mask, dtype=torch.float32)
    
    # Stop function
    def step(self, action):
        self.steps += 1

        mask = self.get_action_mask()
        if mask[action] == 0:
            return self._get_state(), -50.0, True, {}

        edge_data = self.G[self.current_node][action]
        travel_time = edge_data["travel_time"]
        hazard = edge_data["flood_score"] + edge_data["landslide_score"]

        self.total_time += travel_time
        self.total_hazard += hazard

        self.current_node = action

        norm_time = travel_time / self.avg_base_time
        norm_hazard = hazard / self.avg_edge_hazard
        reward = -self.time_weight * norm_time - self.hazard_weight * norm_hazard

        if action in self.delivery_nodes:
            self.completed.add(action)

        done = False

        if len(self.completed) == len(self.delivery_nodes):
            reward += 100
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}


# DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=5000):
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
                total_reward += -50.0
                break

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if len(env.completed) == len(env.delivery_nodes):
            successes += 1

    model.train()
    return float(np.mean(rewards)), successes / max(num_episodes, 1)


# Training loop
def train(
    num_episodes=1000,
    log_every=20,
    target_update_every=20,
    eval_episodes=100,
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

    state_dim = 2 * env.num_nodes + env.rain_dim
    action_dim = env.num_nodes

    online = DQN(state_dim, action_dim)
    target = DQN(state_dim, action_dim)
    target.load_state_dict(online.state_dict())

    optimizer = optim.Adam(online.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    batch_size = 32

    print(
        f"Graph stats | Nodes: {env.num_nodes}, Edges: {base_graph.number_of_edges()}, "
        f"AvgBaseTime: {env.avg_base_time:.4f}, AvgHazard: {env.avg_edge_hazard:.4f}"
    )

    reward_history = []
    success_history = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            mask = env.get_action_mask()
            action = select_action(online, state, mask, epsilon)
            if action is None:
                # No traversable roads from current node under active hazards.
                total_reward += -50.0
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

        reward_history.append(total_reward)
        success_history.append(1 if len(env.completed) == len(env.delivery_nodes) else 0)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % target_update_every == 0:
            target.load_state_dict(online.state_dict())

        if (episode + 1) % log_every == 0:
            window_rewards = reward_history[-log_every:]
            window_success = success_history[-log_every:]
            avg_reward = float(np.mean(window_rewards))
            success_rate = float(np.mean(window_success))
            print(
                f"Episode {episode + 1}, "
                f"LastReward: {total_reward:.2f}, "
                f"AvgReward({log_every}): {avg_reward:.2f}, "
                f"SuccessRate({log_every}): {success_rate:.2%}, "
                f"Epsilon: {epsilon:.3f}"
            )

    eval_mean_reward, eval_success_rate = evaluate_policy(
        online, env, num_episodes=eval_episodes, epsilon=0.0
    )
    print(
        f"Evaluation over {eval_episodes} episodes | "
        f"MeanReward: {eval_mean_reward:.2f}, "
        f"SuccessRate: {eval_success_rate:.2%}"
    )


# Run training
if __name__ == "__main__":    
    train()
    # visualize_graph(create_base_graph(num_nodes=30, min_nodes=30, max_nodes=40), title="Base Graph (No Hazards)")

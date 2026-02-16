import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# =====================
# Reproducibility
# =====================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Mock Graph (With Hazard Scores)
def create_base_graph(num_nodes=10, k_neighbors=3):
    G = nx.Graph()

    # Create node positions
    positions = {i: np.random.rand(2) for i in range(num_nodes)}

    for i in range(num_nodes):
        G.add_node(i, pos=positions[i])

    # Connect to nearest neighbors
    for i in range(num_nodes):
        distances = []
        for j in range(num_nodes):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append((j, dist))
        distances.sort(key=lambda x: x[1])

        for j, dist in distances[:k_neighbors]:
            if not G.has_edge(i, j):
                G.add_edge(
                    i, j,
                    length=dist,
                    base_time=dist,
                    flood_score=np.random.rand(),
                    landslide_score=np.random.rand()
                )

    return G


# Hazard Activation Model
RAIN_LEVELS = {
    "RI1": {"block_mult": 0.2, "slow_mult": 1.1},
    "RI2": {"block_mult": 0.4, "slow_mult": 1.3},
    "RI3": {"block_mult": 0.6, "slow_mult": 1.6},
}

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
        self.num_deliveries = num_deliveries
        self.max_steps = 50

    def reset(self):
        rain_key = random.choice(list(RAIN_LEVELS.keys()))
        self.G = activate_hazards(self.base_graph, rain_key)
        self.rain_scalar = list(RAIN_LEVELS.keys()).index(rain_key) / 2.0

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

        state = np.concatenate([node_onehot, delivery_vec, [self.rain_scalar]])
        return torch.tensor(state, dtype=torch.float32)

    # State Representation
    def _get_state(self):
        node_onehot = np.zeros(self.num_nodes)
        node_onehot[self.current_node] = 1

        delivery_vec = np.zeros(self.num_nodes)
        for d in self.delivery_nodes:
            if d not in self.completed:
                delivery_vec[d] = 1

        state = np.concatenate([node_onehot, delivery_vec, [self.rain_scalar]])
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

        reward = -travel_time - 0.5 * hazard

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
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# Training loop
def train():
    base_graph = create_base_graph(num_nodes=10)
    env = HazardRoutingEnv(base_graph)

    state_dim = 2 * 10 + 1
    action_dim = 10

    online = DQN(state_dim, action_dim)
    target = DQN(state_dim, action_dim)
    target.load_state_dict(online.state_dict())

    optimizer = optim.Adam(online.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    batch_size = 64

    for episode in range(300):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            mask = env.get_action_mask()

            if random.random() < epsilon:
                valid_actions = torch.where(mask == 1)[0]
                action = random.choice(valid_actions).item()
            else:
                with torch.no_grad():
                    q_values = online(state)
                    q_values[mask == 0] = -1e9
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)

            buffer.store((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                q_values = online(states)
                q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_actions = online(next_states).argmax(dim=1)
                    next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                    target_q = rewards + gamma * next_q * (1 - dones)

                loss = nn.MSELoss()(q_selected, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 20 == 0:
            target.load_state_dict(online.state_dict())
            print(f"Episode {episode}, Reward: {total_reward:.2f}")


# Run training
if __name__ == "__main__":
    train()
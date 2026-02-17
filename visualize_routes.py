import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D

import mock_rl_routing as mrr


def load_env_and_model(config_path="experiment_config.json", checkpoint_path="checkpoints/best_model.pt"):
    cfg = mrr.load_config(config_path)
    mrr.set_seed(cfg["seed"])
    mrr.apply_runtime_config(cfg)

    graph_cfg = cfg["graph"]
    env_cfg = cfg["environment"]
    reward_cfg = cfg["reward"]
    model_cfg = cfg["model"]

    base_graph = mrr.create_base_graph(
        num_nodes=int(graph_cfg["num_nodes"]),
        min_nodes=int(graph_cfg["min_nodes"]),
        max_nodes=int(graph_cfg["max_nodes"]),
        force_download=bool(graph_cfg.get("force_download", False)),
    )
    env = mrr.HazardRoutingEnv(
        base_graph,
        num_deliveries=int(env_cfg["num_deliveries"]),
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
    )

    model = mrr.DQN(env.state_dim, env.num_nodes, hidden_sizes=tuple(model_cfg["hidden_sizes"]))

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return env, model


def run_episode(env, model, epsilon=0.0):
    state = env.reset()
    rain_key = mrr.RAIN_KEYS[int(np.argmax(env.rain_onehot))]

    start_node = env.current_node
    delivery_nodes = set(env.delivery_nodes)

    path_nodes = [start_node]
    path_edges = []
    total_reward = 0.0
    done = False
    reason = None

    while not done:
        mask = env.get_action_mask()
        action = mrr.select_action(model, state, mask, epsilon)

        if action is None:
            total_reward += env.failure_penalty("blockage")
            reason = "no_valid_action"
            break

        prev = env.current_node
        next_state, reward, done, info = env.step(action)
        path_edges.append((prev, action))
        path_nodes.append(action)
        total_reward += reward
        state = next_state
        reason = info.get("termination_reason")

    success = len(env.completed) == len(env.delivery_nodes)
    blocked_edges = sum(1 for _, _, d in env.G.edges(data=True) if d.get("blocked", False))
    route_hazard = 0.0
    for u, v in path_edges:
        data = env.G[u][v]
        route_hazard += float(data.get("flood_score", 0.0) + data.get("landslide_score", 0.0))

    return {
        "graph": env.G.copy(),
        "rain_key": rain_key,
        "start_node": start_node,
        "delivery_nodes": delivery_nodes,
        "completed_nodes": set(env.completed),
        "path_nodes": path_nodes,
        "path_edges": path_edges,
        "total_reward": total_reward,
        "success": success,
        "reason": reason,
        "blocked_edges": blocked_edges,
        "route_hazard": route_hazard,
        "steps": len(path_edges),
    }


def _draw_episode(ax, result, episode_idx):
    G = result["graph"]
    pos = nx.get_node_attributes(G, "pos")

    # Base edges: hazard-colored, blocked in black.
    non_blocked_edges = []
    non_blocked_colors = []
    non_blocked_widths = []
    blocked_edges = []
    for u, v, d in G.edges(data=True):
        if d.get("blocked", False):
            blocked_edges.append((u, v))
        else:
            hz = float(d.get("flood_score", 0.0) + d.get("landslide_score", 0.0))
            hz_norm = min(hz / 2.0, 1.0)
            non_blocked_edges.append((u, v))
            non_blocked_colors.append((hz_norm, 0.2, 1.0 - hz_norm))
            non_blocked_widths.append(1.2)

    if non_blocked_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=non_blocked_edges,
            edge_color=non_blocked_colors,
            width=non_blocked_widths,
            alpha=0.85,
        )
    if blocked_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=blocked_edges,
            edge_color="black",
            width=2.5,
            alpha=0.95,
        )

    # Route path overlay.
    if result["path_edges"]:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=result["path_edges"],
            edge_color="#FFD400",
            width=3.2,
            alpha=0.95,
        )

    # Nodes.
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color="#d9d9d9", node_size=120, edgecolors="white", linewidths=0.8
    )

    start = result["start_node"]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=[start], node_color="#1f9d55", node_size=190, edgecolors="white", linewidths=1.0
    )

    deliveries = list(result["delivery_nodes"])
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=deliveries, node_color="#d62828", node_size=170, edgecolors="white", linewidths=1.0
    )

    completed = list(result["completed_nodes"])
    if completed:
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=completed,
            node_color="#ff9f1c",
            node_size=90,
            edgecolors="white",
            linewidths=0.8,
        )

    # Labels only for key nodes.
    sx, sy = pos[start]
    ax.text(sx, sy, "S", fontsize=8, color="white", ha="center", va="center", fontweight="bold")
    for d in deliveries:
        dx, dy = pos[d]
        ax.text(dx, dy, "D", fontsize=7, color="white", ha="center", va="center", fontweight="bold")

    status = "SUCCESS" if result["success"] else "FAIL"
    ax.set_title(
        f"Ep {episode_idx + 1} | {status}\n"
        f"Rain={result['rain_key']} Steps={result['steps']} Reward={result['total_reward']:.1f}\n"
        f"Blocked={result['blocked_edges']} RouteHaz={result['route_hazard']:.2f} Reason={result['reason']}",
        fontsize=8,
    )
    ax.set_axis_off()


def visualize_episodes(
    config_path="experiment_config.json",
    checkpoint_path="checkpoints/best_model.pt",
    num_episodes=12,
    epsilon=0.0,
    cols=5,
    save_path="checkpoints/route_visualization.png",
):
    env, model = load_env_and_model(config_path=config_path, checkpoint_path=checkpoint_path)
    results = [run_episode(env, model, epsilon=epsilon) for _ in range(num_episodes)]

    rows = math.ceil(num_episodes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.2))
    axes = np.array(axes).reshape(-1)

    for i in range(num_episodes):
        _draw_episode(axes[i], results[i], i)

    for j in range(num_episodes, len(axes)):
        axes[j].set_axis_off()

    legend_handles = [
        Line2D([0], [0], color="#FFD400", lw=3.0, label="Traversed route"),
        Line2D([0], [0], color="black", lw=2.5, label="Blocked edge"),
        Line2D([0], [0], color="#7a7a7a", lw=1.2, label="Passable edge (hazard-colored)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f9d55", label="Start", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62828", label="Delivery target", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff9f1c", label="Completed delivery", markersize=7),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle(
        f"Routing Visualizer | Episodes={num_episodes} | Policy epsilon={epsilon:.2f} | Checkpoint={Path(checkpoint_path).name}",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_file, dpi=180, bbox_inches="tight")
    print(f"Saved visualization to: {save_file}")
    plt.show()


if __name__ == "__main__":
    visualize_episodes(
        config_path="experiment_config.json",
        checkpoint_path="checkpoints/best_model.pt",
        num_episodes=12,
        epsilon=0.0,
        cols=4,
        save_path="checkpoints/route_visualization.png",
    )

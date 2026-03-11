"""
Visualize routing episodes from a trained RL model checkpoint. This script loads the 
specified checkpoint, reconstructs the environment and model, and runs multiple episodes
to visualize the agent's routing decisions in the context of the underlying graph and hazards.
The resulting visualizations show the traversed route, blocked edges, hazard levels,
and key nodes (start, delivery targets, completed deliveries). Command-line options allow 
customization of the number of episodes, epsilon for action selection, layout of subplots,
and output path for saving the visualization.
"""

import math
import importlib
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_and_model(
    config_path="configs/experiment_config.json",
    checkpoint_path="checkpoints/best_model.pt",
    seed=None,
    use_config_seed=True,
):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    candidate_modules = [
        "rl_routing",
        "rl_routing_wCUDA",
        "rl_routing_curriculum",
        "mock_rl_routing",
    ]
    last_error = None

    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            last_error = e
            continue
        cfg = module.load_config(config_path)
        resolved_seed = seed
        if resolved_seed is None and use_config_seed:
            resolved_seed = cfg["seed"]
        if resolved_seed is not None:
            module.set_seed(resolved_seed)

        module.apply_runtime_config(cfg)

        graph_cfg = cfg["graph"]
        env_cfg = cfg["environment"]
        reward_cfg = cfg["reward"]
        model_cfg = cfg["model"]

        if "base_graph_node_link" in checkpoint:
            base_graph = nx.node_link_graph(checkpoint["base_graph_node_link"], edges="edges")
            print(f"Visualizer: loaded base graph snapshot from checkpoint ({module_name}).")
        else:
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
            print(f"Visualizer: checkpoint has no graph snapshot, rebuilt graph from config ({module_name}).")

        env = module.HazardRoutingEnv(
            base_graph,
            num_deliveries=int(env_cfg["num_deliveries"]),
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
        )
        action_dim = int(getattr(env, "action_dim", env.num_nodes))
        try:
            model = module.DQN(
                env.state_dim,
                action_dim,
                num_nodes=env.num_nodes,
                num_delivery_slots=env.num_deliveries,
                hidden_sizes=tuple(model_cfg["hidden_sizes"]),
                node_embedding_dim=int(model_cfg.get("node_embedding_dim", 16)),
            )
        except TypeError:
            model = module.DQN(env.state_dim, action_dim, hidden_sizes=tuple(model_cfg["hidden_sizes"]))
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"Visualizer: loaded checkpoint with module {module_name}.")
            return env, model, module
        except RuntimeError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not load checkpoint with supported model variants: {', '.join(candidate_modules)}."
    ) from last_error


def run_episode(env, model, module, epsilon=0.0):
    state = env.reset()
    rain_key = module.RAIN_KEYS[int(np.argmax(env.rain_onehot))]

    start_node = env.current_node
    delivery_nodes = set(env.delivery_nodes)

    path_nodes = [start_node]
    path_edges = []
    total_reward = 0.0
    done = False
    reason = None

    while not done:
        mask = env.get_action_mask()
        action = module.select_action(model, state, mask, epsilon)

        if action is None:
            total_reward += env.failure_penalty("blockage")
            reason = "no_valid_action"
            break

        prev = env.current_node
        next_state, reward, done, info = env.step(action)
        next_node = env.current_node
        path_edges.append((prev, next_node))
        path_nodes.append(next_node)
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
        G, pos, ax=ax, node_color="#d9d9d9", node_size=20, edgecolors="white", linewidths=0.8
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


def _resolve_save_path(save_path):
    if save_path:
        return Path(save_path)
    return Path("results/visualization_runs") / "last_run_visualization.png"


def visualize_episodes(
    config_path="configs/experiment_config.json",
    checkpoint_path="checkpoints/best_model.pt",
    num_episodes=12,
    epsilon=0.0,
    cols=5,
    save_path=None,
    seed=None,
    use_config_seed=False,
):
    env, model, module = load_env_and_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        seed=seed,
        use_config_seed=use_config_seed,
    )
    results = [run_episode(env, model, module, epsilon=epsilon) for _ in range(num_episodes)]

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
        f"Routing Visualizer | Episodes={num_episodes} | Policy epsilon={epsilon:.2f} | "
        f"Seed={'config' if use_config_seed and seed is None else seed} | Checkpoint={Path(checkpoint_path).name}",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    save_file = _resolve_save_path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_file, dpi=180, bbox_inches="tight")
    print(f"Saved visualization to: {save_file}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RL routing episodes from a checkpoint.")
    parser.add_argument("--config-path", type=str, default="configs/experiment_config.json")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--num-episodes", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Optional explicit output path. Leave empty to use results/visualization_runs/last_run_visualization.png.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override. Ignored unless provided.",
    )
    parser.add_argument(
        "--use-config-seed",
        action="store_true",
        help="Use seed from config when --seed is not provided.",
    )
    args = parser.parse_args()

    visualize_episodes(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        num_episodes=args.num_episodes,
        epsilon=args.epsilon,
        cols=args.cols,
        save_path=(args.save_path if args.save_path else None),
        seed=args.seed,
        use_config_seed=args.use_config_seed,
    )
    
# python utils/visualize_routes.py --checkpoint-path checkpoints/no_hazard_control/best_model.pt --save-path results/visualization_runs/my_routes.png
# python utils/visualize_routes.py --save-path results/visualization_runs/pretrain_routes.png --config-path configs/no_hazard_training/no_hazard_config.json --num-episodes 6 --cols 3

# Nodes: 150
# python utils/visualize_routes.py --checkpoint-path checkpoints/staged_training/stage_150_further/best_model.pt --save-path results/visualization_runs/my_routes_150_further_test.png --config-path configs/staged_training/stage_150_further.json --num-episodes 1 --cols 1

#Nodes: 200
# python utils/visualize_routes.py --checkpoint-path checkpoints/stage_control_1_200n_further/best_model.pt --save-path results/visualization_runs/my_routes_200_further_d5_test.png --config-path configs/staged_training/stage_control_1_200n_further.json --num-episodes 1 --cols 1
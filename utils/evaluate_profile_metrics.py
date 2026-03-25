"""
Compare the three warm-start 100-node routing profiles using manuscript-aligned metrics.

Why this script exists
----------------------
Your manuscript already defines the key evaluation metrics that matter for profile
comparison:

1. Success Rate
2. Average Travel Time for Successful Episodes
3. Hazard Exposure Score

This utility adds one practical diagnostic that has been useful in the codebase:

4. Timeout Rate

The script runs evaluation episodes for each profile checkpoint, aggregates those
metrics, saves a single comparison image, and writes a raw per-episode CSV for
documentation and later analysis.

Current scope
-------------
- Focused on the three 100-node warm-start profiles by default:
  `balanced_warm`, `safe_warm`, and `fast_warm`
- Supports custom comparisons via repeated `--spec` flags
- Uses greedy evaluation by default (`epsilon=0.0`) because that is the cleanest
  way to compare the learned policies themselves

Hazard exposure implementation
------------------------------
The manuscript describes hazard exposure as cumulative hazard-weighted distance over
the traversed route. In code, this is implemented as:

    sum( (w_flood * flood_score + w_landslide * landslide_score) * edge_length )

for each traversed edge.

To keep route-quality comparisons fair, both travel time and hazard exposure are
summarized over successful episodes only. Success rate and timeout rate are computed
over all episodes.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ProfileSpec:
    label: str
    checkpoint_path: str
    config_path: str


DEFAULT_PROFILE_SPECS = [
    ProfileSpec(
        label="Balanced",
        checkpoint_path="checkpoints/staged_training_new/stage_100_balanced_warm/best_model.pt",
        config_path="configs/stage_training_new/stage_100_balanced_warm.json",
    ),
    ProfileSpec(
        label="Safe",
        checkpoint_path="checkpoints/staged_training_new/stage_100_safe_warm/best_model.pt",
        config_path="configs/stage_training_new/stage_100_safe_warm.json",
    ),
    ProfileSpec(
        label="Fast",
        checkpoint_path="checkpoints/staged_training_new/stage_100_fast_warm/best_model.pt",
        config_path="configs/stage_training_new/stage_100_fast_warm.json",
    ),
]


PROFILE_COLORS = {
    "Balanced": "#457b9d",
    "Safe": "#2a9d8f",
    "Fast": "#f4a261",
}


def parse_spec_arg(raw_spec: str) -> ProfileSpec:
    """Parse a single CLI profile spec entry.

    Expected format:
        Label|checkpoint_path|config_path
    """
    parts = [p.strip() for p in raw_spec.split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Each --spec must follow the format: Label|checkpoint_path|config_path"
        )
    return ProfileSpec(label=parts[0], checkpoint_path=parts[1], config_path=parts[2])


def load_env_and_model(config_path: str, checkpoint_path: str, seed: int | None = None):
    """Load the environment and model from a config/checkpoint pair.

    The repository has gone through several trainer variants over time, so this
    loader tries the currently relevant modules in a stable order.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    candidate_modules = [
        "rl_routing_wCUDA_wCheckP",
        "rl_routing_wCUDA_wCheckP_latest",
        "rl_routing",
        "rl_routing_wCUDA",
        "rl_routing_curriculum",
        "mock_rl_routing",
    ]
    last_error = None

    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue

        try:
            cfg = module.load_config(config_path)
            resolved_seed = seed if seed is not None else cfg.get("seed")
            if resolved_seed is not None:
                module.set_seed(int(resolved_seed))
            module.apply_runtime_config(cfg)

            graph_cfg = cfg["graph"]
            env_cfg = cfg["environment"]
            reward_cfg = cfg["reward"]
            model_cfg = cfg["model"]

            if "base_graph_node_link" in checkpoint:
                base_graph = nx.node_link_graph(checkpoint["base_graph_node_link"], edges="edges")
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

            env = module.HazardRoutingEnv(
                base_graph,
                num_deliveries=int(env_cfg["num_deliveries"]),
                env_cfg=env_cfg,
                reward_cfg=reward_cfg,
            )
            action_dim = int(getattr(env, "action_dim", env.num_nodes))
            model = module.DQN(
                env.state_dim,
                action_dim,
                num_nodes=env.num_nodes,
                num_delivery_slots=env.num_deliveries,
                hidden_sizes=tuple(model_cfg["hidden_sizes"]),
                node_embedding_dim=int(model_cfg.get("node_embedding_dim", 16)),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return env, model, module
        except Exception as exc:  # pragma: no cover - fallback chain is intentional
            last_error = exc
            continue

    raise RuntimeError(
        "Could not load checkpoint/config pair with the supported trainer modules."
    ) from last_error


def evaluate_profile(profile_spec: ProfileSpec, num_episodes: int, epsilon: float) -> pd.DataFrame:
    """Run evaluation episodes and return raw per-episode rows."""
    env, model, module = load_env_and_model(
        config_path=profile_spec.config_path,
        checkpoint_path=profile_spec.checkpoint_path,
    )

    rows: list[dict] = []
    model.eval()

    for episode_idx in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        reason = None
        path_hazard_sum = 0.0
        hazard_exposure = 0.0
        total_distance = 0.0
        total_reward = 0.0

        while not done:
            mask = env.get_action_mask()
            action = module.select_action(model, state, mask, epsilon)
            if action is None:
                total_reward += env.failure_penalty("blockage")
                reason = "no_valid_action"
                break

            prev_node = env.current_node
            next_state, reward, done, info = env.step(action)
            edge_data = env.G[prev_node][env.current_node]

            flood_score = float(edge_data.get("flood_score", 0.0))
            landslide_score = float(edge_data.get("landslide_score", 0.0))
            edge_length = float(edge_data.get("length", 0.0))

            path_hazard_sum += env.w_flood * flood_score + env.w_landslide * landslide_score
            hazard_exposure += (env.w_flood * flood_score + env.w_landslide * landslide_score) * edge_length
            total_distance += edge_length

            total_reward += float(reward)
            state = next_state
            if done:
                reason = info.get("termination_reason")

        success = len(env.completed) == len(env.delivery_nodes)
        timeout = reason in {"timeout", "step_guard_timeout"}

        rows.append(
            {
                "profile": profile_spec.label,
                "episode": episode_idx,
                "config_path": profile_spec.config_path,
                "checkpoint_path": profile_spec.checkpoint_path,
                "success": int(success),
                "timeout": int(timeout),
                "termination_reason": reason or ("success" if success else "unknown"),
                "steps": int(env.steps),
                "total_time": float(env.total_time),
                "total_distance": float(total_distance),
                "total_reward": float(total_reward),
                "hazard_exposure": float(hazard_exposure),
                "hazard_score_sum": float(path_hazard_sum),
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate manuscript-aligned profile metrics from raw episode rows."""
    summaries = []

    for profile, group in raw_df.groupby("profile", sort=False):
        successful = group[group["success"] == 1]
        summaries.append(
            {
                "profile": profile,
                "success_rate_pct": float(group["success"].mean() * 100.0),
                "avg_time_success": float(successful["total_time"].mean()) if not successful.empty else np.nan,
                "avg_hazard_exposure_success": float(successful["hazard_exposure"].mean())
                if not successful.empty
                else np.nan,
                "timeout_rate_pct": float(group["timeout"].mean() * 100.0),
                "episodes": int(len(group)),
                "num_success": int(group["success"].sum()),
            }
        )

    return pd.DataFrame(summaries)


def plot_metric_comparison(summary_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Render a 2x2 bar-chart dashboard for the key comparison metrics."""
    profiles = summary_df["profile"].tolist()
    colors = [PROFILE_COLORS.get(name, "#6c757d") for name in profiles]

    metrics = [
        ("success_rate_pct", "Success Rate (%)"),
        ("avg_time_success", "Avg Travel Time (Successful Episodes)"),
        ("avg_hazard_exposure_success", "Hazard Exposure (Successful Episodes)"),
        ("timeout_rate_pct", "Timeout Rate (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, (column, display_name) in zip(axes, metrics):
        values = summary_df[column].tolist()
        bars = ax.bar(profiles, values, color=colors, edgecolor="#1f1f1f", linewidth=0.8)
        ax.set_title(display_name)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)

        for bar, value in zip(bars, values):
            if pd.isna(value):
                label = "n/a"
                ypos = 0.0
            else:
                label = f"{value:.2f}"
                ypos = value
            offset = max(abs(ypos) * 0.015, 0.8)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                ypos + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_default_output_prefix(epsilon: float) -> str:
    eps_str = str(epsilon).replace(".", "p")
    return f"stage_100_profiles_eps_{eps_str}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare manuscript-aligned metrics across routing profiles."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes per profile.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Evaluation epsilon. Use 0.0 for greedy evaluation.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Optional custom profile spec in the format Label|checkpoint_path|config_path. "
        "Repeat this flag to compare multiple profiles. If omitted, the default three 100-node warm-start profiles are used.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Base filename prefix for outputs under results/evaluation.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")

    if args.spec:
        profile_specs = [parse_spec_arg(raw) for raw in args.spec]
    else:
        profile_specs = DEFAULT_PROFILE_SPECS

    output_prefix = args.output_prefix.strip() or build_default_output_prefix(args.epsilon)
    evaluation_dir = PROJECT_ROOT / "results" / "evaluation"
    image_path = evaluation_dir / f"{output_prefix}.png"
    csv_path = evaluation_dir / f"{output_prefix}_raw.csv"

    raw_frames = []
    for spec in profile_specs:
        print(f"Evaluating profile: {spec.label}")
        raw_frames.append(evaluate_profile(spec, num_episodes=args.episodes, epsilon=args.epsilon))

    raw_df = pd.concat(raw_frames, ignore_index=True)
    summary_df = summarize_metrics(raw_df)
    summary_df = summary_df.set_index("profile").loc[[spec.label for spec in profile_specs]].reset_index()

    title = (
        f"Profile Comparison | Episodes per Profile: {args.episodes} | "
        f"Epsilon: {args.epsilon:.2f}"
    )
    plot_metric_comparison(summary_df, title=title, output_path=image_path)

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(csv_path, index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved image: {image_path}")
    print(f"Saved raw CSV: {csv_path}")


if __name__ == "__main__":
    main()


# python utils/evaluate_profile_metrics.py --episodes 200 --epsilon 0.0

# python utils/evaluate_profile_metrics.py --episodes 200 --epsilon 0.0 --output-prefix stage_100_profile_eval

"""
python utils/evaluate_profile_metrics.py \
  --spec "Balanced|checkpoints/staged_training_new/stage_100_balanced_hard/best_model.pt|configs/stage_training_new/hard_training/stage_100_balanced_hard.json" \
  --spec "Safe|checkpoints/staged_training_new/stage_100_safe_hard/best_model.pt|configs/stage_training_new/hard_training/stage_100_safe_hard.json" \
  --spec "Fast|checkpoints/staged_training_new/stage_100_fast_hard/best_model.pt|configs/stage_training_new/hard_training/stage_100_fast_hard.json" \
  --output-prefix stage_100_hard_profiles_eval
"""

# Warm training new
"""
python utils/evaluate_profile_metrics.py \
  --spec "Balanced|checkpoints/warm_training_new/stage_100_balanced_warm_no_HF/best_model.pt|configs/stage_training_new/warm_training_new/stage_100_balanced_warm_no_HF.json" \
  --spec "Safe|checkpoints/warm_training_new/stage_100_safe_warm_no_HF/best_model.pt|configs/stage_training_new/warm_training_new/stage_100_safe_warm_no_HF.json" \
  --spec "Fast|checkpoints/warm_training_new/stage_100_fast_warm_no_HF/best_model.pt|configs/stage_training_new/warm_training_new/stage_100_fast_warm_no_HF.json" \
  --output-prefix stage_100_warm_profiles_eval_no_HF
"""

# Warm profile training new
"""
python utils/evaluate_profile_metrics.py \
  --spec "Balanced|checkpoints/profile_training_new/stage_200_balanced_warm/best_model.pt|configs/profile_training_new/stage_200_balanced_warm.json" \
  --spec "Safe|checkpoints/profile_training_new/stage_200_safe_warm/best_model.pt|configs/profile_training_new/stage_200_safe_warm.json" \
  --spec "Fast|checkpoints/profile_training_new/stage_200_fast_warm/best_model.pt|configs/profile_training_new/stage_200_fast_warm.json" \
  --output-prefix stage_200_warm_profiles_eval_no_HF
"""

# Warm and alt_lr and HF
"""
python utils/evaluate_profile_metrics.py \
  --spec "Balanced|checkpoints/profile_training_new/stage_200_balanced_warm/best_model.pt|configs/profile_training_new/stage_200_balanced_warm.json" \
  --spec "Safe|checkpoints/profile_training_new/stage_200_safe_warm/best_model.pt|configs/profile_training_new/stage_200_safe_warm.json" \
  --spec "Fast|checkpoints/profile_training_new/stage_200_fast_warm/best_model.pt|configs/profile_training_new/stage_200_fast_warm.json" \
  --spec "Balanced_alt_lr|checkpoints/profile_training_new/stage_200_balanced_warm_alt_lr/best_model.pt|configs/profile_training_new/stage_200_balanced_warm_alt_lr.json" \
  --spec "Safe_alt_lr|checkpoints/profile_training_new/stage_200_safe_warm_alt_lr/best_model.pt|configs/profile_training_new/stage_200_safe_warm_alt_lr.json" \
  --spec "Fast_alt_lr|checkpoints/profile_training_new/stage_200_fast_warm_alt_lr/best_model.pt|configs/profile_training_new/stage_200_fast_warm_alt_lr.json" \
  --spec "Balanced_HF|checkpoints/profile_training_new/stage_200_balanced_warm_HF/best_model.pt|configs/profile_training_new/with_HF/stage_200_balanced_warm_HF.json" \
  --spec "Safe_HF|checkpoints/profile_training_new/stage_200_safe_warm_HF/best_model.pt|configs/profile_training_new/with_HF/stage_200_safe_warm_HF.json" \
  --spec "Fast_HF|checkpoints/profile_training_new/stage_200_fast_warm_HF/best_model.pt|configs/profile_training_new/with_HF/stage_200_fast_warm_HF.json" \
  --output-prefix stage_200_warm_profiles_eval_no_HF_vs_alt_lr_vs_HF
"""
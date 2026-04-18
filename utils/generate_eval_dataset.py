from __future__ import annotations

import argparse
import random
from pathlib import Path

from eval_pipeline_common import (
    choose_episode_nodes_feasible,
    choose_episode_nodes_random,
    episode_blocked_edges,
    initialize_env_from_episode,
    load_env_from_config,
    write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a shared JSON evaluation dataset of routing episodes."
    )
    parser.add_argument("--config", required=True, help="Config used to build the graph/environment.")
    parser.add_argument("--episodes", type=int, required=True, help="Number of episodes to generate.")
    parser.add_argument(
        "--num-deliveries",
        type=int,
        default=None,
        help="Override number of delivery nodes per episode. Defaults to config value.",
    )
    parser.add_argument(
        "--rain-keys",
        type=str,
        default="",
        help="Comma-separated rainfall keys to sample from. Defaults to config active_rain_keys.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=40,
        help="Random seed for dataset generation.",
    )
    parser.add_argument(
        "--feasible-only",
        action="store_true",
        help="Generate only feasible-at-reset episodes.",
    )
    parser.add_argument(
        "--sample-feasible-directly",
        action="store_true",
        help="Sample start+delivery nodes directly from passable connected components.",
    )
    parser.add_argument(
        "--max-resamples",
        type=int,
        default=1000,
        help="Maximum resamples per episode when --feasible-only is enabled.",
    )
    parser.add_argument(
        "--include-blocked-edges",
        action="store_true",
        help="Store blocked edge lists in the dataset for easier audit/debugging.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.max_resamples <= 0:
        raise ValueError("--max-resamples must be > 0")

    random.seed(int(args.seed))

    loaded = load_env_from_config(
        config_path=args.config,
        seed=int(args.seed),
        num_deliveries_override=args.num_deliveries,
    )
    module = loaded.module
    cfg = loaded.cfg
    env = loaded.env

    if args.rain_keys.strip():
        rain_keys = [key.strip() for key in args.rain_keys.split(",") if key.strip()]
    else:
        rain_keys = list(module.ACTIVE_RAIN_KEYS)
    invalid = [key for key in rain_keys if key not in module.RAIN_KEYS]
    if invalid:
        raise ValueError(f"Unknown rain keys: {invalid}")

    episodes = []
    total_resamples = 0

    for episode_id in range(1, args.episodes + 1):
        attempts = 0
        while True:
            rain_key = random.choice(rain_keys)
            env.G = module.activate_hazards(env.base_graph, rain_key)

            if args.sample_feasible_directly:
                sampled = choose_episode_nodes_feasible(env)
                if sampled is None:
                    attempts += 1
                    total_resamples += 1
                    if attempts >= args.max_resamples:
                        raise RuntimeError(
                            f"Could not sample a feasible component for episode {episode_id} within {args.max_resamples} attempts."
                        )
                    continue
                start_node, delivery_nodes = sampled
            else:
                start_node, delivery_nodes = choose_episode_nodes_random(env)

            initialize_env_from_episode(env, module, rain_key, start_node, delivery_nodes)
            feasible = bool(env.is_current_episode_feasible())
            if feasible or not args.feasible_only:
                record = {
                    "episode_id": episode_id,
                    "rain_key": rain_key,
                    "start_node": int(start_node),
                    "delivery_nodes": [int(node) for node in delivery_nodes],
                    "feasible_at_reset": feasible,
                    "num_blocked_edges": int(sum(1 for _, _, d in env.G.edges(data=True) if d.get("blocked", False))),
                }
                if args.include_blocked_edges:
                    record["blocked_edges"] = episode_blocked_edges(env)
                episodes.append(record)
                total_resamples += attempts
                break

            attempts += 1
            total_resamples += 1
            if attempts >= args.max_resamples:
                raise RuntimeError(
                    f"Could not generate a feasible episode {episode_id} within {args.max_resamples} attempts."
                )

    graph_cfg = cfg["graph"]
    payload = {
        "schema_version": 1,
        "dataset_type": "routing_episode_dataset",
        "generation": {
            "config_path": str(Path(args.config)),
            "seed": int(args.seed),
            "episodes": int(args.episodes),
            "num_deliveries": int(env.num_deliveries),
            "rain_keys": rain_keys,
            "feasible_only": bool(args.feasible_only),
            "sample_feasible_directly": bool(args.sample_feasible_directly),
            "max_resamples": int(args.max_resamples),
            "total_resamples": int(total_resamples),
        },
        "graph": {
            "num_nodes": int(env.num_nodes),
            "num_edges": int(env.base_graph.number_of_edges()),
            "prebuilt_graphml_path": str(graph_cfg.get("prebuilt_graphml_path", "") or ""),
            "node_ids": [int(node) for node in env.base_graph.nodes()],
        },
        "episodes": episodes,
    }
    write_json(args.output, payload)
    print(f"Saved dataset: {args.output}")
    print(f"Episodes: {len(episodes)} | Total resamples: {total_resamples}")


if __name__ == "__main__":
    main()


# Generate shared dataset
'''
python utils/generate_eval_dataset.py \
  --config configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI2_det.json \
  --episodes 1000 \
  --num-deliveries 2 \
  --rain-keys RI1,RI2,RI3,RI4,RI5 \
  --output results/evaluation_datasets/eval_200n_d2_all_ri.json

'''

# Generate feasible-only dataset with direct sampling
'''
python utils/generate_eval_dataset.py \
  --config configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI2_det.json \
  --episodes 1000 \
  --num-deliveries 2 \
  --rain-keys RI1,RI2,RI3,RI4,RI5 \
  --feasible-only \
  --sample-feasible-directly \
  --output results/evaluation_datasets/eval_200n_d2_all_ri_feasible.json
'''

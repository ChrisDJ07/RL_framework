from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx

from utils.subgraph_selection.graph_io import (
    DEFAULT_FLOOD_TRANSFORM,
    DEFAULT_LANDSLIDE_TRANSFORM,
    DEFAULT_RI_MULTIPLIERS,
    RiskModel,
    graphml_safe_graph,
    load_simple_graph,
)
from utils.subgraph_selection.selection import (
    CandidateSubgraph,
    aggregate_corridor_weights,
    candidate_to_json,
    graph_positions,
    grow_tradeoff_subgraph,
    score_candidate_subgraph,
    select_seed_nodes,
)
from utils.subgraph_selection.tradeoff import sample_tradeoff_pairs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a connected tradeoff-rich subgraph based on fastest-vs-safest path divergence.",
    )
    parser.add_argument("--graphml", default="data/la_trinidad_hazard_graph.graphml", help="Source graph GraphML.")
    parser.add_argument("--num-nodes", type=int, required=True, help="Target node budget for the selected subgraph.")
    parser.add_argument("--num-pairs", type=int, default=5000, help="Number of OD pairs to sample on the full graph.")
    parser.add_argument("--rain-keys", default="RI2,RI3,RI4", help="Comma-separated RI keys used during tradeoff scoring.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--candidate-seeds", type=int, default=30, help="How many top seed regions to try.")
    parser.add_argument("--min-time-minutes", type=float, default=5.0, help="Ignore OD pairs with shorter min-time paths.")
    parser.add_argument("--max-time-minutes", type=float, default=0.0, help="Optional max min-time path length; 0 disables.")
    parser.add_argument("--min-delta-time-rel", type=float, default=0.05, help="Minimum relative time penalty for the safest path.")
    parser.add_argument("--min-delta-hazard-rel", type=float, default=0.05, help="Minimum relative hazard savings of the safest path.")
    parser.add_argument("--min-divergence", type=float, default=0.10, help="Minimum path divergence between fastest and safest paths.")
    parser.add_argument("--exclusive-edge-bonus", type=float, default=1.5, help="Extra weight for edges used by only one of the two paths.")
    parser.add_argument("--distance-penalty", type=float, default=0.15, help="Penalty used to keep grown candidates cohesive.")
    parser.add_argument("--core-fraction", type=float, default=0.25, help="Approximate fraction of budget devoted to the core corridor set.")
    parser.add_argument("--interesting-threshold", type=float, default=0.01, help="Minimum covered pair score counted as interesting inside a candidate.")
    parser.add_argument("--output-dir", default="subgraph_selection/output", help="Directory for the exported graph and diagnostics.")
    parser.add_argument("--output-stem", default="", help="Optional custom output stem.")
    parser.add_argument("--dry-run", action="store_true", help="Print diagnostics without writing files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    rain_keys = [part.strip() for part in args.rain_keys.split(",") if part.strip()]
    risk_model = RiskModel(
        flood_transform=DEFAULT_FLOOD_TRANSFORM.copy(),
        landslide_transform=DEFAULT_LANDSLIDE_TRANSFORM.copy(),
        ri_multipliers=DEFAULT_RI_MULTIPLIERS.copy(),
    )

    graph = load_simple_graph(args.graphml)
    max_time_minutes = None if args.max_time_minutes <= 0 else args.max_time_minutes

    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Sampling up to {args.num_pairs} OD pairs across {rain_keys} ...")

    pair_results = sample_tradeoff_pairs(
        graph=graph,
        risk_model=risk_model,
        rain_keys=rain_keys,
        num_pairs=args.num_pairs,
        seed=args.seed,
        min_time_minutes=args.min_time_minutes,
        max_time_minutes=max_time_minutes,
        min_delta_time_rel=args.min_delta_time_rel,
        min_delta_hazard_rel=args.min_delta_hazard_rel,
        min_divergence=args.min_divergence,
    )
    if not pair_results:
        raise RuntimeError("No tradeoff-rich OD pairs were found with the current thresholds.")

    print(f"Scored {len(pair_results)} OD pairs. Top mean score: {pair_results[0].mean_score:.4f}")

    edge_weights, node_weights = aggregate_corridor_weights(
        pair_results=pair_results,
        exclusive_edge_bonus=args.exclusive_edge_bonus,
    )
    seed_nodes = select_seed_nodes(graph, node_weights, args.candidate_seeds)
    print(f"Evaluating {len(seed_nodes)} candidate seed regions ...")

    candidates: List[CandidateSubgraph] = []
    for seed in seed_nodes:
        candidate_graph = grow_tradeoff_subgraph(
            graph=graph,
            seed=seed,
            node_weights=node_weights,
            edge_weights=edge_weights,
            num_nodes=args.num_nodes,
            distance_penalty=args.distance_penalty,
            core_fraction=args.core_fraction,
        )
        if candidate_graph.number_of_nodes() < args.num_nodes or not nx.is_connected(candidate_graph):
            continue

        candidate = score_candidate_subgraph(
            subgraph=candidate_graph,
            pair_results=pair_results,
            edge_weights=edge_weights,
            interesting_threshold=args.interesting_threshold,
        )
        if candidate is None:
            continue

        candidate.seed = seed
        candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No valid candidate subgraph survived the scoring stage.")

    candidates.sort(key=lambda item: item.score_total, reverse=True)
    best = candidates[0]

    print("Top candidate summary:")
    for index, candidate in enumerate(candidates[:5], start=1):
        print(
            f"[{index}] seed={candidate.seed} score={candidate.score_total:.4f} "
            f"pair_mean={candidate.score_pair_mean:.4f} pair_frac={candidate.score_pair_fraction:.4f} "
            f"edge_capture={candidate.score_edge_capture:.4f} "
            f"nodes={candidate.graph.number_of_nodes()} edges={candidate.graph.number_of_edges()} "
            f"internal_pairs={candidate.internal_pair_count}"
        )

    if args.output_stem:
        stem = args.output_stem
    else:
        stem = f"tradeoff_subgraph_n{args.num_nodes}"
    output_dir = Path(args.output_dir)
    graphml_path = output_dir / f"{stem}.graphml"
    json_path = output_dir / f"{stem}_summary.json"
    plot_path = output_dir / f"{stem}.png"

    if args.dry_run:
        print("Dry run enabled. No files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    best_graph = best.graph.copy()
    best_graph.graph["selection_score_total"] = float(best.score_total)
    best_graph.graph["selection_seed_node"] = str(best.seed)
    nx.write_graphml(graphml_safe_graph(best_graph), graphml_path)

    summary = {
        "parameters": {
            "graphml": args.graphml,
            "num_nodes": args.num_nodes,
            "num_pairs": args.num_pairs,
            "rain_keys": rain_keys,
            "seed": args.seed,
            "candidate_seeds": args.candidate_seeds,
            "min_time_minutes": args.min_time_minutes,
            "max_time_minutes": max_time_minutes,
            "min_delta_time_rel": args.min_delta_time_rel,
            "min_delta_hazard_rel": args.min_delta_hazard_rel,
            "min_divergence": args.min_divergence,
            "exclusive_edge_bonus": args.exclusive_edge_bonus,
            "distance_penalty": args.distance_penalty,
            "core_fraction": args.core_fraction,
            "interesting_threshold": args.interesting_threshold,
        },
        "best_candidate": candidate_to_json(best),
        "top_candidates": [candidate_to_json(candidate) for candidate in candidates[:10]],
        "top_pairs": [pair.to_json() for pair in pair_results[:50]],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_subgraph(best.graph, plot_path, best)

    print(f"Saved GraphML: {graphml_path}")
    print(f"Saved summary: {json_path}")
    print(f"Saved plot: {plot_path}")


def _plot_subgraph(graph: nx.Graph, plot_path: Path, candidate: CandidateSubgraph) -> None:
    positions = graph_positions(graph)
    if len(positions) != graph.number_of_nodes():
        positions = nx.spring_layout(graph, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#4b5563", width=1.2, alpha=0.85)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=28,
        node_color="#f8fafc",
        edgecolors="#0f172a",
        linewidths=0.4,
    )
    ax.set_title(
        f"Tradeoff Subgraph | nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
        f"score={candidate.score_total:.3f}"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx

try:
    from utils.subgraph_selection.graph_io import (
        DEFAULT_FLOOD_TRANSFORM,
        DEFAULT_LANDSLIDE_TRANSFORM,
        DEFAULT_RI_MULTIPLIERS,
        RiskModel,
        load_simple_graph,
    )
    from utils.subgraph_selection.tradeoff import compute_pair_tradeoff
except ImportError:
    from utils.subgraph_selection.graph_io import (
        DEFAULT_FLOOD_TRANSFORM,
        DEFAULT_LANDSLIDE_TRANSFORM,
        DEFAULT_RI_MULTIPLIERS,
        RiskModel,
        load_simple_graph,
    )
    from utils.subgraph_selection.tradeoff import compute_pair_tradeoff


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare candidate subgraphs by evaluating OD tradeoff scores inside each graph.",
    )
    parser.add_argument(
        "--graph",
        dest="graphs",
        action="append",
        required=True,
        help="GraphML path for a candidate subgraph. Repeat this flag for multiple graphs.",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=[],
        help="Optional label for the preceding graph. If omitted, folder/file names are used.",
    )
    parser.add_argument(
        "--rain-keys",
        default="RI1",
        help="Comma-separated RI keys used for comparison, e.g. RI1 or RI2,RI3.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampled-pair mode.")
    parser.add_argument(
        "--mode",
        choices=("all", "sample"),
        default="all",
        help="Use all unordered OD pairs inside each graph or sample a fixed number.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=10000,
        help="Number of unordered OD pairs to sample per graph when mode=sample.",
    )
    parser.add_argument("--min-time-minutes", type=float, default=5.0)
    parser.add_argument("--max-time-minutes", type=float, default=0.0)
    parser.add_argument("--min-delta-time-rel", type=float, default=0.05)
    parser.add_argument("--min-delta-hazard-rel", type=float, default=0.05)
    parser.add_argument("--min-divergence", type=float, default=0.10)
    parser.add_argument(
        "--output-dir",
        default="subgraph_selection/comparisons",
        help="Directory to write comparison JSON/CSV outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default="candidate_comparison",
        help="Filename stem for output files.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    graph_paths = [Path(path) for path in args.graphs]
    labels = normalize_labels(graph_paths, args.labels)
    rain_keys = [part.strip() for part in args.rain_keys.split(",") if part.strip()]
    max_time_minutes = None if args.max_time_minutes <= 0 else args.max_time_minutes

    risk_model = RiskModel(
        flood_transform=DEFAULT_FLOOD_TRANSFORM.copy(),
        landslide_transform=DEFAULT_LANDSLIDE_TRANSFORM.copy(),
        ri_multipliers=DEFAULT_RI_MULTIPLIERS.copy(),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_summaries: List[Dict[str, object]] = []
    per_rain_rows: List[Dict[str, object]] = []

    for label, graph_path in zip(labels, graph_paths):
        graph = load_simple_graph(graph_path)
        pair_list = choose_pairs(
            graph=graph,
            mode=args.mode,
            num_pairs=args.num_pairs,
            seed=args.seed,
        )

        summary, per_rain = evaluate_graph_pairs(
            label=label,
            graph=graph,
            pairs=pair_list,
            rain_keys=rain_keys,
            risk_model=risk_model,
            min_time_minutes=args.min_time_minutes,
            max_time_minutes=max_time_minutes,
            min_delta_time_rel=args.min_delta_time_rel,
            min_delta_hazard_rel=args.min_delta_hazard_rel,
            min_divergence=args.min_divergence,
        )
        graph_summaries.append(summary)
        per_rain_rows.extend(per_rain)

        print(
            f"{label}: nodes={summary['num_nodes']} edges={summary['num_edges']} "
            f"valid_pairs={summary['valid_pair_count']} mean_score={summary['mean_score']:.6f} "
            f"interesting_fraction={summary['interesting_fraction']:.4f}"
        )

    json_path = output_dir / f"{args.output_stem}.json"
    csv_path = output_dir / f"{args.output_stem}_overall.csv"
    per_rain_csv_path = output_dir / f"{args.output_stem}_per_ri.csv"

    payload = {
        "parameters": {
            "graphs": [str(path) for path in graph_paths],
            "labels": labels,
            "rain_keys": rain_keys,
            "mode": args.mode,
            "num_pairs": args.num_pairs,
            "seed": args.seed,
            "min_time_minutes": args.min_time_minutes,
            "max_time_minutes": max_time_minutes,
            "min_delta_time_rel": args.min_delta_time_rel,
            "min_delta_hazard_rel": args.min_delta_hazard_rel,
            "min_divergence": args.min_divergence,
        },
        "overall": graph_summaries,
        "per_rain": per_rain_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, graph_summaries)
    write_csv(per_rain_csv_path, per_rain_rows)

    print(f"Saved JSON: {json_path}")
    print(f"Saved overall CSV: {csv_path}")
    print(f"Saved per-RI CSV: {per_rain_csv_path}")


def normalize_labels(graph_paths: Sequence[Path], raw_labels: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for index, graph_path in enumerate(graph_paths):
        if index < len(raw_labels) and raw_labels[index]:
            labels.append(raw_labels[index])
        else:
            labels.append(graph_path.parent.name or graph_path.stem)
    return labels


def choose_pairs(graph: nx.Graph, mode: str, num_pairs: int, seed: int) -> List[Tuple[str, str]]:
    nodes = list(graph.nodes())
    all_pairs = list(itertools.combinations(nodes, 2))
    if mode == "all" or len(all_pairs) <= num_pairs:
        return all_pairs
    rng = random.Random(seed)
    return rng.sample(all_pairs, num_pairs)


def evaluate_graph_pairs(
    label: str,
    graph: nx.Graph,
    pairs: Sequence[Tuple[str, str]],
    rain_keys: Sequence[str],
    risk_model: RiskModel,
    min_time_minutes: float,
    max_time_minutes: float | None,
    min_delta_time_rel: float,
    min_delta_hazard_rel: float,
    min_divergence: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    aggregate_scores: List[float] = []
    aggregate_dt: List[float] = []
    aggregate_dh: List[float] = []
    aggregate_div: List[float] = []
    valid_pair_count = 0

    per_rain_acc: Dict[str, Dict[str, List[float] | int]] = {
        rain_key: {
            "scores": [],
            "delta_time": [],
            "delta_hazard": [],
            "divergence": [],
            "valid_pairs": 0,
        }
        for rain_key in rain_keys
    }

    for source, target in pairs:
        pair_details = []
        skip_pair = False
        for rain_key in rain_keys:
            try:
                detail = compute_pair_tradeoff(
                    graph=graph,
                    source=source,
                    target=target,
                    risk_model=risk_model,
                    rain_key=rain_key,
                    min_delta_time_rel=min_delta_time_rel,
                    min_delta_hazard_rel=min_delta_hazard_rel,
                    min_divergence=min_divergence,
                )
            except nx.NetworkXNoPath:
                skip_pair = True
                break

            if detail is None:
                skip_pair = True
                break

            if detail.time_path_time < min_time_minutes:
                skip_pair = True
                break
            if max_time_minutes is not None and detail.time_path_time > max_time_minutes:
                skip_pair = True
                break

            pair_details.append(detail)

        if skip_pair or not pair_details:
            continue

        valid_pair_count += 1
        mean_score = statistics.mean(detail.score for detail in pair_details)
        mean_dt = statistics.mean(detail.delta_time_rel for detail in pair_details)
        mean_dh = statistics.mean(detail.delta_hazard_rel for detail in pair_details)
        mean_div = statistics.mean(detail.divergence for detail in pair_details)

        aggregate_scores.append(mean_score)
        aggregate_dt.append(mean_dt)
        aggregate_dh.append(mean_dh)
        aggregate_div.append(mean_div)

        for detail in pair_details:
            acc = per_rain_acc[detail.rain_key]
            acc["scores"].append(detail.score)
            acc["delta_time"].append(detail.delta_time_rel)
            acc["delta_hazard"].append(detail.delta_hazard_rel)
            acc["divergence"].append(detail.divergence)
            acc["valid_pairs"] += 1

    overall_summary = {
        "label": label,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "evaluated_pair_count": len(pairs),
        "valid_pair_count": valid_pair_count,
        "valid_pair_fraction": safe_ratio(valid_pair_count, len(pairs)),
        "mean_score": safe_mean(aggregate_scores),
        "median_score": safe_median(aggregate_scores),
        "interesting_fraction": safe_ratio(sum(score > 0 for score in aggregate_scores), len(aggregate_scores)),
        "mean_delta_time_rel": safe_mean(aggregate_dt),
        "mean_delta_hazard_rel": safe_mean(aggregate_dh),
        "mean_divergence": safe_mean(aggregate_div),
        "score_p90": safe_quantile(aggregate_scores, 0.9),
        "score_p95": safe_quantile(aggregate_scores, 0.95),
    }

    per_rain_rows: List[Dict[str, object]] = []
    for rain_key in rain_keys:
        acc = per_rain_acc[rain_key]
        scores = list(acc["scores"])
        row = {
            "label": label,
            "rain_key": rain_key,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "valid_pair_count": acc["valid_pairs"],
            "mean_score": safe_mean(scores),
            "median_score": safe_median(scores),
            "interesting_fraction": safe_ratio(sum(score > 0 for score in scores), len(scores)),
            "mean_delta_time_rel": safe_mean(acc["delta_time"]),
            "mean_delta_hazard_rel": safe_mean(acc["delta_hazard"]),
            "mean_divergence": safe_mean(acc["divergence"]),
        }
        per_rain_rows.append(row)

    return overall_summary, per_rain_rows


def safe_mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def safe_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return float(ordered[index])


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()


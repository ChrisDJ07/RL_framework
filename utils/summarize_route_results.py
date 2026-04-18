from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.eval_pipeline_common import read_json


def build_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate route output JSON files into summary metrics."
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        required=True,
        help="One or more route output JSON files.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix for summary CSV files.",
    )
    return parser


def load_rows(route_path: str) -> list[dict]:
    payload = read_json(route_path)
    if payload.get("route_output_type") != "routing_route_output":
        raise ValueError(f"Unsupported route output: {route_path}")

    method = payload.get("method", {})
    label = method.get("label", Path(route_path).stem)
    rows = []
    for ep in payload.get("episodes", []):
        rows.append(
            {
                "method": label,
                "episode_id": int(ep["episode_id"]),
                "rain_key": ep["rain_key"],
                "feasible_at_reset": int(bool(ep.get("feasible_at_reset", True))),
                "success": int(bool(ep.get("success", False))),
                "timeout": int(bool(ep.get("timeout", False))),
                "termination_reason": ep.get("termination_reason", "unknown"),
                "steps": int(ep.get("steps", 0)),
                "total_time": float(ep.get("total_time", 0.0)),
                "total_distance": float(ep.get("total_distance", 0.0)),
                "total_reward": float(ep.get("total_reward", 0.0)),
                "total_hazard_raw": float(ep.get("total_hazard_raw", 0.0)),
                "hazard_exposure": float(ep.get("hazard_exposure", 0.0)),
                "hazard_score_sum": float(ep.get("hazard_score_sum", 0.0)),
                "runtime_sec": float(ep.get("runtime_sec", 0.0)),
            }
        )
    return rows


def summarize(group: pd.DataFrame) -> dict:
    feasible = group[group["feasible_at_reset"] == 1]
    success_only = group[group["success"] == 1]
    feasible_success = feasible["success"].mean() * 100.0 if len(feasible) else None

    return {
        "episodes": int(len(group)),
        "success_rate_pct": float(group["success"].mean() * 100.0),
        "timeout_rate_pct": float(group["timeout"].mean() * 100.0),
        "infeasible_at_reset_rate_pct": float((1.0 - group["feasible_at_reset"].mean()) * 100.0),
        "feasible_success_rate_pct": None if feasible_success is None else float(feasible_success),
        "avg_time_success": float(success_only["total_time"].mean()) if len(success_only) else None,
        "avg_distance_success": float(success_only["total_distance"].mean()) if len(success_only) else None,
        "avg_steps_success": float(success_only["steps"].mean()) if len(success_only) else None,
        "avg_hazard_exposure_success": float(success_only["hazard_exposure"].mean()) if len(success_only) else None,
        "avg_hazard_score_success": float(success_only["hazard_score_sum"].mean()) if len(success_only) else None,
        "avg_hazard_score_all": float(group["hazard_score_sum"].mean()) if len(group) else None,
        "avg_runtime_sec": float(group["runtime_sec"].mean()) if len(group) else None,
    }


def main():
    args = build_parser().parse_args()
    rows = []
    for route_path in args.routes:
        rows.extend(load_rows(route_path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows loaded from route outputs.")

    overall_rows = []
    for method, group in df.groupby("method"):
        overall_rows.append({"method": method, **summarize(group)})
    overall_df = pd.DataFrame(overall_rows).sort_values("method").reset_index(drop=True)

    per_ri_rows = []
    for (method, rain_key), group in df.groupby(["method", "rain_key"]):
        per_ri_rows.append({"method": method, "rain_key": rain_key, **summarize(group)})
    per_ri_df = pd.DataFrame(per_ri_rows).sort_values(["method", "rain_key"]).reset_index(drop=True)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    overall_path = output_prefix.parent / f"{output_prefix.name}_overall.csv"
    per_ri_path = output_prefix.parent / f"{output_prefix.name}_per_ri.csv"
    raw_path = output_prefix.parent / f"{output_prefix.name}_raw.csv"

    overall_df.to_csv(overall_path, index=False)
    per_ri_df.to_csv(per_ri_path, index=False)
    df.to_csv(raw_path, index=False)

    print("Overall summary:")
    print(overall_df.to_string(index=False))
    print(f"\nSaved: {overall_path}")
    print(f"Saved: {per_ri_path}")
    print(f"Saved: {raw_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


RUN_CONFIG_RE = re.compile(r"^Run config \| (?P<body>.+)$")
GRAPH_STATS_RE = re.compile(
    r"^Graph stats \| Nodes: (?P<nodes>\d+), Edges: (?P<edges>\d+), "
    r"AvgBaseTime: (?P<avg_base_time>[-+]?\d*\.?\d+), AvgHazard: (?P<avg_hazard>[-+]?\d*\.?\d+), "
    r"StateDim: (?P<state_dim>\d+), ActionDim: (?P<action_dim>\d+), Tmax\(min\): (?P<tmax>[-+]?\d*\.?\d+), "
    r"Deliveries: (?P<deliveries>\d+), Device: (?P<device>.+)$"
)
EVAL_RE = re.compile(
    r"^\[Eval @ Episode (?P<episode>\d+)(?: \| [^\]]+)?\] "
    r"eps=0\.00 -> MeanReward: (?P<r0>[-+]?\d*\.?\d+), SuccessRate: (?P<s0>[-+]?\d*\.?\d+)% \| "
    r"eps=0\.05 -> MeanReward: (?P<r5>[-+]?\d*\.?\d+), SuccessRate: (?P<s5>[-+]?\d*\.?\d+)%$"
)
BEST_CKPT_RE = re.compile(
    r"^Saved best checkpoint: (?P<path>.+?) "
    r"\(episode (?P<episode>\d+), success=(?P<success>[-+]?\d*\.?\d+)%?, reward=(?P<reward>[-+]?\d*\.?\d+)\)$"
)


def parse_kv_body(body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in body.split(" | "):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def bool_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def rel_config_path(raw: str) -> str:
    raw = raw.replace("\\", "/")
    marker = "/configs/"
    idx = raw.lower().find(marker)
    if idx != -1:
        return raw[idx + 1 :]
    idx = raw.lower().find("configs/")
    if idx != -1:
        return raw[idx:]
    return raw


@dataclass
class RunSummary:
    run_log: str
    run_family: str
    config_path_logged: str
    config_path_repo: str
    config_exists: str
    graphml_path: str
    graph_num_nodes_cfg: str
    num_deliveries_cfg: str
    use_pretrained_model_cfg: str
    pretrained_model_path_cfg: str
    resume_training_cfg: str
    flood_time_weight_cfg: str
    landslide_time_weight_cfg: str
    hazard_lambda_cfg: str
    active_rain_keys_cfg: str
    graph_nodes_log: str
    graph_edges_log: str
    tmax_log: str
    device_log: str
    run_count: str
    first_eval_episode: str
    last_eval_episode: str
    best_eval_eps0_episode: str
    best_eval_eps0_success: str
    best_eval_eps0_reward: str
    best_eval_eps005_episode: str
    best_eval_eps005_success: str
    best_eval_eps005_reward: str
    final_eval_eps0_success: str
    final_eval_eps005_success: str
    final_eval_eps0_reward: str
    final_eval_eps005_reward: str
    best_ckpt_episode_logged: str
    best_ckpt_success_logged: str
    best_ckpt_reward_logged: str


def summarize_run(root: Path, run_path: Path) -> RunSummary:
    lines = run_path.read_text(encoding="utf-8", errors="replace").splitlines()

    run_configs: list[dict[str, str]] = []
    evals: list[dict[str, float]] = []
    graph_stats: dict[str, str] = {}
    best_ckpt: dict[str, str] = {}

    for line in lines:
        m = RUN_CONFIG_RE.match(line)
        if m:
            run_configs.append(parse_kv_body(m.group("body")))
            continue

        m = GRAPH_STATS_RE.match(line)
        if m:
            graph_stats = m.groupdict()
            continue

        m = EVAL_RE.match(line)
        if m:
            evals.append(
                {
                    "episode": float(m.group("episode")),
                    "r0": float(m.group("r0")),
                    "s0": float(m.group("s0")),
                    "r5": float(m.group("r5")),
                    "s5": float(m.group("s5")),
                }
            )
            continue

        m = BEST_CKPT_RE.match(line)
        if m:
            best_ckpt = m.groupdict()

    latest_run_cfg = run_configs[-1] if run_configs else {}
    logged_config = latest_run_cfg.get("config_path", "")
    repo_config = rel_config_path(logged_config)
    repo_config_path = root / repo_config if repo_config.startswith("configs/") else root / repo_config
    cfg_json = load_json_if_exists(repo_config_path)

    graph_cfg = (cfg_json or {}).get("graph", {})
    env_cfg = (cfg_json or {}).get("environment", {})
    hazard_cfg = (cfg_json or {}).get("hazard", {})
    reward_cfg = (cfg_json or {}).get("reward", {})
    training_cfg = (cfg_json or {}).get("training", {})

    def fmt_num(value: float | None) -> str:
        return "" if value is None else f"{value:.2f}"

    best_eps0 = max(evals, key=lambda e: (e["s0"], e["r0"]), default=None)
    best_eps5 = max(evals, key=lambda e: (e["s5"], e["r5"]), default=None)
    first_eval = evals[0] if evals else None
    last_eval = evals[-1] if evals else None

    run_family = str(run_path.relative_to(root / "results" / "runs").parent).replace("\\", "/")

    return RunSummary(
        run_log=str(run_path.relative_to(root)).replace("\\", "/"),
        run_family=run_family,
        config_path_logged=logged_config,
        config_path_repo=str(repo_config).replace("\\", "/"),
        config_exists=bool_str(cfg_json is not None),
        graphml_path=str(graph_cfg.get("prebuilt_graphml_path", "")),
        graph_num_nodes_cfg=str(graph_cfg.get("num_nodes", "")),
        num_deliveries_cfg=str(env_cfg.get("num_deliveries", latest_run_cfg.get("num_deliveries", ""))),
        use_pretrained_model_cfg=bool_str(training_cfg.get("use_pretrained_model", latest_run_cfg.get("use_pretrained_model", ""))),
        pretrained_model_path_cfg=str(training_cfg.get("pretrained_model_path", latest_run_cfg.get("pretrained_model_path", ""))),
        resume_training_cfg=bool_str(training_cfg.get("resume_training", latest_run_cfg.get("resume_training", ""))),
        flood_time_weight_cfg=str(hazard_cfg.get("flood_time_weight", "")),
        landslide_time_weight_cfg=str(hazard_cfg.get("landslide_time_weight", "")),
        hazard_lambda_cfg=str(reward_cfg.get("hazard_lambda", "")),
        active_rain_keys_cfg=",".join(hazard_cfg.get("active_rain_keys", [])) if isinstance(hazard_cfg.get("active_rain_keys"), list) else str(hazard_cfg.get("active_rain_keys", "")),
        graph_nodes_log=graph_stats.get("nodes", ""),
        graph_edges_log=graph_stats.get("edges", ""),
        tmax_log=graph_stats.get("tmax", ""),
        device_log=graph_stats.get("device", ""),
        run_count=str(len(run_configs)),
        first_eval_episode=str(int(first_eval["episode"])) if first_eval else "",
        last_eval_episode=str(int(last_eval["episode"])) if last_eval else "",
        best_eval_eps0_episode=str(int(best_eps0["episode"])) if best_eps0 else "",
        best_eval_eps0_success=fmt_num(best_eps0["s0"] if best_eps0 else None),
        best_eval_eps0_reward=fmt_num(best_eps0["r0"] if best_eps0 else None),
        best_eval_eps005_episode=str(int(best_eps5["episode"])) if best_eps5 else "",
        best_eval_eps005_success=fmt_num(best_eps5["s5"] if best_eps5 else None),
        best_eval_eps005_reward=fmt_num(best_eps5["r5"] if best_eps5 else None),
        final_eval_eps0_success=fmt_num(last_eval["s0"] if last_eval else None),
        final_eval_eps005_success=fmt_num(last_eval["s5"] if last_eval else None),
        final_eval_eps0_reward=fmt_num(last_eval["r0"] if last_eval else None),
        final_eval_eps005_reward=fmt_num(last_eval["r5"] if last_eval else None),
        best_ckpt_episode_logged=best_ckpt.get("episode", ""),
        best_ckpt_success_logged=best_ckpt.get("success", ""),
        best_ckpt_reward_logged=best_ckpt.get("reward", ""),
    )


def write_csv(path: Path, rows: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(RunSummary.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RL training configs and run logs.")
    parser.add_argument("--runs-dir", default="results/runs", help="Directory containing run logs.")
    parser.add_argument(
        "--output-csv",
        default="results/analysis/training_run_summary.csv",
        help="CSV path for summarized runs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    runs_dir = root / args.runs_dir
    rows = [summarize_run(root, path) for path in sorted(runs_dir.rglob("*.txt"))]
    write_csv(root / args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {root / args.output_csv}")


if __name__ == "__main__":
    main()

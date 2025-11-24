#!/usr/bin/env python3
"""Summarize move decision times from automated Quoridor match logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, Iterable, List, Tuple

DEFAULT_LOG_DIR = Path(__file__).with_name("match_logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean and variance of move decision times from match logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing JSON match logs (default: examples/match_logs)",
    )
    parser.add_argument(
        "--agent",
        choices=["minimax", "baseline", "random", "all"],
        default="minimax",
        help="Which agent to include when summarizing timings (default: minimax).",
    )
    return parser.parse_args()


def load_logs(log_dir: Path) -> List[Dict[str, object]]:
    logs: List[Dict[str, object]] = []
    for path in sorted(log_dir.glob("*.json")):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as log_file:
            logs.append(json.load(log_file))
    return logs


def derive_agent_key(move: Dict[str, object], log: Dict[str, object]) -> str:
    if move.get("agent_key"):
        return str(move["agent_key"])

    minimax_player = log.get("minimax_player")
    opponent_key = (log.get("opponent") or {}).get("key")
    if move.get("player") == minimax_player:
        return "minimax"
    return str(opponent_key or "unknown")


def derive_settings(agent_key: str, log: Dict[str, object]) -> Dict[str, object]:
    if agent_key == "minimax":
        params = log.get("search_parameters") or {}
        return {
            "depth": params.get("depth"),
            "max_wall_moves": params.get("max_wall_moves"),
        }

    opponent = log.get("opponent") or {}
    if opponent.get("key") == agent_key:
        return {"opponent": opponent.get("label") or agent_key}

    return {}


def collect_timings(
    logs: Iterable[Dict[str, object]], agent_filter: str
) -> Tuple[Dict[Tuple[str, Tuple[Tuple[str, object], ...]], List[float]], int]:
    grouped: Dict[Tuple[str, Tuple[Tuple[str, object], ...]], List[float]] = {}
    missing_timing = 0

    for log in logs:
        for move in log.get("moves", []):
            time_ms = move.get("decision_time_ms")
            if time_ms is None:
                missing_timing += 1
                continue

            agent_key = derive_agent_key(move, log)
            if agent_filter != "all" and agent_key != agent_filter:
                continue

            settings = derive_settings(agent_key, log)
            settings_key = tuple(sorted(settings.items()))
            grouped.setdefault((agent_key, settings_key), []).append(float(time_ms))

    return grouped, missing_timing


def format_settings(settings: Tuple[Tuple[str, object], ...]) -> str:
    if not settings:
        return "default"
    return ", ".join(f"{key}={value}" for key, value in settings)


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir

    if not log_dir.exists():
        print(f"No logs found. Directory does not exist: {log_dir}")
        return 1

    logs = load_logs(log_dir)
    if not logs:
        print(f"No JSON logs found in {log_dir}")
        return 1

    grouped, missing = collect_timings(logs, args.agent)
    if not grouped:
        message = "No timing data found for the requested agent."
        if missing:
            message += " Existing logs may predate timing capture; rerun matches to generate new data."
        print(message)
        return 1

    print(
        f"Move time summary for agent={args.agent} "
        f"(files scanned: {len(logs)}, log_dir: {log_dir})"
    )
    for (agent_key, settings), samples in sorted(grouped.items()):
        mean_ms = mean(samples)
        variance_ms2 = pvariance(samples)
        print(
            f"- {agent_key} [{format_settings(settings)}]: "
            f"n={len(samples)}, mean={mean_ms:.2f} ms, variance={variance_ms2:.2f} ms^2"
        )

    if missing:
        print(f"\nSkipped {missing} moves without timing data (older logs).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

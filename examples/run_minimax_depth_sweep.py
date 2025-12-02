#!/usr/bin/env python3
"""Sweep minimax search depths against other agents and summarize results."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pvariance
from time import perf_counter
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.baseline_agent import BaselineAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.game.config import load_game_config
from src.game.state import GameResult, State, create_initial_state

Action = Tuple[str, Tuple[int, int]]
AgentFactory = Callable[[int], object]

OPPONENTS = {
    "random": ("Random", RandomAgent),
    "baseline": ("Baseline", BaselineAgent),
}


@dataclass
class PairingStats:
    wins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    losses: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    draws: int = 0
    move_times_ms: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    move_counts: List[int] = field(default_factory=list)

    def record_outcome(self, winner_key: str | None, agent_keys: Dict[int, str]) -> None:
        if winner_key is None:
            self.draws += 1
            return
        loser_key = next(k for k in agent_keys.values() if k != winner_key)
        self.wins[winner_key] += 1
        self.losses[loser_key] += 1

    def extend_times(self, times: Dict[str, List[float]]) -> None:
        for key, samples in times.items():
            self.move_times_ms[key].extend(samples)


def measure_match(
    agent_factories: Dict[int, AgentFactory],
    agent_keys: Dict[int, str],
    initial_state: State,
) -> Tuple[GameResult, Dict[str, List[float]], int]:
    """Play one match and capture per-move decision times (ms) by agent key."""
    state = initial_state
    agents = {player: factory(player) for player, factory in agent_factories.items()}
    timing: Dict[str, List[float]] = defaultdict(list)

    while not state.game_over:
        current = state.current_player
        agent = agents[current]
        start = perf_counter()
        action_type, position = agent.select_action(state)
        elapsed_ms = (perf_counter() - start) * 1000.0
        timing[agent_keys[current]].append(elapsed_ms)
        state = state.make_move(action_type, position)

    result = state.check_termination()
    return result, timing, state.move_count


def summarize_times(samples: List[float]) -> Tuple[float | None, float | None, int]:
    if not samples:
        return None, None, 0
    return mean(samples), pvariance(samples), len(samples)


def run_pairing(
    games: int,
    config_name: str,
    depth: int,
    opponent_key: str,
    max_wall_moves: int | None,
) -> PairingStats:
    """Run N games for a minimax vs opponent pairing."""
    config = load_game_config(config_name)
    opponent_label, opponent_cls = OPPONENTS[opponent_key]
    stats = PairingStats()

    print(
        f"\n--- Depth {depth} vs {opponent_label} ({games} games, config={config_name}, "
        f"wall limit={max_wall_moves if max_wall_moves is not None else 'unbounded'}) ---"
    )

    def get_factories(game_idx: int):
        # Alternate starting player for fairness.
        minimax_first = game_idx % 2 == 0
        if minimax_first:
            return (
                {
                    1: lambda _: MinimaxAgent(1, depth=depth, max_wall_moves=max_wall_moves),
                    2: lambda _: opponent_cls(2),
                },
                {1: "minimax", 2: opponent_key},
            )
        return (
            {
                1: lambda _: opponent_cls(1),
                2: lambda _: MinimaxAgent(2, depth=depth, max_wall_moves=max_wall_moves),
            },
            {1: opponent_key, 2: "minimax"},
            )

    for game_idx in range(games):
        agent_factories, agent_keys = get_factories(game_idx)
        initial_state = create_initial_state(config)
        print(f"  Game {game_idx + 1}/{games}...", end="", flush=True)
        result, times, move_count = measure_match(agent_factories, agent_keys, initial_state)
        winner_key = agent_keys.get(result.winner) if result and result.winner else None
        stats.record_outcome(winner_key, agent_keys)
        stats.extend_times(times)
        stats.move_counts.append(move_count)
        if winner_key is None:
            outcome_label = "draw"
        else:
            outcome_label = "minimax wins" if winner_key == "minimax" else f"{opponent_label} wins"
        print(f" done ({outcome_label}, moves={move_count})")

    return stats


def format_time_cell(mean_val: float | None, variance: float | None, samples: int) -> str:
    if samples == 0 or mean_val is None or variance is None:
        return "no data"
    return f"{mean_val:.1f} ms (var {variance:.1f}, n={samples})"


def print_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No results to display.")
        return

    headers = [
        "Depth",
        "Opponent",
        "Games",
        "Minimax W",
        "Opp W",
        "Draws",
        "Minimax Win%",
        "Opp Win%",
        "Avg Moves",
        "Minimax Time",
        "Opponent Time",
    ]

    table_rows: List[List[str]] = []
    for row in rows:
        table_rows.append(
            [
                str(row["depth"]),
                row["opponent"],
                str(row["games"]),
                str(row["minimax_wins"]),
                str(row["opponent_wins"]),
                str(row["draws"]),
                f"{row['minimax_win_pct']:.1f}%",
                f"{row['opponent_win_pct']:.1f}%",
                f"{row['avg_moves']:.1f}" if row["avg_moves"] is not None else "n/a",
                row["minimax_time_fmt"],
                row["opponent_time_fmt"],
            ]
        )

    col_widths = [
        max(len(h), *(len(row[idx]) for row in table_rows)) for idx, h in enumerate(headers)
    ]

    def render_row(parts: List[str]) -> str:
        return " | ".join(part.ljust(col_widths[i]) for i, part in enumerate(parts))

    print("\nMinimax depth sweep summary:")
    print(render_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in table_rows:
        print(render_row(row))
    print()


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "depth",
        "opponent",
        "games",
        "minimax_wins",
        "opponent_wins",
        "draws",
        "minimax_win_pct",
        "opponent_win_pct",
        "avg_moves",
        "minimax_mean_ms",
        "minimax_variance_ms2",
        "minimax_samples",
        "opponent_mean_ms",
        "opponent_variance_ms2",
        "opponent_samples",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run N trials of minimax vs other agents across depths and summarize stats."
    )
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=10,
        help="Number of games per depth/opponent pairing (default: 10).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "small", "tiny"],
        help="Game configuration (default: standard).",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=1,
        help="Lowest minimax search depth to test (default: 1).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Highest minimax search depth to test (default: 5).",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=list(OPPONENTS.keys()),
        default=list(OPPONENTS.keys()),
        help="Opponent agent keys to include (default: random baseline).",
    )
    parser.add_argument(
        "--max-wall-moves",
        type=int,
        default=8,
        help="Wall placement branching limit for minimax (default: 8, 0 for no limit).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "match_logs" / "minimax_depth_sweep.csv"),
        help="Path to save the summary table as CSV (default: examples/match_logs/minimax_depth_sweep.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_depth < 1 or args.max_depth < args.min_depth:
        print("Invalid depth range. Ensure 1 <= min-depth <= max-depth.")
        return 1

    max_walls = None if args.max_wall_moves <= 0 else args.max_wall_moves
    depths = list(range(args.min_depth, args.max_depth + 1))
    opponents = list(dict.fromkeys(args.opponents))  # Remove duplicates while preserving order.

    summary_rows: List[Dict[str, object]] = []
    print(
        f"Starting minimax depth sweep: depths {depths}, opponents {opponents}, "
        f"{args.games} games per pairing, config={args.config}, "
        f"wall limit={max_walls if max_walls is not None else 'unbounded'}"
    )
    for depth in depths:
        for opponent_key in opponents:
            stats = run_pairing(
                games=args.games,
                config_name=args.config,
                depth=depth,
                opponent_key=opponent_key,
                max_wall_moves=max_walls,
            )
            minimax_mean, minimax_var, minimax_samples = summarize_times(
                stats.move_times_ms.get("minimax", [])
            )
            opponent_mean, opponent_var, opponent_samples = summarize_times(
                stats.move_times_ms.get(opponent_key, [])
            )
            total_games = args.games
            minimax_wins = stats.wins.get("minimax", 0)
            opponent_wins = stats.wins.get(opponent_key, 0)
            draws = stats.draws
            minimax_win_pct = (minimax_wins / total_games) * 100
            opponent_win_pct = (opponent_wins / total_games) * 100
            avg_moves = mean(stats.move_counts) if stats.move_counts else None
            opponent_label = OPPONENTS[opponent_key][0]

            summary_rows.append(
                {
                    "depth": depth,
                    "opponent": opponent_label,
                    "games": total_games,
                    "minimax_wins": minimax_wins,
                    "opponent_wins": opponent_wins,
                    "draws": draws,
                    "minimax_win_pct": minimax_win_pct,
                    "opponent_win_pct": opponent_win_pct,
                    "avg_moves": avg_moves,
                    "minimax_mean_ms": minimax_mean,
                    "minimax_variance_ms2": minimax_var,
                    "minimax_samples": minimax_samples,
                    "opponent_mean_ms": opponent_mean,
                    "opponent_variance_ms2": opponent_var,
                    "opponent_samples": opponent_samples,
                    "minimax_time_fmt": format_time_cell(minimax_mean, minimax_var, minimax_samples),
                    "opponent_time_fmt": format_time_cell(opponent_mean, opponent_var, opponent_samples),
                }
            )

    print_table(summary_rows)
    output_path = Path(args.output)
    write_csv(output_path, summary_rows)
    print(f"Summary table written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run repeated agent matchups and report win/loss rates and move-time stats."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pvariance
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.baseline_agent import BaselineAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.game.config import load_game_config
from src.game.state import GameResult, State, create_initial_state

Action = Tuple[str, Tuple[int, int]]
AgentFactory = Callable[[int], object]


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


def summarize_times(samples: List[float]) -> str:
    if not samples:
        return "no data"
    return f"mean={mean(samples):.2f} ms, variance={pvariance(samples):.2f} ms^2, n={len(samples)}"


def run_pairing(
    name: str,
    games: int,
    config_name: str,
    get_factories: Callable[[int], Tuple[Dict[int, AgentFactory], Dict[int, str]]],
) -> None:
    """Run N games for a pairing, alternating start order for fairness."""
    config = load_game_config(config_name)
    stats = PairingStats()

    print(f"\n=== {name} ({games} games, config={config_name}) ===")

    for game_idx in range(games):
        agent_factories, agent_keys = get_factories(game_idx)
        initial_state = create_initial_state(config)
        result, times, move_count = measure_match(agent_factories, agent_keys, initial_state)
        winner_key = agent_keys.get(result.winner) if result and result.winner else None
        stats.record_outcome(winner_key, agent_keys)
        stats.extend_times(times)
        stats.move_counts.append(move_count)

    total_games = games
    for agent_key in sorted({*stats.wins.keys(), *stats.losses.keys(), *stats.move_times_ms.keys()}):
        wins = stats.wins.get(agent_key, 0)
        losses = stats.losses.get(agent_key, 0)
        draws = stats.draws
        win_rate = (wins / total_games) * 100
        loss_rate = (losses / total_games) * 100
        print(
            f"- {agent_key}: wins={wins} ({win_rate:.1f}%), "
            f"losses={losses} ({loss_rate:.1f}%), draws={draws}, "
            f"move_time[{summarize_times(stats.move_times_ms.get(agent_key, []))}]"
        )

    if stats.move_counts:
        avg_moves = mean(stats.move_counts)
        print(f"Avg moves per game: {avg_moves:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated agent matchups and report summary statistics."
    )
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=10,
        help="Number of games per pairing (default: 10).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "small", "tiny"],
        help="Game configuration (default: standard).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Minimax search depth (plies) when applicable (default: 3).",
    )
    parser.add_argument(
        "--max-wall-moves",
        type=int,
        default=8,
        help="Wall placement branching limit for minimax (default: 8, 0 for unlimited).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_walls = None if args.max_wall_moves <= 0 else args.max_wall_moves

    def baseline_vs_random_factories(game_idx: int):
        # Alternate starting player for fairness.
        if game_idx % 2 == 0:
            return (
                {1: lambda _: BaselineAgent(1), 2: lambda _: RandomAgent(2)},
                {1: "baseline", 2: "random"},
            )
        return (
            {1: lambda _: RandomAgent(1), 2: lambda _: BaselineAgent(2)},
            {1: "random", 2: "baseline"},
        )

    def minimax_vs_baseline_factories(game_idx: int):
        if game_idx % 2 == 0:
            return (
                {
                    1: lambda _: MinimaxAgent(1, depth=args.depth, max_wall_moves=max_walls),
                    2: lambda _: BaselineAgent(2),
                },
                {1: "minimax", 2: "baseline"},
            )
        return (
            {
                1: lambda _: BaselineAgent(1),
                2: lambda _: MinimaxAgent(2, depth=args.depth, max_wall_moves=max_walls),
            },
            {1: "baseline", 2: "minimax"},
        )

    def minimax_vs_random_factories(game_idx: int):
        if game_idx % 2 == 0:
            return (
                {
                    1: lambda _: MinimaxAgent(1, depth=args.depth, max_wall_moves=max_walls),
                    2: lambda _: RandomAgent(2),
                },
                {1: "minimax", 2: "random"},
            )
        return (
            {
                1: lambda _: RandomAgent(1),
                2: lambda _: MinimaxAgent(2, depth=args.depth, max_wall_moves=max_walls),
            },
            {1: "random", 2: "minimax"},
        )

    run_pairing("Baseline vs Random", args.games, args.config, baseline_vs_random_factories)
    run_pairing("Minimax vs Baseline", args.games, args.config, minimax_vs_baseline_factories)
    run_pairing("Minimax vs Random", args.games, args.config, minimax_vs_random_factories)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

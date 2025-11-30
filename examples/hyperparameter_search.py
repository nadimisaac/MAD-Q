#!/usr/bin/env python3
"""Grid search for minimax agent hyperparameters using round-robin tournament evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Dict, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.minimax_agent import MinimaxAgent
from src.game.config import load_game_config
from src.game.state import GameResult, State, create_initial_state


@dataclass
class HyperparamConfig:
    """Represents a specific hyperparameter configuration."""
    path_weight: float
    progress_weight: float
    wall_weight: float
    depth: int = 2
    max_wall_moves: int = 8

    def __hash__(self):
        return hash((self.path_weight, self.progress_weight, self.wall_weight, self.depth, self.max_wall_moves))

    def __eq__(self, other):
        if not isinstance(other, HyperparamConfig):
            return False
        return (self.path_weight == other.path_weight and
                self.progress_weight == other.progress_weight and
                self.wall_weight == other.wall_weight and
                self.depth == other.depth and
                self.max_wall_moves == other.max_wall_moves)

    def to_dict(self):
        return {
            "path_weight": self.path_weight,
            "progress_weight": self.progress_weight,
            "wall_weight": self.wall_weight,
            "depth": self.depth,
            "max_wall_moves": self.max_wall_moves,
        }

    def to_string(self):
        return f"P{self.path_weight}_Pr{self.progress_weight}_W{self.wall_weight}"


@dataclass
class TournamentStats:
    """Track results for all configurations in the tournament."""
    wins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    losses: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    draws: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_games: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_move_time_ms: Dict[str, float] = field(default_factory=dict)

    def record_game(self, config1_name: str, config2_name: str, winner_name: str | None):
        """Record the outcome of a single game."""
        self.total_games[config1_name] += 1
        self.total_games[config2_name] += 1

        if winner_name is None:
            self.draws[config1_name] += 1
            self.draws[config2_name] += 1
        elif winner_name == config1_name:
            self.wins[config1_name] += 1
            self.losses[config2_name] += 1
        else:
            self.wins[config2_name] += 1
            self.losses[config1_name] += 1

    def get_win_rate(self, config_name: str) -> float:
        """Calculate win rate for a configuration."""
        total = self.total_games.get(config_name, 0)
        if total == 0:
            return 0.0
        wins = self.wins.get(config_name, 0)
        draws = self.draws.get(config_name, 0)
        # Draws count as 0.5 wins
        return (wins + 0.5 * draws) / total

    def get_rankings(self) -> List[Tuple[str, float, int, int, int]]:
        """Return configurations ranked by win rate."""
        rankings = []
        for config_name in self.total_games.keys():
            win_rate = self.get_win_rate(config_name)
            wins = self.wins.get(config_name, 0)
            losses = self.losses.get(config_name, 0)
            draws = self.draws.get(config_name, 0)
            rankings.append((config_name, win_rate, wins, losses, draws))

        # Sort by win rate (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


def generate_parameter_grid(
    path_weights: List[float],
    progress_weights: List[float],
    wall_weights: List[float],
    depth: int = 2,
    max_wall_moves: int = 8,
) -> List[HyperparamConfig]:
    """Generate all combinations of hyperparameters."""
    configs = []
    for path_w, progress_w, wall_w in product(path_weights, progress_weights, wall_weights):
        configs.append(HyperparamConfig(
            path_weight=path_w,
            progress_weight=progress_w,
            wall_weight=wall_w,
            depth=depth,
            max_wall_moves=max_wall_moves,
        ))
    return configs


def play_single_game(
    config1: HyperparamConfig,
    config2: HyperparamConfig,
    player1_num: int,
    player2_num: int,
    initial_state: State,
) -> Tuple[int | None, Dict[int, List[float]]]:
    """
    Play a single game between two configurations.
    Returns (winner_player_num, timing_data).
    """
    agent1 = MinimaxAgent(
        player_number=player1_num,
        depth=config1.depth,
        max_wall_moves=config1.max_wall_moves,
        path_weight=config1.path_weight,
        progress_weight=config1.progress_weight,
        wall_weight=config1.wall_weight,
    )
    agent2 = MinimaxAgent(
        player_number=player2_num,
        depth=config2.depth,
        max_wall_moves=config2.max_wall_moves,
        path_weight=config2.path_weight,
        progress_weight=config2.progress_weight,
        wall_weight=config2.wall_weight,
    )

    agents = {player1_num: agent1, player2_num: agent2}
    state = initial_state
    timing = defaultdict(list)

    while not state.game_over:
        current = state.current_player
        agent = agents[current]
        start = perf_counter()
        action_type, position = agent.select_action(state)
        elapsed_ms = (perf_counter() - start) * 1000.0
        timing[current].append(elapsed_ms)
        state = state.make_move(action_type, position)

    result = state.check_termination()
    winner = result.winner if result else None
    return winner, timing


def run_matchup(
    config1: HyperparamConfig,
    config2: HyperparamConfig,
    games_per_matchup: int,
    game_config: str,
) -> Tuple[int, int, int, Dict[str, float]]:
    """
    Run multiple games between two configurations, alternating starting positions.
    Returns (config1_wins, config2_wins, draws, avg_times).
    """
    config = load_game_config(game_config)
    config1_wins = 0
    config2_wins = 0
    draws = 0

    all_times = {config1.to_string(): [], config2.to_string(): []}

    for game_idx in range(games_per_matchup):
        initial_state = create_initial_state(config)

        # Alternate which configuration plays as player 1
        if game_idx % 2 == 0:
            winner, timing = play_single_game(config1, config2, 1, 2, initial_state)
            all_times[config1.to_string()].extend(timing.get(1, []))
            all_times[config2.to_string()].extend(timing.get(2, []))

            if winner == 1:
                config1_wins += 1
            elif winner == 2:
                config2_wins += 1
            else:
                draws += 1
        else:
            winner, timing = play_single_game(config2, config1, 1, 2, initial_state)
            all_times[config2.to_string()].extend(timing.get(1, []))
            all_times[config1.to_string()].extend(timing.get(2, []))

            if winner == 1:
                config2_wins += 1
            elif winner == 2:
                config1_wins += 1
            else:
                draws += 1

    # Calculate average times
    avg_times = {}
    for config_name, times in all_times.items():
        avg_times[config_name] = mean(times) if times else 0.0

    return config1_wins, config2_wins, draws, avg_times


def run_tournament(
    configs: List[HyperparamConfig],
    games_per_matchup: int,
    game_config: str,
    verbose: bool = True,
) -> TournamentStats:
    """
    Run round-robin tournament where each configuration plays every other configuration.
    """
    stats = TournamentStats()
    total_matchups = len(configs) * (len(configs) - 1) // 2
    matchup_count = 0

    print(f"\nStarting tournament with {len(configs)} configurations")
    print(f"Total matchups: {total_matchups}")
    print(f"Games per matchup: {games_per_matchup}")
    print(f"Total games: {total_matchups * games_per_matchup}\n")

    # Round-robin: each config plays every other config
    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs):
            if i >= j:
                continue  # Skip self-play and duplicate matchups

            matchup_count += 1
            config1_name = config1.to_string()
            config2_name = config2.to_string()

            if verbose:
                print(f"[{matchup_count}/{total_matchups}] {config1_name} vs {config2_name}...", end=" ", flush=True)

            c1_wins, c2_wins, draws, avg_times = run_matchup(
                config1, config2, games_per_matchup, game_config
            )

            # Record all games from this matchup
            for _ in range(c1_wins):
                stats.record_game(config1_name, config2_name, config1_name)
            for _ in range(c2_wins):
                stats.record_game(config1_name, config2_name, config2_name)
            for _ in range(draws):
                stats.record_game(config1_name, config2_name, None)

            # Store average move times
            for config_name, avg_time in avg_times.items():
                if config_name not in stats.avg_move_time_ms:
                    stats.avg_move_time_ms[config_name] = avg_time
                else:
                    # Running average
                    stats.avg_move_time_ms[config_name] = (
                        stats.avg_move_time_ms[config_name] + avg_time
                    ) / 2

            if verbose:
                print(f"{config1_name}={c1_wins}, {config2_name}={c2_wins}, draws={draws}")

    return stats


def save_results(
    stats: TournamentStats,
    configs: List[HyperparamConfig],
    output_path: Path,
):
    """Save tournament results to JSON file."""
    config_map = {config.to_string(): config for config in configs}

    results = {
        "rankings": [],
        "detailed_stats": {},
    }

    for config_name, win_rate, wins, losses, draws in stats.get_rankings():
        config = config_map[config_name]
        results["rankings"].append({
            "rank": len(results["rankings"]) + 1,
            "config_name": config_name,
            "parameters": config.to_dict(),
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": stats.total_games[config_name],
            "avg_move_time_ms": stats.avg_move_time_ms.get(config_name, 0.0),
        })

        results["detailed_stats"][config_name] = {
            "parameters": config.to_dict(),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": stats.total_games[config_name],
            "win_rate": win_rate,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def print_rankings(stats: TournamentStats, top_n: int = 10):
    """Print top N configurations by win rate."""
    print("\n" + "=" * 80)
    print("TOURNAMENT RESULTS")
    print("=" * 80)

    rankings = stats.get_rankings()

    print(f"\nTop {min(top_n, len(rankings))} Configurations:\n")
    print(f"{'Rank':<6} {'Configuration':<30} {'Win Rate':<10} {'W-L-D':<15} {'Avg Time (ms)':<15}")
    print("-" * 80)

    for rank, (config_name, win_rate, wins, losses, draws) in enumerate(rankings[:top_n], 1):
        wld = f"{wins}-{losses}-{draws}"
        avg_time = stats.avg_move_time_ms.get(config_name, 0.0)
        print(f"{rank:<6} {config_name:<30} {win_rate:>8.1%}  {wld:<15} {avg_time:>10.2f}")

    print("\n" + "=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for minimax hyperparameters using round-robin tournament."
    )
    parser.add_argument(
        "--games-per-matchup",
        "-n",
        type=int,
        default=10,
        help="Number of games per matchup (default: 10).",
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
        default=2,
        help="Minimax search depth for all configs (default: 2).",
    )
    parser.add_argument(
        "--max-wall-moves",
        type=int,
        default=8,
        help="Wall placement branching limit (default: 8).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/hyperparameter_search_results.json",
        help="Output JSON file path (default: results/hyperparameter_search_results.json).",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Use coarse grid (3x3x3 = 27 configs).",
    )
    parser.add_argument(
        "--fine",
        action="store_true",
        help="Use fine grid (5x5x5 = 125 configs).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations to display (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Define parameter ranges
    if args.fine:
        path_weights = [8.0, 10.0, 12.0, 15.0, 20.0]
        progress_weights = [0.5, 1.0, 1.5, 2.0, 3.0]
        wall_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        print("Using FINE grid (5x5x5 = 125 configurations)")
    elif args.coarse:
        path_weights = [8.0, 12.0, 16.0]
        progress_weights = [1.0, 2.0, 3.0]
        wall_weights = [1.0, 3.0, 5.0]
        print("Using COARSE grid (3x3x3 = 27 configurations)")
    else:
        # Default: coarse grid
        path_weights = [8.0, 12.0, 16.0]
        progress_weights = [1.0, 2.0, 3.0]
        wall_weights = [1.0, 3.0, 5.0]
        print("Using COARSE grid (3x3x3 = 27 configurations) - use --fine for finer grid")

    # Generate all configurations
    configs = generate_parameter_grid(
        path_weights=path_weights,
        progress_weights=progress_weights,
        wall_weights=wall_weights,
        depth=args.depth,
        max_wall_moves=args.max_wall_moves,
    )

    print(f"Game config: {args.config}")
    print(f"Games per matchup: {args.games_per_matchup}")
    print(f"Fixed parameters: depth={args.depth}, max_wall_moves={args.max_wall_moves}\n")

    # Run tournament
    start_time = perf_counter()
    stats = run_tournament(configs, args.games_per_matchup, args.config, verbose=True)
    elapsed_time = perf_counter() - start_time

    # Print results
    print_rankings(stats, top_n=args.top_n)

    # Save results
    output_path = Path(args.output)
    save_results(stats, configs, output_path)

    print(f"\nTotal tournament time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

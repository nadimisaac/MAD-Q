#!/usr/bin/env python3
"""Analyze minimax performance across different search depths."""

import sys
import csv
import argparse
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.config import load_game_config
from src.game.state import create_initial_state
from src.agents.baseline_agent import BaselineAgent
from src.agents.minimax_agent import MinimaxAgent


def run_depth_analysis(
    depths: list[int],
    games_per_depth: int,
    config_name: str,
    max_wall_moves: int,
    output_path: Path
) -> None:
    """
    Run games across different minimax depths vs baseline agent.

    Args:
        depths: List of depths to test
        games_per_depth: Number of games per depth
        config_name: Board configuration name
        max_wall_moves: Wall branching limit
        output_path: Path to save CSV results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = load_game_config(config_name)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'depth',
            'game_num',
            'winner',
            'num_moves',
            'minimax_player',
            'minimax_avg_time_ms',
            'baseline_avg_time_ms',
            'total_time_sec'
        ])

        for depth in depths:
            print(f"\n{'='*60}")
            print(f"Testing depth: {depth} (config: {config_name})")
            print(f"{'='*60}")

            wins = {1: 0, 2: 0, None: 0}

            for game_num in range(games_per_depth):
                # Alternate which player is minimax
                minimax_player = 1 if game_num % 2 == 0 else 2
                baseline_player = 3 - minimax_player

                max_walls_param = None if max_wall_moves <= 0 else max_wall_moves

                minimax_agent = MinimaxAgent(
                    player_number=minimax_player,
                    depth=depth,
                    max_wall_moves=max_walls_param
                )
                baseline_agent = BaselineAgent(baseline_player)

                agents = {
                    minimax_player: minimax_agent,
                    baseline_player: baseline_agent
                }

                state = create_initial_state(config)

                # Track timing per agent type
                minimax_times = []
                baseline_times = []
                game_start = perf_counter()

                while not state.game_over:
                    current_player = state.current_player
                    agent = agents[current_player]

                    move_start = perf_counter()
                    action_type, position = agent.select_action(state)
                    move_time = (perf_counter() - move_start) * 1000  # ms

                    if current_player == minimax_player:
                        minimax_times.append(move_time)
                    else:
                        baseline_times.append(move_time)

                    state = state.make_move(action_type, position)

                game_time = perf_counter() - game_start

                # Determine winner
                result = state.check_termination()
                winner = result.winner if result else None
                wins[winner] = wins.get(winner, 0) + 1

                # Compute average times
                minimax_avg = sum(minimax_times) / len(minimax_times) if minimax_times else 0
                baseline_avg = sum(baseline_times) / len(baseline_times) if baseline_times else 0

                # Write result
                writer.writerow([
                    depth,
                    game_num + 1,
                    winner if winner else 'draw',
                    state.move_count,
                    minimax_player,
                    minimax_avg,
                    baseline_avg,
                    game_time
                ])

                if (game_num + 1) % 10 == 0:
                    print(f"  Completed {game_num + 1}/{games_per_depth} games")

            # Print summary for this depth
            minimax_wins = sum(1 for w in [wins[1], wins[2]] if w > 0)
            total_decided = games_per_depth - wins.get(None, 0)
            win_rate = (sum([wins[1], wins[2]]) / total_decided * 100) if total_decided > 0 else 0

            print(f"\nDepth {depth} Summary:")
            print(f"  Minimax wins: {wins[1] + wins[2]}/{games_per_depth} ({win_rate:.1f}%)")
            print(f"  Draws: {wins.get(None, 0)}")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze minimax performance across search depths"
    )
    parser.add_argument(
        '--depths',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
        help='Depths to test (default: 1 2 3 4 5)'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=100,
        help='Number of games per depth (default: 100)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='small',
        choices=['tiny', 'small', 'standard'],
        help='Board configuration (default: small)'
    )
    parser.add_argument(
        '--max-wall-moves',
        type=int,
        default=8,
        help='Wall branching limit (default: 8, use 0 for unlimited)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/depth_analysis.csv',
        help='Output CSV file path'
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    print("\n" + "="*60)
    print("DEPTH PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Depths: {args.depths}")
    print(f"Games per depth: {args.games}")
    print(f"Max wall moves: {args.max_wall_moves if args.max_wall_moves > 0 else 'unlimited'}")
    print(f"Output: {output_path}")
    print("="*60)

    run_depth_analysis(
        depths=args.depths,
        games_per_depth=args.games,
        config_name=args.config,
        max_wall_moves=args.max_wall_moves,
        output_path=output_path
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Analyze game complexity across different board sizes."""

import sys
import csv
import argparse
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.config import load_game_config
from src.game.state import create_initial_state
from src.agents.baseline_agent import BaselineAgent


def run_board_size_analysis(
    board_configs: list[str],
    games_per_size: int,
    output_path: Path
) -> None:
    """
    Run games across different board sizes to analyze complexity.

    Args:
        board_configs: List of config names (e.g., ['tiny', 'small', 'standard'])
        games_per_size: Number of games to run per board size
        output_path: Path to save CSV results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'board_size',
            'game_num',
            'num_moves',
            'total_time_sec',
            'avg_move_time_ms',
            'player1_avg_time_ms',
            'player2_avg_time_ms'
        ])

        for config_name in board_configs:
            config = load_game_config(config_name)
            board_size = config.board_size

            print(f"\n{'='*60}")
            print(f"Testing board size: {board_size}x{board_size} ({config_name})")
            print(f"{'='*60}")

            for game_num in range(games_per_size):
                # Alternate starting player
                if game_num % 2 == 0:
                    agent1 = BaselineAgent(1)
                    agent2 = BaselineAgent(2)
                else:
                    agent1 = BaselineAgent(2)
                    agent2 = BaselineAgent(1)

                agents = {1: agent1, 2: agent2}
                state = create_initial_state(config)

                # Track timing per player
                player_times = {1: [], 2: []}
                game_start = perf_counter()

                while not state.game_over:
                    current_player = state.current_player
                    agent = agents[current_player]

                    move_start = perf_counter()
                    action_type, position = agent.select_action(state)
                    move_time = (perf_counter() - move_start) * 1000  # ms

                    player_times[current_player].append(move_time)

                    state = state.make_move(action_type, position)

                game_time = perf_counter() - game_start

                # Compute statistics
                num_moves = state.move_count
                avg_move_time = (game_time * 1000) / num_moves if num_moves > 0 else 0
                player1_avg = sum(player_times[1]) / len(player_times[1]) if player_times[1] else 0
                player2_avg = sum(player_times[2]) / len(player_times[2]) if player_times[2] else 0

                # Write result
                writer.writerow([
                    board_size,
                    game_num + 1,
                    num_moves,
                    game_time,
                    avg_move_time,
                    player1_avg,
                    player2_avg
                ])

                if (game_num + 1) % 10 == 0:
                    print(f"  Completed {game_num + 1}/{games_per_size} games")

            print(f"Completed {games_per_size} games for {board_size}x{board_size}")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze game complexity across board sizes"
    )
    parser.add_argument(
        '--games',
        type=int,
        default=100,
        help='Number of games per board size (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/board_size_analysis.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--configs',
        nargs='+',
        default=['tiny', 'small', 'standard'],
        help='Board configurations to test (default: tiny small standard)'
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    print("\n" + "="*60)
    print("BOARD SIZE COMPLEXITY ANALYSIS")
    print("="*60)
    print(f"Board sizes: {', '.join(args.configs)}")
    print(f"Games per size: {args.games}")
    print(f"Output: {output_path}")
    print("="*60)

    run_board_size_analysis(
        board_configs=args.configs,
        games_per_size=args.games,
        output_path=output_path
    )


if __name__ == '__main__':
    main()

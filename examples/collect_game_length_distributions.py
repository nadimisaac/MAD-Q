#!/usr/bin/env python3
"""Collect game length distributions across different matchup types."""

import sys
import csv
import json
import argparse
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.config import load_game_config
from src.game.state import create_initial_state
from src.agents.random_agent import RandomAgent
from src.agents.baseline_agent import BaselineAgent
from src.agents.minimax_agent import MinimaxAgent


def load_best_minimax_config(results_path: Path) -> dict:
    """Load the best minimax configuration from hyperparameter search results."""
    if not results_path.exists():
        print(f"Warning: {results_path} not found, using default minimax config")
        return {
            'path_weight': 12.0,
            'progress_weight': 1.5,
            'wall_weight': 2.0,
            'depth': 2,
            'max_wall_moves': 8
        }

    with open(results_path) as f:
        data = json.load(f)

    # Get top ranked configuration
    if data['rankings']:
        best = data['rankings'][0]
        return best['parameters']

    # Fallback to default
    return {
        'path_weight': 12.0,
        'progress_weight': 1.5,
        'wall_weight': 2.0,
        'depth': 2,
        'max_wall_moves': 8
    }


def run_matchup(
    matchup_type: str,
    games: int,
    config_name: str,
    minimax_config: dict,
    writer: csv.writer
) -> None:
    """
    Run games for a specific matchup type.

    Args:
        matchup_type: Type of matchup (e.g., 'Random-Random')
        games: Number of games to run
        config_name: Board configuration
        minimax_config: Dictionary with minimax parameters
        writer: CSV writer object
    """
    config = load_game_config(config_name)

    print(f"\nRunning {matchup_type} matchup ({games} games)...")

    for game_num in range(games):
        # Create agents based on matchup type
        if matchup_type == 'Random-Random':
            agent1 = RandomAgent(1)
            agent2 = RandomAgent(2)
        elif matchup_type == 'Baseline-Random':
            if game_num % 2 == 0:
                agent1 = BaselineAgent(1)
                agent2 = RandomAgent(2)
            else:
                agent1 = RandomAgent(1)
                agent2 = BaselineAgent(2)
        elif matchup_type == 'Baseline-Baseline':
            agent1 = BaselineAgent(1)
            agent2 = BaselineAgent(2)
        elif matchup_type == 'Minimax-Baseline':
            if game_num % 2 == 0:
                agent1 = MinimaxAgent(
                    player_number=1,
                    **minimax_config
                )
                agent2 = BaselineAgent(2)
            else:
                agent1 = BaselineAgent(1)
                agent2 = MinimaxAgent(
                    player_number=2,
                    **minimax_config
                )
        elif matchup_type == 'Minimax-Minimax':
            agent1 = MinimaxAgent(player_number=1, **minimax_config)
            agent2 = MinimaxAgent(player_number=2, **minimax_config)
        else:
            raise ValueError(f"Unknown matchup type: {matchup_type}")

        agents = {1: agent1, 2: agent2}
        state = create_initial_state(config)

        game_start = perf_counter()

        while not state.game_over:
            current_player = state.current_player
            agent = agents[current_player]
            action_type, position = agent.select_action(state)
            state = state.make_move(action_type, position)

        game_time = perf_counter() - game_start

        # Write result
        writer.writerow([
            matchup_type,
            game_num + 1,
            state.move_count,
            game_time
        ])

        if (game_num + 1) % 10 == 0:
            print(f"  Completed {game_num + 1}/{games} games")


def main():
    parser = argparse.ArgumentParser(
        description="Collect game length distributions across matchup types"
    )
    parser.add_argument(
        '--games',
        type=int,
        default=100,
        help='Number of games per matchup type (default: 100)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='small',
        choices=['tiny', 'small', 'standard'],
        help='Board configuration (default: small)'
    )
    parser.add_argument(
        '--minimax-results',
        type=str,
        default='results/hyperparameter_search_results.json',
        help='Path to hyperparameter search results JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/game_length_distributions.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--matchups',
        nargs='+',
        default=['Random-Random', 'Baseline-Random', 'Baseline-Baseline',
                 'Minimax-Baseline', 'Minimax-Minimax'],
        help='Matchup types to test'
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load best minimax configuration
    minimax_config = load_best_minimax_config(Path(args.minimax_results))

    print("\n" + "="*60)
    print("GAME LENGTH DISTRIBUTION COLLECTION")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Games per matchup: {args.games}")
    print(f"Matchups: {', '.join(args.matchups)}")
    print("\nBest Minimax Config:")
    for key, value in minimax_config.items():
        print(f"  {key}: {value}")
    print("="*60)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'matchup_type',
            'game_num',
            'num_moves',
            'total_time_sec'
        ])

        for matchup in args.matchups:
            run_matchup(
                matchup_type=matchup,
                games=args.games,
                config_name=args.config,
                minimax_config=minimax_config,
                writer=writer
            )

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

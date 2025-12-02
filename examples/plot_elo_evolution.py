#!/usr/bin/env python3
"""Plot Elo rating evolution from hyperparameter search tournament."""

import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.plotting_utils import (
    setup_publication_style,
    save_figure,
    create_colorblind_palette
)
from src.utils.elo_ratings import track_rating_evolution


def plot_elo_evolution(
    results_path: Path,
    output_dir: Path,
    formats: list[str],
    top_n: int = 10
) -> None:
    """
    Create Elo rating evolution plot from tournament results.

    Shows how Elo ratings change over the course of the tournament
    for the top N performing configurations.

    Args:
        results_path: Path to hyperparameter_search_results.json
        output_dir: Directory for output plots
        formats: List of output formats (e.g., ['png', 'pdf'])
        top_n: Number of top configurations to plot
    """
    setup_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    # Get game sequence from match history
    game_sequence = []
    for match in data.get('match_history', []):
        player1_name = match['player1_config']
        player2_name = match['player2_config']
        winner_name = match.get('winner_config')

        game_sequence.append((player1_name, player2_name, winner_name))

    if not game_sequence:
        print("Error: No match history found in results file.")
        return

    print(f"\nTracking Elo evolution over {len(game_sequence)} games...")

    # Track rating evolution
    evolution = track_rating_evolution(game_sequence, k=32, initial_rating=1500)

    # Get top N configurations by final rating
    final_ratings = {player: ratings[-1] for player, ratings in evolution.items()}
    top_configs = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_config_names = [name for name, _ in top_configs]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = create_colorblind_palette(len(top_config_names))

    # Plot evolution for top configs
    for i, config_name in enumerate(top_config_names):
        ratings = evolution[config_name]
        games = list(range(len(ratings)))

        ax.plot(
            games,
            ratings,
            linewidth=2,
            alpha=0.8,
            color=colors[i],
            label=f"{config_name} (final: {ratings[-1]:.0f})"
        )

    # Add horizontal line at initial rating
    ax.axhline(1500, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='Initial rating')

    # Styling
    ax.set_xlabel('Game Number', fontsize=12)
    ax.set_ylabel('Elo Rating', fontsize=12)
    ax.set_title(
        f'Elo Rating Evolution (Top {top_n} Configurations)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0.95
    )

    plt.tight_layout()

    save_figure(fig, 'elo_evolution', output_dir, formats)
    print(f"\nElo evolution plot saved to: {output_dir}")

    # Print final rankings
    print("\n" + "="*70)
    print(f"FINAL ELO RANKINGS (Top {top_n})")
    print("="*70)
    print(f"{'Rank':<6} {'Configuration':<35} {'Initial':<10} {'Final':<10} {'Change':<10}")
    print("-"*70)

    for rank, (config_name, final_elo) in enumerate(top_configs, 1):
        initial_elo = 1500.0
        change = final_elo - initial_elo

        print(
            f"{rank:<6} "
            f"{config_name:<35} "
            f"{initial_elo:<10.0f} "
            f"{final_elo:<10.0f} "
            f"{change:>+9.0f}"
        )

    print("="*70)

    # Statistical summary
    all_final = list(final_ratings.values())
    print("\nOVERALL STATISTICS:")
    print(f"  Mean final Elo: {np.mean(all_final):.1f}")
    print(f"  Std final Elo: {np.std(all_final):.1f}")
    print(f"  Range: [{np.min(all_final):.0f}, {np.max(all_final):.0f}]")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Elo rating evolution from tournament"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/hyperparameter_search_results.json',
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['png', 'pdf'],
        help='Output formats (default: png pdf)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top configurations to plot (default: 10)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run hyperparameter_search.py first to generate results.")
        return 1

    plot_elo_evolution(input_path, output_dir, args.formats, args.top_n)
    return 0


if __name__ == '__main__':
    sys.exit(main())

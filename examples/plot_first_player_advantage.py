#!/usr/bin/env python3
"""Analyze and plot first-player advantage from depth performance data."""

import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.plotting_utils import (
    setup_publication_style,
    save_figure,
    create_colorblind_palette
)
from src.utils.statistical_analysis import compute_win_rate_ci, compare_win_rates


def plot_first_player_advantage(
    depth_data_path: Path,
    output_dir: Path,
    formats: list[str]
) -> None:
    """
    Analyze and plot first-player advantage.

    Uses depth analysis data where minimax_player alternates between 1 and 2
    to determine if there's a first-player advantage.

    Args:
        depth_data_path: Path to depth_analysis.csv
        output_dir: Directory for output plots
        formats: List of output formats (e.g., ['png', 'pdf'])
    """
    setup_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(depth_data_path)

    # Get unique depths
    depths = sorted(df['depth'].unique())

    # Analyze first-player advantage
    stats = {
        'depth': [],
        'player1_win_rate': [],
        'p1_wr_lower': [],
        'p1_wr_upper': [],
        'player2_win_rate': [],
        'p2_wr_lower': [],
        'p2_wr_upper': []
    }

    for depth in depths:
        depth_df = df[df['depth'] == depth]

        # Count wins by player
        player1_wins = sum(depth_df['winner'] == 1)
        player2_wins = sum(depth_df['winner'] == 2)
        total_games = len(depth_df)

        # Compute win rates with confidence intervals
        p1_wr, p1_lower, p1_upper = compute_win_rate_ci(
            player1_wins, total_games, confidence=0.95
        )
        p2_wr, p2_lower, p2_upper = compute_win_rate_ci(
            player2_wins, total_games, confidence=0.95
        )

        stats['depth'].append(depth)
        stats['player1_win_rate'].append(p1_wr * 100)
        stats['p1_wr_lower'].append(p1_lower * 100)
        stats['p1_wr_upper'].append(p1_upper * 100)
        stats['player2_win_rate'].append(p2_wr * 100)
        stats['p2_wr_lower'].append(p2_lower * 100)
        stats['p2_wr_upper'].append(p2_upper * 100)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = create_colorblind_palette(2)
    color_p1 = colors[0]
    color_p2 = colors[1]

    x = np.array(stats['depth'])
    width = 0.35

    # Plot player 1 win rate
    ax.bar(
        x - width/2,
        stats['player1_win_rate'],
        width,
        yerr=[
            [wr - l for wr, l in zip(stats['player1_win_rate'], stats['p1_wr_lower'])],
            [u - wr for wr, u in zip(stats['player1_win_rate'], stats['p1_wr_upper'])]
        ],
        label='Player 1 (First)',
        color=color_p1,
        capsize=5,
        alpha=0.8
    )

    # Plot player 2 win rate
    ax.bar(
        x + width/2,
        stats['player2_win_rate'],
        width,
        yerr=[
            [wr - l for wr, l in zip(stats['player2_win_rate'], stats['p2_wr_lower'])],
            [u - wr for wr, u in zip(stats['player2_win_rate'], stats['p2_wr_upper'])]
        ],
        label='Player 2 (Second)',
        color=color_p2,
        capsize=5,
        alpha=0.8
    )

    # Add 50% reference line
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='50% (no advantage)')

    # Styling
    ax.set_xlabel('Search Depth (plies)', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title(
        'First-Player Advantage Analysis\n(Minimax vs Baseline)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xticks(depths)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_figure(fig, 'first_player_advantage', output_dir, formats)
    print(f"\nFirst-player advantage plot saved to: {output_dir}")

    # Print statistics
    print("\n" + "="*90)
    print("FIRST-PLAYER ADVANTAGE STATISTICS")
    print("="*90)
    print(f"{'Depth':<8} {'P1 Win %':<12} {'P1 95% CI':<20} {'P2 Win %':<12} {'P2 95% CI':<20}")
    print("-"*90)

    for i, depth in enumerate(depths):
        p1_str = f"{stats['player1_win_rate'][i]:.1f}%"
        p1_ci = f"[{stats['p1_wr_lower'][i]:.1f}%, {stats['p1_wr_upper'][i]:.1f}%]"
        p2_str = f"{stats['player2_win_rate'][i]:.1f}%"
        p2_ci = f"[{stats['p2_wr_lower'][i]:.1f}%, {stats['p2_wr_upper'][i]:.1f}%]"

        print(f"{depth:<8} {p1_str:<12} {p1_ci:<20} {p2_str:<12} {p2_ci:<20}")

    print("="*90)

    # Statistical tests
    print("\nSTATISTICAL SIGNIFICANCE TESTS:")
    print("-"*90)

    for i, depth in enumerate(depths):
        depth_df = df[df['depth'] == depth]

        player1_wins = sum(depth_df['winner'] == 1)
        player2_wins = sum(depth_df['winner'] == 2)
        total_games = len(depth_df)

        p_value, interp = compare_win_rates(
            player1_wins, total_games,
            player2_wins, total_games
        )

        advantage = "First-player" if player1_wins > player2_wins else "Second-player"
        magnitude = abs(player1_wins - player2_wins) / total_games * 100

        print(
            f"Depth {depth}: {advantage} advantage of {magnitude:.1f}% "
            f"({player1_wins} vs {player2_wins} wins) - {interp}"
        )

    print("="*90)

    # Overall analysis
    print("\nOVERALL ANALYSIS:")
    print("-"*90)

    total_p1_wins = sum(df['winner'] == 1)
    total_p2_wins = sum(df['winner'] == 2)
    total_games = len(df)

    overall_p1_wr, overall_p1_lower, overall_p1_upper = compute_win_rate_ci(
        total_p1_wins, total_games, confidence=0.95
    )
    overall_p2_wr, overall_p2_lower, overall_p2_upper = compute_win_rate_ci(
        total_p2_wins, total_games, confidence=0.95
    )

    print(f"Player 1 (First):  {overall_p1_wr*100:.1f}% [{overall_p1_lower*100:.1f}%, {overall_p1_upper*100:.1f}%]")
    print(f"Player 2 (Second): {overall_p2_wr*100:.1f}% [{overall_p2_lower*100:.1f}%, {overall_p2_upper*100:.1f}%]")

    p_value, interp = compare_win_rates(
        total_p1_wins, total_games,
        total_p2_wins, total_games
    )

    print(f"\nOverall difference: {interp}")

    if overall_p1_wr > 0.55:
        print("Conclusion: Significant first-player advantage detected")
    elif overall_p2_wr > 0.55:
        print("Conclusion: Significant second-player advantage detected")
    else:
        print("Conclusion: No strong first/second-player advantage")

    print("="*90)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot first-player advantage"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/depth_analysis.csv',
        help='Input CSV file path (depth analysis)'
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

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run analyze_depth_performance.py first to generate data.")
        return 1

    plot_first_player_advantage(input_path, output_dir, args.formats)
    return 0


if __name__ == '__main__':
    sys.exit(main())

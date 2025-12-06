#!/usr/bin/env python3
"""Plot minimax depth performance analysis."""

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
from src.utils.statistical_analysis import compute_win_rate_ci


def plot_depth_performance(
    data_path: Path,
    output_dir: Path,
    formats: list[str]
) -> None:
    """
    Create depth performance visualization with overlaid timing.

    Creates a dual-axis plot showing:
    - Win rate vs depth (left y-axis)
    - Average move time vs depth (right y-axis)

    Args:
        data_path: Path to depth_analysis.csv
        output_dir: Directory for output plots
        formats: List of output formats (e.g., ['png', 'pdf'])
    """
    setup_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Compute statistics per depth
    depths = sorted(df['depth'].unique())

    stats = {
        'depth': [],
        'win_rate': [],
        'wr_ci_lower': [],
        'wr_ci_upper': [],
        'avg_minimax_time': [],
        'avg_baseline_time': []
    }

    for depth in depths:
        depth_data = df[df['depth'] == depth]

        # Win rate computation
        # Winner column contains player number (1 or 2) or 'draw'
        # Minimax wins when winner == minimax_player
        minimax_wins = sum(
            (depth_data['winner'] == depth_data['minimax_player']).astype(int)
        )
        total_games = len(depth_data)

        win_rate, wr_lower, wr_upper = compute_win_rate_ci(
            minimax_wins,
            total_games,
            confidence=0.95
        )

        # Average move times
        avg_minimax_time = depth_data['minimax_avg_time_ms'].mean()
        avg_baseline_time = depth_data['baseline_avg_time_ms'].mean()

        stats['depth'].append(depth)
        stats['win_rate'].append(win_rate * 100)  # Convert to percentage
        stats['wr_ci_lower'].append(wr_lower * 100)
        stats['wr_ci_upper'].append(wr_upper * 100)
        stats['avg_minimax_time'].append(avg_minimax_time)
        stats['avg_baseline_time'].append(avg_baseline_time)

    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = create_colorblind_palette(3)
    color_wr = colors[0]  # Blue for win rate
    color_minimax = colors[1]  # Orange for minimax time
    color_baseline = colors[2]  # Green for baseline time

    # Plot win rate on left axis
    ax1.errorbar(
        stats['depth'],
        stats['win_rate'],
        yerr=[
            [wr - l for wr, l in zip(stats['win_rate'], stats['wr_ci_lower'])],
            [u - wr for wr, u in zip(stats['win_rate'], stats['wr_ci_upper'])]
        ],
        marker='o',
        markersize=10,
        linewidth=2.5,
        capsize=5,
        capthick=2,
        color=color_wr,
        label='Win Rate (vs Baseline)',
        zorder=3
    )

    ax1.set_xlabel('Search Depth (plies)', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', fontsize=12, color=color_wr)
    ax1.tick_params(axis='y', labelcolor=color_wr)
    ax1.set_ylim([0, 105])
    ax1.set_xticks(depths)
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.axhline(50, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='50% baseline')

    # Create second y-axis for timing
    ax2 = ax1.twinx()

    # Plot minimax timing on right axis
    ax2.plot(
        stats['depth'],
        stats['avg_minimax_time'],
        marker='s',
        markersize=8,
        linewidth=2,
        linestyle='--',
        color=color_minimax,
        label='Minimax Move Time',
        zorder=2
    )

    # Plot baseline timing on right axis
    ax2.plot(
        stats['depth'],
        stats['avg_baseline_time'],
        marker='^',
        markersize=8,
        linewidth=2,
        linestyle=':',
        color=color_baseline,
        label='Baseline Move Time',
        zorder=2
    )

    ax2.set_ylabel('Average Move Time (ms)', fontsize=12)
    ax2.set_yscale('log')  # Log scale for timing

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper left',
        fontsize=10,
        framealpha=0.95
    )

    plt.title(
        'Minimax Performance vs Search Depth',
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    fig.tight_layout()

    save_figure(fig, 'depth_performance', output_dir, formats)
    print(f"\nDepth performance plot saved to: {output_dir}")

    # Print statistics table
    print("\n" + "="*80)
    print("DEPTH PERFORMANCE STATISTICS")
    print("="*80)
    print(f"{'Depth':<8} {'Win Rate':<15} {'95% CI':<25} {'Minimax Time':<15} {'Baseline Time':<15}")
    print("-"*80)

    for i, depth in enumerate(depths):
        wr_str = f"{stats['win_rate'][i]:.1f}%"
        ci_str = f"[{stats['wr_ci_lower'][i]:.1f}%, {stats['wr_ci_upper'][i]:.1f}%]"
        mm_time = f"{stats['avg_minimax_time'][i]:.2f} ms"
        bl_time = f"{stats['avg_baseline_time'][i]:.2f} ms"

        print(f"{depth:<8} {wr_str:<15} {ci_str:<25} {mm_time:<15} {bl_time:<15}")

    print("-"*80)

    # Compute speedup ratios
    print("\nSpeedup Ratios (Minimax Time / Baseline Time):")
    for i, depth in enumerate(depths):
        ratio = stats['avg_minimax_time'][i] / stats['avg_baseline_time'][i]
        print(f"  Depth {depth}: {ratio:.1f}x slower")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Plot minimax depth performance analysis"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/depth_analysis.csv',
        help='Input CSV file path'
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

    plot_depth_performance(input_path, output_dir, args.formats)
    return 0


if __name__ == '__main__':
    sys.exit(main())

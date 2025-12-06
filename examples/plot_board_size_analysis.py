#!/usr/bin/env python3
"""Plot board size complexity analysis results."""

import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.plotting_utils import (
    setup_publication_style,
    save_figure,
    create_colorblind_palette
)
from src.utils.statistical_analysis import compute_confidence_interval


def plot_board_size_complexity(
    data_path: Path,
    output_dir: Path,
    formats: list[str]
) -> None:
    """
    Create board size complexity visualization.

    Creates a figure justifying the 5x5 board choice by showing:
    - Game length vs board size
    - Computation time vs board size

    Args:
        data_path: Path to board_size_analysis.csv
        output_dir: Directory for output plots
        formats: List of output formats (e.g., ['png', 'pdf'])
    """
    setup_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Compute statistics per board size
    board_sizes = sorted(df['board_size'].unique())

    stats = {
        'board_size': [],
        'avg_moves': [],
        'moves_ci_lower': [],
        'moves_ci_upper': [],
        'avg_time': [],
        'time_ci_lower': [],
        'time_ci_upper': []
    }

    for size in board_sizes:
        size_data = df[df['board_size'] == size]

        # Game length statistics
        moves = size_data['num_moves'].tolist()
        mean_moves, lower_moves, upper_moves = compute_confidence_interval(moves)

        # Total time statistics
        times = size_data['total_time_sec'].tolist()
        mean_time, lower_time, upper_time = compute_confidence_interval(times)

        stats['board_size'].append(size)
        stats['avg_moves'].append(mean_moves)
        stats['moves_ci_lower'].append(lower_moves)
        stats['moves_ci_upper'].append(upper_moves)
        stats['avg_time'].append(mean_time * 1000)  # Convert to ms
        stats['time_ci_lower'].append(lower_time * 1000)
        stats['time_ci_upper'].append(upper_time * 1000)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = create_colorblind_palette(1)
    color = colors[0]

    # Plot 1: Game length
    ax1.errorbar(
        stats['board_size'],
        stats['avg_moves'],
        yerr=[
            [m - l for m, l in zip(stats['avg_moves'], stats['moves_ci_lower'])],
            [u - m for m, u in zip(stats['avg_moves'], stats['moves_ci_upper'])]
        ],
        marker='o',
        markersize=8,
        linewidth=2,
        capsize=5,
        capthick=2,
        color=color,
        label='Baseline vs Baseline'
    )

    ax1.set_xlabel('Board Size (NxN)', fontsize=12)
    ax1.set_ylabel('Average Game Length (moves)', fontsize=12)
    ax1.set_title('Game Length vs Board Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(board_sizes)
    ax1.set_xticklabels([f'{s}x{s}' for s in board_sizes])

    # Highlight 5x5 choice
    if 5 in board_sizes:
        idx = board_sizes.index(5)
        ax1.axvline(5, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax1.annotate(
            'Our choice\n(5x5)',
            xy=(5, stats['avg_moves'][idx]),
            xytext=(5 + 0.5, stats['avg_moves'][idx] * 1.15),
            fontsize=10,
            color='red',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )

    # Plot 2: Computation time
    ax2.errorbar(
        stats['board_size'],
        stats['avg_time'],
        yerr=[
            [m - l for m, l in zip(stats['avg_time'], stats['time_ci_lower'])],
            [u - m for m, u in zip(stats['avg_time'], stats['time_ci_upper'])]
        ],
        marker='s',
        markersize=8,
        linewidth=2,
        capsize=5,
        capthick=2,
        color=color,
        label='Baseline vs Baseline'
    )

    ax2.set_xlabel('Board Size (NxN)', fontsize=12)
    ax2.set_ylabel('Average Game Time (ms)', fontsize=12)
    ax2.set_title('Computation Time vs Board Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(board_sizes)
    ax2.set_xticklabels([f'{s}x{s}' for s in board_sizes])

    # Highlight 5x5 choice
    if 5 in board_sizes:
        idx = board_sizes.index(5)
        ax2.axvline(5, color='red', linestyle='--', alpha=0.3, linewidth=1.5)

    plt.tight_layout()

    save_figure(fig, 'board_size_complexity', output_dir, formats)
    print(f"\nBoard size complexity plot saved to: {output_dir}")

    # Print statistics table
    print("\n" + "="*60)
    print("BOARD SIZE COMPLEXITY STATISTICS")
    print("="*60)
    print(f"{'Size':<10} {'Avg Moves':<15} {'95% CI':<20} {'Avg Time (ms)':<15}")
    print("-"*60)

    for i, size in enumerate(board_sizes):
        moves_str = f"{stats['avg_moves'][i]:.1f}"
        ci_str = f"[{stats['moves_ci_lower'][i]:.1f}, {stats['moves_ci_upper'][i]:.1f}]"
        time_str = f"{stats['avg_time'][i]:.2f}"

        highlight = " *" if size == 5 else ""
        print(f"{size}x{size:<8} {moves_str:<15} {ci_str:<20} {time_str:<15}{highlight}")

    print("-"*60)
    print("* Our choice for main experiments")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot board size complexity analysis"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/board_size_analysis.csv',
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
        print("Run analyze_board_size_complexity.py first to generate data.")
        return 1

    plot_board_size_complexity(input_path, output_dir, args.formats)
    return 0


if __name__ == '__main__':
    sys.exit(main())

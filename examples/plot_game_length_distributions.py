#!/usr/bin/env python3
"""Plot game length distribution comparisons across matchup types."""

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
from src.utils.statistical_analysis import (
    compute_summary_stats,
    independent_t_test
)


def plot_game_length_distributions(
    data_path: Path,
    output_dir: Path,
    formats: list[str]
) -> None:
    """
    Create game length distribution comparison plot.

    Creates violin plots comparing game lengths across different matchup types.

    Args:
        data_path: Path to game_length_distributions.csv
        output_dir: Directory for output plots
        formats: List of output formats (e.g., ['png', 'pdf'])
    """
    setup_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Get unique matchup types
    matchup_types = df['matchup_type'].unique()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = create_colorblind_palette(len(matchup_types))

    # Plot 1: Violin plot of game lengths
    parts = ax1.violinplot(
        [df[df['matchup_type'] == mt]['num_moves'].values for mt in matchup_types],
        positions=range(len(matchup_types)),
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Set labels for violin plot
    ax1.set_xlabel('Matchup Type', fontsize=12)
    ax1.set_ylabel('Game Length (moves)', fontsize=12)
    ax1.set_title('Game Length Distributions', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(matchup_types)))
    ax1.set_xticklabels(matchup_types, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot with individual points
    bp = ax2.boxplot(
        [df[df['matchup_type'] == mt]['num_moves'].values for mt in matchup_types],
        positions=range(len(matchup_types)),
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
    )

    # Color the boxes
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points with jitter
    for i, mt in enumerate(matchup_types):
        data = df[df['matchup_type'] == mt]['num_moves'].values
        x = np.random.normal(i, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.3, s=20, color=colors[i])

    # Set labels for box plot
    ax2.set_xlabel('Matchup Type', fontsize=12)
    ax2.set_ylabel('Game Length (moves)', fontsize=12)
    ax2.set_title('Game Length Box Plots', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(matchup_types)))
    ax2.set_xticklabels(matchup_types, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_figure(fig, 'game_length_distributions', output_dir, formats)
    print(f"\nGame length distributions plot saved to: {output_dir}")

    # Print summary statistics
    print("\n" + "="*90)
    print("GAME LENGTH DISTRIBUTION STATISTICS")
    print("="*90)
    print(f"{'Matchup':<20} {'N':<6} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<6} {'Max':<6} {'IQR':<8}")
    print("-"*90)

    stats_list = []
    for mt in matchup_types:
        data = df[df['matchup_type'] == mt]['num_moves'].tolist()
        stats = compute_summary_stats(data)
        stats_list.append((mt, stats))

        print(
            f"{mt:<20} "
            f"{stats['n']:<6} "
            f"{stats['mean']:<8.1f} "
            f"{stats['median']:<8.1f} "
            f"{stats['std']:<8.1f} "
            f"{stats['min']:<6.0f} "
            f"{stats['max']:<6.0f} "
            f"{stats['iqr']:<8.1f}"
        )

    print("="*90)

    # Statistical comparisons
    if len(matchup_types) >= 2:
        print("\nSTATISTICAL COMPARISONS (Independent t-tests):")
        print("-"*90)

        # Compare each pair
        for i in range(len(matchup_types)):
            for j in range(i + 1, len(matchup_types)):
                mt1 = matchup_types[i]
                mt2 = matchup_types[j]

                data1 = df[df['matchup_type'] == mt1]['num_moves'].tolist()
                data2 = df[df['matchup_type'] == mt2]['num_moves'].tolist()

                t_stat, p_value = independent_t_test(data1, data2)

                significance = ""
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"

                mean_diff = np.mean(data1) - np.mean(data2)

                print(
                    f"{mt1:<20} vs {mt2:<20}: "
                    f"t={t_stat:>6.2f}, p={p_value:.4f} {significance:>3} "
                    f"(Δ={mean_diff:>+6.1f} moves)"
                )

        print("\n* p < 0.05, ** p < 0.01, *** p < 0.001")
        print("="*90)


def main():
    parser = argparse.ArgumentParser(
        description="Plot game length distribution comparisons"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/game_length_distributions.csv',
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
        print("Run collect_game_length_distributions.py first to generate data.")
        return 1

    plot_game_length_distributions(input_path, output_dir, args.formats)
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Plot move time distributions by agent type."""

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


def plot_move_time_distributions(
    depth_data_path: Path,
    output_dir: Path,
    formats: list[str]
) -> None:
    """
    Create move time distribution comparison plot.

    Uses depth analysis data to compare move times between agent types.

    Args:
        depth_data_path: Path to depth_analysis.csv
        output_dir: Directory for output plots
        formats: List of output formats (e.g., ['png', 'pdf'])
    """
    setup_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(depth_data_path)

    # Prepare data for plotting
    # We'll compare minimax at different depths vs baseline
    depths = sorted(df['depth'].unique())

    # Create structured data
    plot_data = []
    for depth in depths:
        depth_df = df[df['depth'] == depth]

        # Minimax times
        for time in depth_df['minimax_avg_time_ms']:
            plot_data.append({
                'agent': f'Minimax (d={depth})',
                'time_ms': time,
                'agent_type': 'Minimax',
                'depth': depth
            })

        # Baseline times
        for time in depth_df['baseline_avg_time_ms']:
            plot_data.append({
                'agent': 'Baseline',
                'time_ms': time,
                'agent_type': 'Baseline',
                'depth': depth
            })

    plot_df = pd.DataFrame(plot_data)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Get unique agent names for plotting
    agent_names = [f'Minimax (d={d})' for d in depths] + ['Baseline']
    colors = create_colorblind_palette(len(agent_names))

    # Plot 1: Violin plot
    minimax_data = [plot_df[plot_df['agent'] == f'Minimax (d={d})']['time_ms'].values for d in depths]
    baseline_data = [plot_df[plot_df['agent'] == 'Baseline']['time_ms'].values]

    all_data = minimax_data + baseline_data
    positions = list(range(len(all_data)))

    parts = ax1.violinplot(
        all_data,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax1.set_xlabel('Agent Type', fontsize=12)
    ax1.set_ylabel('Move Time (ms)', fontsize=12)
    ax1.set_title('Move Time Distributions', fontsize=13, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(agent_names, rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot comparison
    bp = ax2.boxplot(
        all_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
    )

    # Color the boxes
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_xlabel('Agent Type', fontsize=12)
    ax2.set_ylabel('Move Time (ms)', fontsize=12)
    ax2.set_title('Move Time Box Plots', fontsize=13, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(agent_names, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_figure(fig, 'move_time_distributions', output_dir, formats)
    print(f"\nMove time distributions plot saved to: {output_dir}")

    # Print summary statistics
    print("\n" + "="*90)
    print("MOVE TIME DISTRIBUTION STATISTICS")
    print("="*90)
    print(f"{'Agent':<20} {'N':<6} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*90)

    for agent_name in agent_names:
        data = plot_df[plot_df['agent'] == agent_name]['time_ms'].tolist()
        if not data:
            continue

        stats = compute_summary_stats(data)

        print(
            f"{agent_name:<20} "
            f"{stats['n']:<6} "
            f"{stats['mean']:<10.3f} "
            f"{stats['median']:<10.3f} "
            f"{stats['std']:<10.3f} "
            f"{stats['min']:<10.3f} "
            f"{stats['max']:<10.3f}"
        )

    print("="*90)

    # Statistical comparisons between consecutive depths
    print("\nSTATISTICAL COMPARISONS:")
    print("-"*90)

    # Compare baseline vs each minimax depth
    baseline_times = plot_df[plot_df['agent'] == 'Baseline']['time_ms'].tolist()

    for depth in depths:
        minimax_times = plot_df[plot_df['agent'] == f'Minimax (d={depth})']['time_ms'].tolist()

        t_stat, p_value = independent_t_test(minimax_times, baseline_times)

        significance = ""
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"

        speedup = np.mean(minimax_times) / np.mean(baseline_times)

        print(
            f"Minimax (d={depth}) vs Baseline: "
            f"t={t_stat:>7.2f}, p={p_value:.4f} {significance:>3} "
            f"({speedup:.1f}x slower)"
        )

    # Compare consecutive minimax depths
    if len(depths) > 1:
        print("\nConsecutive depth comparisons:")
        for i in range(len(depths) - 1):
            depth1 = depths[i]
            depth2 = depths[i + 1]

            times1 = plot_df[plot_df['agent'] == f'Minimax (d={depth1})']['time_ms'].tolist()
            times2 = plot_df[plot_df['agent'] == f'Minimax (d={depth2})']['time_ms'].tolist()

            t_stat, p_value = independent_t_test(times2, times1)

            significance = ""
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"

            slowdown = np.mean(times2) / np.mean(times1)

            print(
                f"Depth {depth2} vs Depth {depth1}: "
                f"t={t_stat:>7.2f}, p={p_value:.4f} {significance:>3} "
                f"({slowdown:.2f}x slower)"
            )

    print("\n* p < 0.05, ** p < 0.01, *** p < 0.001")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(
        description="Plot move time distribution comparisons"
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

    plot_move_time_distributions(input_path, output_dir, args.formats)
    return 0


if __name__ == '__main__':
    sys.exit(main())

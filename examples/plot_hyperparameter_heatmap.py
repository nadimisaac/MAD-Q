#!/usr/bin/env python3
"""Plot hyperparameter search results as heatmaps."""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.plotting_utils import setup_publication_style, save_figure


def plot_hyperparameter_heatmaps(
    results_path: Path,
    output_dir: Path,
    formats: list[str],
    wall_weights: list[float] = None
) -> None:
    """
    Create heatmap visualizations for hyperparameter search results.

    Creates separate heatmaps for each wall_weight value, showing
    Elo rating as a function of path_weight (x) and progress_weight (y).

    Args:
        results_path: Path to hyperparameter_search_results.json
        output_dir: Directory for output plots
        formats: List of output formats (e.g., ['png', 'pdf'])
        wall_weights: Specific wall weights to plot (None = all)
    """
    setup_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    # Extract all results
    all_results = data['all_results']

    # Determine which wall weights to plot
    if wall_weights is None:
        wall_weights = sorted(set(r['parameters']['wall_weight'] for r in all_results))

    # Get unique path and progress weights
    path_weights = sorted(set(r['parameters']['path_weight'] for r in all_results))
    progress_weights = sorted(set(r['parameters']['progress_weight'] for r in all_results))

    print(f"\nCreating heatmaps for wall_weights: {wall_weights}")
    print(f"Path weights: {path_weights}")
    print(f"Progress weights: {progress_weights}")

    # Create a heatmap for each wall weight
    for wall_weight in wall_weights:
        # Filter results for this wall weight
        wall_results = [
            r for r in all_results
            if r['parameters']['wall_weight'] == wall_weight
        ]

        if not wall_results:
            print(f"Warning: No results found for wall_weight={wall_weight}")
            continue

        # Create matrix for heatmap
        # Rows = progress_weight, Cols = path_weight
        heatmap_data = np.full((len(progress_weights), len(path_weights)), np.nan)

        for result in wall_results:
            path_w = result['parameters']['path_weight']
            prog_w = result['parameters']['progress_weight']
            elo = result['final_elo']

            # Find indices
            try:
                path_idx = path_weights.index(path_w)
                prog_idx = progress_weights.index(prog_w)
                heatmap_data[prog_idx, path_idx] = elo
            except ValueError:
                continue

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            center=1500,  # Center colormap at initial Elo
            vmin=np.nanmin(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 1400,
            vmax=np.nanmax(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 1600,
            cbar_kws={'label': 'Final Elo Rating'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            square=False
        )

        # Set labels
        ax.set_xlabel('Path Weight', fontsize=12)
        ax.set_ylabel('Progress Weight', fontsize=12)
        ax.set_title(
            f'Hyperparameter Performance (Wall Weight = {wall_weight})',
            fontsize=14,
            fontweight='bold',
            pad=15
        )

        # Set tick labels
        ax.set_xticks(np.arange(len(path_weights)) + 0.5)
        ax.set_yticks(np.arange(len(progress_weights)) + 0.5)
        ax.set_xticklabels(path_weights, rotation=0)
        ax.set_yticklabels(progress_weights, rotation=0)

        # Highlight best configuration
        if not np.all(np.isnan(heatmap_data)):
            best_idx = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
            best_prog_idx, best_path_idx = best_idx

            # Add rectangle around best cell
            ax.add_patch(
                plt.Rectangle(
                    (best_path_idx, best_prog_idx),
                    1, 1,
                    fill=False,
                    edgecolor='blue',
                    linewidth=3,
                    linestyle='--'
                )
            )

            # Add annotation
            best_elo = heatmap_data[best_prog_idx, best_path_idx]
            best_path = path_weights[best_path_idx]
            best_prog = progress_weights[best_prog_idx]

            ax.text(
                len(path_weights) * 0.98,
                len(progress_weights) * 0.02,
                f'Best: path={best_path}, prog={best_prog}\nElo={best_elo:.0f}',
                ha='right',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                transform=ax.transData
            )

        plt.tight_layout()

        # Save with wall_weight in filename
        filename = f'hyperparameter_heatmap_wall{wall_weight}'
        save_figure(fig, filename, output_dir, formats)
        print(f"Saved heatmap for wall_weight={wall_weight}")

    print(f"\nAll heatmaps saved to: {output_dir}")

    # Print top configurations
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    print(f"{'Rank':<6} {'Elo':<8} {'Path':<8} {'Prog':<8} {'Wall':<8} {'Wins':<8} {'Losses':<8} {'Draws':<8}")
    print("-"*80)

    for i, config in enumerate(data['rankings'][:10], 1):
        params = config['parameters']
        print(
            f"{i:<6} "
            f"{config['final_elo']:<8.0f} "
            f"{params['path_weight']:<8} "
            f"{params['progress_weight']:<8} "
            f"{params['wall_weight']:<8} "
            f"{config['wins']:<8} "
            f"{config['losses']:<8} "
            f"{config['draws']:<8}"
        )

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Plot hyperparameter search results as heatmaps"
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
        '--wall-weights',
        nargs='+',
        type=float,
        default=None,
        help='Specific wall weights to plot (default: all)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run hyperparameter_search.py first to generate results.")
        return 1

    plot_hyperparameter_heatmaps(
        input_path,
        output_dir,
        args.formats,
        args.wall_weights
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())

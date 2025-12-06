#!/usr/bin/env python3
"""Generate summary tables and statistics for all experimental results."""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.statistical_analysis import (
    compute_summary_stats,
    compute_win_rate_ci
)


def generate_markdown_table(df: pd.DataFrame, caption: str) -> str:
    """Generate a Markdown-formatted table."""
    lines = [f"\n### {caption}\n"]

    # Header
    headers = " | ".join(df.columns)
    lines.append(f"| {headers} |")

    # Separator
    separator = " | ".join([":---:" for _ in df.columns])
    lines.append(f"| {separator} |")

    # Rows
    for _, row in df.iterrows():
        values = " | ".join([str(v) for v in row])
        lines.append(f"| {values} |")

    return "\n".join(lines)


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate a LaTeX-formatted table."""
    n_cols = len(df.columns)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule"
    ]

    # Header
    headers = " & ".join(df.columns) + " \\\\"
    lines.append(headers)
    lines.append("\\midrule")

    # Rows
    for _, row in df.iterrows():
        values = " & ".join([str(v) for v in row]) + " \\\\"
        lines.append(values)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)


def summarize_hyperparameter_search(results_path: Path) -> tuple[str, str]:
    """Generate summary for hyperparameter search results."""
    with open(results_path) as f:
        data = json.load(f)

    # Get top 10 configurations
    top_configs = data['rankings'][:10]

    rows = []
    for config in top_configs:
        params = config['parameters']
        rows.append({
            'Rank': config['rank'],
            'Path Wt': params['path_weight'],
            'Prog Wt': params['progress_weight'],
            'Wall Wt': params['wall_weight'],
            'Elo': f"{config['final_elo']:.0f}",
            'Win %': f"{config['win_rate']*100:.1f}",
            'W-L-D': f"{config['wins']}-{config['losses']}-{config['draws']}"
        })

    df = pd.DataFrame(rows)

    md = generate_markdown_table(df, "Top 10 Minimax Configurations (Hyperparameter Search)")
    latex = generate_latex_table(df, "Top 10 Minimax Configurations", "tab:hyperparameter_top10")

    return md, latex


def summarize_board_size_complexity(data_path: Path) -> tuple[str, str]:
    """Generate summary for board size complexity analysis."""
    df = pd.read_csv(data_path)

    board_sizes = sorted(df['board_size'].unique())

    rows = []
    for size in board_sizes:
        size_data = df[df['board_size'] == size]

        moves_stats = compute_summary_stats(size_data['num_moves'].tolist())
        time_stats = compute_summary_stats((size_data['total_time_sec'] * 1000).tolist())

        rows.append({
            'Board Size': f"{size}×{size}",
            'Avg Moves': f"{moves_stats['mean']:.1f}",
            'Std Moves': f"{moves_stats['std']:.1f}",
            'Avg Time (ms)': f"{time_stats['mean']:.2f}",
            'Std Time (ms)': f"{time_stats['std']:.2f}",
            'N': moves_stats['n']
        })

    summary_df = pd.DataFrame(rows)

    md = generate_markdown_table(summary_df, "Board Size Complexity Analysis")
    latex = generate_latex_table(summary_df, "Game Complexity by Board Size", "tab:board_complexity")

    return md, latex


def summarize_depth_performance(data_path: Path) -> tuple[str, str]:
    """Generate summary for depth performance analysis."""
    df = pd.read_csv(data_path)

    depths = sorted(df['depth'].unique())

    rows = []
    for depth in depths:
        depth_data = df[df['depth'] == depth]

        # Win rate
        minimax_wins = sum(depth_data['winner'] == depth_data['minimax_player'])
        total_games = len(depth_data)
        win_rate, wr_lower, wr_upper = compute_win_rate_ci(minimax_wins, total_games)

        # Timing
        mm_time = depth_data['minimax_avg_time_ms'].mean()
        bl_time = depth_data['baseline_avg_time_ms'].mean()

        rows.append({
            'Depth': depth,
            'Win Rate': f"{win_rate*100:.1f}%",
            '95% CI': f"[{wr_lower*100:.1f}%, {wr_upper*100:.1f}%]",
            'MM Time (ms)': f"{mm_time:.2f}",
            'BL Time (ms)': f"{bl_time:.2f}",
            'Slowdown': f"{mm_time/bl_time:.1f}×"
        })

    summary_df = pd.DataFrame(rows)

    md = generate_markdown_table(summary_df, "Minimax Performance vs Search Depth")
    latex = generate_latex_table(summary_df, "Minimax Performance by Search Depth", "tab:depth_performance")

    return md, latex


def summarize_game_length_distributions(data_path: Path) -> tuple[str, str]:
    """Generate summary for game length distributions."""
    df = pd.read_csv(data_path)

    matchup_types = df['matchup_type'].unique()

    rows = []
    for matchup in matchup_types:
        matchup_data = df[df['matchup_type'] == matchup]

        stats = compute_summary_stats(matchup_data['num_moves'].tolist())

        rows.append({
            'Matchup': matchup,
            'Mean': f"{stats['mean']:.1f}",
            'Median': f"{stats['median']:.1f}",
            'Std': f"{stats['std']:.1f}",
            'Min': int(stats['min']),
            'Max': int(stats['max']),
            'N': stats['n']
        })

    summary_df = pd.DataFrame(rows)

    md = generate_markdown_table(summary_df, "Game Length by Matchup Type")
    latex = generate_latex_table(summary_df, "Game Length Statistics by Matchup Type", "tab:game_lengths")

    return md, latex


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary tables for experimental results"
    )
    parser.add_argument(
        '--hyperparameter-results',
        type=str,
        default='results/hyperparameter_search_results.json',
        help='Path to hyperparameter search results'
    )
    parser.add_argument(
        '--board-size-data',
        type=str,
        default='results/board_size_analysis.csv',
        help='Path to board size analysis data'
    )
    parser.add_argument(
        '--depth-data',
        type=str,
        default='results/depth_analysis.csv',
        help='Path to depth analysis data'
    )
    parser.add_argument(
        '--game-length-data',
        type=str,
        default='results/game_length_distributions.csv',
        help='Path to game length distributions data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for summary files'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'latex', 'both'],
        default='both',
        help='Output format (default: both)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_sections = []
    latex_sections = []

    print("\n" + "="*70)
    print("GENERATING RESULTS SUMMARY")
    print("="*70)

    # Hyperparameter search summary
    if Path(args.hyperparameter_results).exists():
        print("\n✓ Processing hyperparameter search results...")
        md, latex = summarize_hyperparameter_search(Path(args.hyperparameter_results))
        markdown_sections.append(md)
        latex_sections.append(latex)
    else:
        print(f"\n✗ Skipping hyperparameter results (not found: {args.hyperparameter_results})")

    # Board size analysis
    if Path(args.board_size_data).exists():
        print("✓ Processing board size complexity analysis...")
        md, latex = summarize_board_size_complexity(Path(args.board_size_data))
        markdown_sections.append(md)
        latex_sections.append(latex)
    else:
        print(f"✗ Skipping board size analysis (not found: {args.board_size_data})")

    # Depth performance
    if Path(args.depth_data).exists():
        print("✓ Processing depth performance analysis...")
        md, latex = summarize_depth_performance(Path(args.depth_data))
        markdown_sections.append(md)
        latex_sections.append(latex)
    else:
        print(f"✗ Skipping depth analysis (not found: {args.depth_data})")

    # Game length distributions
    if Path(args.game_length_data).exists():
        print("✓ Processing game length distributions...")
        md, latex = summarize_game_length_distributions(Path(args.game_length_data))
        markdown_sections.append(md)
        latex_sections.append(latex)
    else:
        print(f"✗ Skipping game length analysis (not found: {args.game_length_data})")

    # Write output files
    if args.format in ['markdown', 'both'] and markdown_sections:
        md_path = output_dir / 'results_summary.md'
        with open(md_path, 'w') as f:
            f.write("# Quoridor AI Experimental Results Summary\n")
            f.write("\n".join(markdown_sections))
        print(f"\n✓ Markdown summary saved to: {md_path}")

    if args.format in ['latex', 'both'] and latex_sections:
        latex_path = output_dir / 'results_summary.tex'
        with open(latex_path, 'w') as f:
            f.write("% Quoridor AI Experimental Results Summary\n")
            f.write("% Include this file in your LaTeX document\n\n")
            f.write("\n\n".join(latex_sections))
        print(f"✓ LaTeX summary saved to: {latex_path}")

    print("\n" + "="*70)
    print("SUMMARY GENERATION COMPLETE")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())

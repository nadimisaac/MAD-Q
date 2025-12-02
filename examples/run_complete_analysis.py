#!/usr/bin/env python3
"""Master script to run complete experimental analysis pipeline."""

import sys
import argparse
import subprocess
from pathlib import Path
from time import perf_counter


def run_command(cmd: list[str], description: str, timeout: int = 3600) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command and arguments as list
        description: Human-readable description
        timeout: Timeout in seconds (default: 1 hour)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start_time = perf_counter()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            timeout=timeout,
            capture_output=False,
            text=True
        )
        elapsed = perf_counter() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = perf_counter() - start_time
        print(f"\n✗ Failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False

    except subprocess.TimeoutExpired:
        print(f"\n✗ Timed out after {timeout} seconds")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete experimental analysis pipeline"
    )
    parser.add_argument(
        '--skip-hyperparameter-search',
        action='store_true',
        help='Skip hyperparameter search (use existing results)'
    )
    parser.add_argument(
        '--skip-data-collection',
        action='store_true',
        help='Skip data collection (use existing data)'
    )
    parser.add_argument(
        '--skip-plotting',
        action='store_true',
        help='Skip plot generation (useful for data collection only)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='small',
        choices=['tiny', 'small', 'standard'],
        help='Board configuration for analyses (default: small)'
    )
    parser.add_argument(
        '--hyperparameter-games',
        type=int,
        default=10,
        help='Games per matchup in hyperparameter search (default: 10)'
    )
    parser.add_argument(
        '--analysis-games',
        type=int,
        default=100,
        help='Games per analysis (default: 100)'
    )
    parser.add_argument(
        '--use-coarse-grid',
        action='store_true',
        help='Use coarse hyperparameter grid instead of fine grid'
    )
    parser.add_argument(
        '--plot-formats',
        nargs='+',
        default=['png', 'pdf'],
        help='Plot output formats (default: png pdf)'
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    python_exe = sys.executable

    print("\n" + "="*70)
    print("QUORIDOR AI - COMPLETE EXPERIMENTAL ANALYSIS PIPELINE")
    print("="*70)
    print(f"Board configuration: {args.config}")
    print(f"Analysis games: {args.analysis_games}")
    print(f"Hyperparameter search games: {args.hyperparameter_games}")
    print(f"Hyperparameter grid: {'coarse' if args.use_coarse_grid else 'fine'}")
    print(f"Plot formats: {', '.join(args.plot_formats)}")
    print("="*70)

    overall_start = perf_counter()
    failed_steps = []

    # Step 1: Hyperparameter search
    if not args.skip_hyperparameter_search:
        cmd = [
            python_exe,
            str(project_root / 'examples' / 'hyperparameter_search.py'),
            '--config', args.config,
            '--games-per-matchup', str(args.hyperparameter_games),
            '--output', 'results/hyperparameter_search_results.json'
        ]
        if not args.use_coarse_grid:
            cmd.append('--fine')
        else:
            cmd.append('--coarse')

        if not run_command(cmd, "Hyperparameter Search", timeout=7200):
            failed_steps.append("Hyperparameter Search")
    else:
        print("\n⊘ Skipping hyperparameter search (using existing results)")

    # Step 2: Data collection
    if not args.skip_data_collection:
        data_collection_steps = [
            {
                'script': 'analyze_board_size_complexity.py',
                'description': 'Board Size Complexity Analysis',
                'args': ['--games', str(args.analysis_games), '--configs', 'tiny', 'small', 'standard']
            },
            {
                'script': 'analyze_depth_performance.py',
                'description': 'Depth Performance Analysis',
                'args': ['--games', str(args.analysis_games), '--depths', '1', '2', '3', '4', '5',
                        '--config', args.config]
            },
            {
                'script': 'collect_game_length_distributions.py',
                'description': 'Game Length Distribution Collection',
                'args': ['--games', str(args.analysis_games), '--config', args.config]
            }
        ]

        for step in data_collection_steps:
            cmd = [
                python_exe,
                str(project_root / 'examples' / step['script'])
            ] + step['args']

            if not run_command(cmd, step['description'], timeout=3600):
                failed_steps.append(step['description'])
    else:
        print("\n⊘ Skipping data collection (using existing data)")

    # Step 3: Plot generation
    if not args.skip_plotting:
        plotting_steps = [
            {
                'script': 'plot_board_size_analysis.py',
                'description': 'Board Size Complexity Plot'
            },
            {
                'script': 'plot_depth_analysis.py',
                'description': 'Depth Performance Plot'
            },
            {
                'script': 'plot_hyperparameter_heatmap.py',
                'description': 'Hyperparameter Heatmap Plot'
            },
            {
                'script': 'plot_game_length_distributions.py',
                'description': 'Game Length Distributions Plot'
            },
            {
                'script': 'plot_elo_evolution.py',
                'description': 'Elo Evolution Plot'
            },
            {
                'script': 'plot_move_time_distributions.py',
                'description': 'Move Time Distributions Plot'
            },
            {
                'script': 'plot_first_player_advantage.py',
                'description': 'First-Player Advantage Plot'
            }
        ]

        for step in plotting_steps:
            cmd = [
                python_exe,
                str(project_root / 'examples' / step['script']),
                '--formats'
            ] + args.plot_formats

            if not run_command(cmd, step['description'], timeout=300):
                failed_steps.append(step['description'])
    else:
        print("\n⊘ Skipping plot generation")

    # Step 4: Generate summary tables
    cmd = [
        python_exe,
        str(project_root / 'examples' / 'generate_results_summary.py')
    ]

    if not run_command(cmd, "Results Summary Generation", timeout=60):
        failed_steps.append("Results Summary Generation")

    # Final summary
    overall_elapsed = perf_counter() - overall_start

    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"Total time: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")

    if failed_steps:
        print(f"\n✗ {len(failed_steps)} step(s) failed:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease review the errors above and rerun failed steps individually.")
        return 1
    else:
        print("\n✓ All steps completed successfully!")
        print("\nGenerated outputs:")
        print("  - results/hyperparameter_search_results.json")
        print("  - results/board_size_analysis.csv")
        print("  - results/depth_analysis.csv")
        print("  - results/game_length_distributions.csv")
        print("  - results/plots/ (all publication-quality figures)")
        print("  - results/results_summary.md")
        print("  - results/results_summary.tex")
        print("\nYou can now use these results for your CS221 final report!")
        return 0


if __name__ == '__main__':
    sys.exit(main())

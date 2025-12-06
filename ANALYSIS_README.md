# Quoridor AI Experimental Analysis Guide

This guide explains how to run the complete experimental analysis pipeline for your CS221 final project.

## Overview

The analysis pipeline includes:
- **Hyperparameter search** with Elo-based tournament evaluation (5×5×3 = 75 configurations)
- **Board size complexity analysis** (3×3, 5×5, 9×9 boards)
- **Depth performance analysis** (depths 1-5)
- **Game length distribution analysis** (5 matchup types)
- **7 publication-quality plots** for your report
- **Summary tables** in Markdown and LaTeX formats

## Prerequisites

Install analysis dependencies:
```bash
python -m pip install matplotlib seaborn pandas scipy
```

Or if using uv:
```bash
uv pip install matplotlib seaborn pandas scipy
```

## Quick Start: Run Everything

To run the complete analysis pipeline in one command:

```bash
python examples/run_complete_analysis.py
```

This will:
1. Run hyperparameter search (fine grid: 5×5×3 = 75 configs)
2. Collect all experimental data
3. Generate all 7 plots
4. Create summary tables

**Estimated time:**
- With default settings (100 games per analysis): ~2-4 hours
- With minimal settings (see below): ~30-60 minutes

## Quick Test Run

For a fast test with fewer games:

```bash
python examples/run_complete_analysis.py \
  --analysis-games 20 \
  --hyperparameter-games 4 \
  --config tiny
```

This completes in ~15-30 minutes and validates everything works.

## Step-by-Step: Run Individual Components

If you prefer to run components separately or customize parameters:

### 1. Hyperparameter Search

Run tournament-based hyperparameter search:

```bash
# Fine grid (5×5×3 = 75 configs) - RECOMMENDED for final report
python examples/hyperparameter_search.py --fine --config small --games-per-matchup 10

# Coarse grid (3×3×3 = 27 configs) - faster for testing
python examples/hyperparameter_search.py --coarse --config small --games-per-matchup 10
```

**Output:** `results/hyperparameter_search_results.json`

**Options:**
- `--fine` / `--coarse`: Grid resolution
- `--config`: Board size (tiny/small/standard)
- `--games-per-matchup`: Games between each pair of configs
- `--depth`: Search depth for all configs (default: 2)

### 2. Board Size Complexity Analysis

Analyze game complexity across different board sizes:

```bash
python examples/analyze_board_size_complexity.py --games 100
```

**Output:** `results/board_size_analysis.csv`

**Options:**
- `--games`: Number of games per board size
- `--configs`: Board sizes to test (default: tiny small standard)

### 3. Depth Performance Analysis

Evaluate minimax performance across search depths:

```bash
python examples/analyze_depth_performance.py \
  --games 100 \
  --depths 1 2 3 4 5 \
  --config small
```

**Output:** `results/depth_analysis.csv`

**Options:**
- `--games`: Games per depth
- `--depths`: Depths to test (space-separated)
- `--config`: Board size
- `--max-wall-moves`: Wall branching limit (default: 8)

### 4. Game Length Distributions

Collect game lengths across matchup types:

```bash
python examples/collect_game_length_distributions.py \
  --games 100 \
  --config small
```

**Output:** `results/game_length_distributions.csv`

**Options:**
- `--games`: Games per matchup type
- `--config`: Board size
- `--matchups`: Matchup types to test (default: all 5)

### 5. Generate Plots

Generate all 7 publication-quality plots:

```bash
# Board size complexity (justifies 5×5 choice)
python examples/plot_board_size_analysis.py

# Win rate vs depth with timing overlay
python examples/plot_depth_analysis.py

# Hyperparameter heatmaps (3 plots for wall weights 1, 2, 3)
python examples/plot_hyperparameter_heatmap.py

# Game length distributions
python examples/plot_game_length_distributions.py

# Elo rating evolution
python examples/plot_elo_evolution.py

# Move time distributions
python examples/plot_move_time_distributions.py

# First-player advantage
python examples/plot_first_player_advantage.py
```

**Output:** `results/plots/*.png` and `results/plots/*.pdf`

**Options for all plot scripts:**
- `--formats`: Output formats (default: png pdf)
- `--output-dir`: Directory for plots (default: results/plots)

### 6. Generate Summary Tables

Create Markdown and LaTeX tables for your report:

```bash
python examples/generate_results_summary.py
```

**Output:**
- `results/results_summary.md` (for README/wiki)
- `results/results_summary.tex` (for LaTeX report)

## Advanced Usage

### Custom Analysis Pipeline

Skip certain steps if you already have data:

```bash
# Skip hyperparameter search, use existing results
python examples/run_complete_analysis.py --skip-hyperparameter-search

# Only collect data, no plotting
python examples/run_complete_analysis.py --skip-plotting

# Use custom game counts
python examples/run_complete_analysis.py \
  --hyperparameter-games 20 \
  --analysis-games 200 \
  --config small
```

### Parallel Execution

If you have multiple cores, run data collection in parallel:

```bash
# Terminal 1
python examples/analyze_board_size_complexity.py --games 100

# Terminal 2
python examples/analyze_depth_performance.py --games 100 --depths 1 2 3 4 5

# Terminal 3
python examples/collect_game_length_distributions.py --games 100
```

Then generate all plots after data collection completes.

### Regenerate Plots Only

If you've already collected data and just want to regenerate plots:

```bash
# Regenerate all plots
for script in plot_*.py; do
    python examples/$script --formats png pdf
done

# Or use the master script
python examples/run_complete_analysis.py \
  --skip-hyperparameter-search \
  --skip-data-collection
```

## Output Files

After running the complete pipeline, you'll have:

```
results/
├── hyperparameter_search_results.json  # Tournament results with Elo ratings
├── board_size_analysis.csv             # Board complexity data
├── depth_analysis.csv                  # Depth performance data
├── game_length_distributions.csv       # Game length data
├── results_summary.md                  # Markdown tables
├── results_summary.tex                 # LaTeX tables
└── plots/
    ├── board_size_complexity.png       # Figure 1
    ├── depth_performance.png           # Figure 2
    ├── hyperparameter_heatmap_wall1.0.png  # Figure 3a
    ├── hyperparameter_heatmap_wall2.0.png  # Figure 3b
    ├── hyperparameter_heatmap_wall3.0.png  # Figure 3c
    ├── game_length_distributions.png   # Figure 4
    ├── elo_evolution.png               # Figure 5
    ├── move_time_distributions.png     # Figure 6
    ├── first_player_advantage.png      # Figure 7 (bonus)
    └── *.pdf versions of all plots
```

## Recommended Settings for Final Report

For high-quality results suitable for publication:

```bash
python examples/run_complete_analysis.py \
  --config small \
  --hyperparameter-games 10 \
  --analysis-games 100 \
  --plot-formats png pdf
```

**Estimated time:** 2-4 hours depending on your machine.

For faster results with still-acceptable quality:

```bash
python examples/run_complete_analysis.py \
  --config small \
  --hyperparameter-games 6 \
  --analysis-games 50 \
  --plot-formats png
```

**Estimated time:** 1-2 hours.

## Troubleshooting

### "Module not found" errors
Install analysis dependencies:
```bash
python -m pip install matplotlib seaborn pandas scipy
```

### Plots look different from examples
Ensure you're using the same matplotlib backend:
```bash
export MPLBACKEND=Agg  # For headless environments
```

### Out of memory errors
Reduce game counts or use a smaller board:
```bash
python examples/run_complete_analysis.py \
  --config tiny \
  --analysis-games 50 \
  --hyperparameter-games 4
```

### Want to resume after failure
Run individual scripts manually, starting from where it failed. The master script will use existing data files if they exist.

## Using Results in Your Report

### Figures
All plots are saved in both PNG (for presentations) and PDF (for LaTeX) formats. Use the PDF versions in your LaTeX report for best quality.

### Tables
Include the generated LaTeX tables directly in your report:
```latex
\input{results/results_summary.tex}
```

Or use the Markdown tables in your README/wiki.

### Statistics
All scripts print detailed statistics to the console. Redirect to a file for later reference:
```bash
python examples/run_complete_analysis.py 2>&1 | tee analysis_log.txt
```

## Questions?

- Check the source code for each script - they all have detailed docstrings
- Run any script with `--help` to see all options
- Review the test data we generated to see expected output formats

## Citation

If you use this analysis infrastructure in your work, please cite:

```
Quoridor AI Analysis Pipeline
Stanford CS221 Fall 2024
```

Good luck with your final report! 🎓

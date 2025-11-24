# GAZE-Quoridor
**G**umbel · **A**lpha · **Z**ero · **E**ngine — *Quoridor*

A deep reinforcement learning agent for the board game **Quoridor**, based on the AlphaZero algorithm and enhanced with Gumbel-based Monte Carlo Tree Search.

## Overview

This project will implement a Quoridor game engine optimized for reinforcement learning training with Gumbel AlphaZero.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Playing the Game

The project includes interactive command-line interfaces for playing Quoridor.

### Human vs Human

Play a local game with two human players:

```bash
uv run python examples/play_human_vs_human.py
```

Options:
- `--config [standard|small|tiny]` - Choose board size (default: standard)
  - `standard`: 9x9 board, 10 walls per player
  - `small`: 5x5 board, 5 walls per player
  - `tiny`: 3x3 board, 2 walls per player

Example:
```bash
uv run python examples/play_human_vs_human.py --config tiny
```

### Human vs Random Agent

Play against a random agent that uses a 50-50 move type selection strategy:

```bash
uv run python examples/play_human_vs_random.py
```

Options:
- `--config [standard|small|tiny]` - Choose board size (default: standard)
- `--human-player [1|2]` - Choose which player you control (default: 1)

Examples:
```bash
# Play as Player 1 on standard board
uv run python examples/play_human_vs_random.py

# Play as Player 2 on tiny board
uv run python examples/play_human_vs_random.py --config tiny --human-player 2
```

### Human vs Baseline Agent

Play against a baseline agent that uses pathfinding strategy:

```bash
uv run python examples/play_human_vs_baseline.py
```

The baseline agent:
- Uses 50-50 selection between pawn moves and wall placements
- **Pawn moves**: Follows the shortest path to its goal
- **Wall placements**: Blocks your next move on your shortest path

This provides a more strategic opponent than the random agent!

Options:
- `--config [standard|small|tiny]` - Choose board size (default: standard)
- `--human-player [1|2]` - Choose which player you control (default: 1)

Examples:
```bash
# Play as Player 1 on standard board
uv run python examples/play_human_vs_baseline.py

# Play as Player 2 on small board
uv run python examples/play_human_vs_baseline.py --config small --human-player 2
```

### Human vs Minimax Agent

Challenge a minimax agent that searches ahead using alpha-beta pruning:

```bash
uv run python examples/play_human_vs_minimax.py
```

Options:
- `--config [standard|small|tiny]` - Choose board size (default: standard)
- `--human-player [1|2]` - Choose which player you control (default: 1)
- `--depth N` - Set search depth in plies (default: 3)
- `--max-wall-moves M` - Limit wall placements explored per node (default: 8, use 0 for no limit)

Examples:
```bash
# Play as Player 1 with deeper search
uv run python examples/play_human_vs_minimax.py --depth 4

# Play as Player 2 on a tiny board with no wall pruning
uv run python examples/play_human_vs_minimax.py --config tiny --human-player 2 --max-wall-moves 0
```

### Minimax vs Automated Agents

Run automated matches where the minimax agent battles other built-in agents (random and baseline). Every move is streamed to the CLI and a JSON log containing the complete move history plus the final result is saved under `examples/match_logs`.

```bash
# Minimax (Player 1) vs random agent, log stored automatically
uv run python examples/play_minimax_vs_agents.py

# Play both automated opponents back-to-back with deeper search
uv run python examples/play_minimax_vs_agents.py --opponent all --depth 4

# Make minimax control Player 2 on the small board configuration
uv run python examples/play_minimax_vs_agents.py --config small --minimax-player 2
```

Key options:
- `--opponent [random|baseline|all]` - Choose specific opponent or face all automated agents sequentially.
- `--depth` / `--max-wall-moves` - Same search controls as the human-vs-minimax example.
- `--log-dir PATH` - Customize where match logs (with move history + winner metadata) are archived.
- `--show-board` - Render the board after every automated move for step-by-step visualization.

Each move recorded in automated matches now includes the agent key and decision time in milliseconds (`decision_time_ms`) for both the minimax and heuristic opponents.

### Move Time Statistics

Summarize move decision times across saved logs:

```bash
uv run python examples/summarize_move_times.py --agent minimax
# To inspect the heuristic baseline instead:
uv run python examples/summarize_move_times.py --agent baseline
```

The summary groups samples by agent and configuration (for minimax: depth and wall branching limit) and reports the mean and variance of per-move timings.

### Batch Agent Benchmarks

Run repeated matchups and print win/loss rates plus move-time statistics (mean + variance):

```bash
# Baseline vs Random, Minimax vs Baseline, Minimax vs Random (10 games each)
uv run python examples/run_agent_benchmarks.py -n 10 --config standard --depth 3
```

The script alternates starting players for fairness and reports per-agent win/loss/draw counts along with decision-time summaries. Use `--max-wall-moves 0` to disable wall-branch pruning for minimax.

### Move Notation

When playing, use the following notation:

- **Pawn moves**: `<column><row>` (e.g., `e5`)
  - Columns are letters (a-i for standard board)
  - Rows are numbers (1-9 for standard board)

- **Horizontal walls**: `<column><row>h` (e.g., `e4h`)
  - Blocks vertical movement between rows

- **Vertical walls**: `<column><row>v` (e.g., `e4v`)
  - Blocks horizontal movement between columns

The game displays all legal moves before each turn to help you play.

## Development

```bash
# Install with dev dependencies
uv sync

# Run all tests (when implemented)
uv run pytest

# Run tests with coverage
uv run pytest --cov=src
```

## Documentation

- [docs/SPECIFICATION.md](docs/SPECIFICATION.md) - Complete technical specification including state representation, action encoding, neural network architecture, and all game rules
- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) - Repository setup, build order, and implementation checklist

## License

This project is for educational purposes as part of Stanford CS221 (Artificial Intelligence: Principles and Techniques).

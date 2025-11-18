# Quoridor Examples

This directory contains example scripts for playing and testing Quoridor games.

## Available Examples

### Interactive Games

#### Human vs Human
Play a local two-player game:
```bash
uv run python examples/play_human_vs_human.py [--config BOARD_SIZE]
```

#### Human vs Random Agent
Play against a random agent:
```bash
uv run python examples/play_human_vs_random.py [--config BOARD_SIZE] [--human-player PLAYER_NUM]
```

#### Human vs Baseline Agent
Play against a baseline agent with pathfinding strategy:
```bash
uv run python examples/play_human_vs_baseline.py [--config BOARD_SIZE] [--human-player PLAYER_NUM]
```

**Baseline Agent Strategy:**
- 50-50 selection between pawn moves and wall placements
- Pawn moves: Follows shortest path to goal (A* pathfinding)
- Wall placements: Blocks opponent's next move on their shortest path

**Options:**
- `--config [standard|small|tiny]`: Board size (default: standard)
  - `standard`: 9x9 board, 10 walls per player
  - `small`: 5x5 board, 5 walls per player
  - `tiny`: 3x3 board, 2 walls per player
- `--human-player [1|2]`: Which player you control (default: 1, only for vs agent games)

#### Human vs Minimax Agent
Play against an alpha-beta minimax agent with configurable search depth:
```bash
uv run python examples/play_human_vs_minimax.py [--config BOARD_SIZE] [--human-player PLAYER_NUM] [--depth N] [--max-wall-moves M]
```

**Minimax Agent Highlights:**
- Minimax search with alpha-beta pruning and move ordering
- Heuristic considers path distance, pawn progress, and wall reserves
- Optional pruning of low-priority wall placements via `--max-wall-moves`

**Options:**
- `--depth N`: Search depth in plies (default: 3)
- `--max-wall-moves M`: Limit explored wall placements per node (default: 8, set `0` for no limit)

#### Minimax vs Automated Agents
Watch the minimax agent battle the other automated agents (random or baseline) without any human input. Every move is printed to the CLI and a JSON log is saved under `examples/match_logs`.
```bash
uv run python examples/play_minimax_vs_agents.py [--opponent OPP] [--depth N] [--max-wall-moves M]
```

**Highlights:**
- Supports one-off matches (e.g., `--opponent random`) or a full gauntlet (`--opponent all`)
- Streams each move with notation, move type, and player number
- Stores detailed logs (config, parameters, move history, winner metadata, final board snapshot)

**Useful options:**
- `--minimax-player [1|2]`: Choose which side the minimax agent controls (default: 1)
- `--log-dir PATH`: Customize where JSON match logs are written
- `--show-board`: Render the board after each move for detailed play-by-play

### Agent Examples (Coming Soon)

The following examples are placeholders for future implementations:
- `play_baseline.py` - Baseline agent gameplay
- `play_oracle.py` - Oracle agent gameplay
- `play_random.py` - Random agent gameplay

## Move Notation Quick Reference

| Move Type | Notation | Example | Description |
|-----------|----------|---------|-------------|
| Pawn move | `<col><row>` | `e5` | Move pawn to position |
| Horizontal wall | `<col><row>h` | `e4h` | Place horizontal wall (blocks vertical movement) |
| Vertical wall | `<col><row>v` | `e4v` | Place vertical wall (blocks horizontal movement) |

**Notes:**
- Columns are letters: `a`, `b`, `c`, ... (up to `i` for standard board)
- Rows are numbers: `1`, `2`, `3`, ... (up to `9` for standard board)
- The game displays all legal moves before each turn

## Quick Start

For a quick test game on a small board:
```bash
uv run python examples/play_human_vs_random.py --config tiny
```

This starts a 3x3 game where you can quickly see how the game works!

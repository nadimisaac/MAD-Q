# Quoridor Examples

This directory contains example scripts for playing and testing Quoridor games.

## Available Examples

### 🎮 Interactive Games

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
  - `standard`: 9×9 board, 10 walls per player
  - `small`: 5×5 board, 5 walls per player
  - `tiny`: 3×3 board, 2 walls per player
- `--human-player [1|2]`: Which player you control (default: 1, only for vs agent games)

### 🤖 Agent Examples (Coming Soon)

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

This starts a 3×3 game where you can quickly see how the game works!

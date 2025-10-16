# Playing Quoridor

## Quick Start

To play a human vs human game:

```bash
# Standard 9x9 board with 10 walls per player
uv run python examples/play_human_vs_human.py

# Small 5x5 board with 5 walls per player (faster games)
uv run python examples/play_human_vs_human.py --config small

# Tiny 3x3 board with 2 walls per player (quick test)
uv run python examples/play_human_vs_human.py --config tiny
```

## Game Rules

**Objective:**
- **Player 1** (bottom): Start at row 1, reach row 9 (top)
- **Player 2** (top): Start at row 9, reach row 1 (bottom)

**How to Move:**

### Pawn Moves
- Move one square in any of 4 directions (up, down, left, right)
- **Example:** `e5` - Move pawn to column e, row 5

### Jumping
- **Straight jump:** If opponent is adjacent, jump over them (2 squares)
- **Diagonal jump:** If opponent is adjacent but blocked behind, jump diagonally

### Wall Placement
- Place 2-cell walls to block opponent's path
- **Horizontal wall:** `e4h` - Blocks vertical movement above row 4
- **Vertical wall:** `e4v` - Blocks horizontal movement right of column e
- Each player starts with 10 walls (standard game)
- Walls cannot overlap or cross
- Walls cannot completely block a player from reaching their goal

## Move Notation

### Coordinates
- **Columns:** a-i (left to right)
- **Rows:** 1-9 (bottom to top)

### Examples
```
e5      - Move pawn to e5
e2      - Move pawn to e2
d4h     - Place horizontal wall at d4 (blocks movement between rows 4 and 5)
c3v     - Place vertical wall at c3 (blocks movement between columns c and d)
```

## Board Display

```
  a b c d e f g h i
9 . . . . 2 . . . . 9

8 . . . . . . . . . 8

7 . . . . . . . . . 7

6 . . . . . . . . . 6

5 . . . . . . . . . 5

4 . . . .|. . . . . 4

3 . . . .|. . . . . 3
       ---
2 . . . . . . . . . 2

1 . . . . 1 . . . . 1
  a b c d e f g h i
```

- `1` = Player 1's pawn
- `2` = Player 2's pawn
- `|` = Vertical wall (blocks left-right movement)
- `---` = Horizontal wall (blocks up-down movement)

## Tips

1. **Start with pawn moves** to get comfortable with the controls
2. **Plan ahead** - walls are limited and permanent
3. **Block strategically** - use walls to extend opponent's path
4. **Watch for shortcuts** - jumping can save moves
5. **Don't wall yourself in** - always maintain a path to your goal

## Game End

The game ends when:
- A player reaches their goal row (WIN)
- Maximum moves reached (DRAW)
- No walls placed for too long (DRAW - prevents stalling)

## Controls

- Type move and press Enter
- `Ctrl+C` to quit anytime

Enjoy your game!

# Pygame UI for Quoridor

A graphical user interface for playing Quoridor using Pygame.

## Features

### Visual Display
- **Grid board** with clear cell boundaries
- **Coordinate labels** (a-i columns, 1-9 rows) on all four sides
- **Player pawns** with distinct colors:
  - Player 1: Light brown/tan with "1" label
  - Player 2: Dark brown with "2" label
- **Walls** displayed as dark brown bars
- **Legal moves overlay** with green highlighting and notation labels
- **Hover tooltips** showing cell notation

### Game Information Sidebar
- Current player indicator
- Player positions in notation format (e.g., "e5", "e7")
- Walls remaining for each player
- Move count
- Mode indicator (PAWN MODE / WALL MODE)
- Game over display (winner announcement or draw)
- **Move history in 2-column format** (chess-style notation)
  - Column 1: Player 1's moves
  - Column 2: Player 2's moves
  - Move numbers on the left
- Control instructions

### Interactive Controls

**Mouse:**
- **Click on a cell** to move your pawn to that position
- **Click on wall position** to place a wall (when in wall mode)

**Keyboard:**
- **W** - Toggle between pawn movement mode and wall placement mode
- **R** - Rotate wall orientation (Horizontal ↔ Vertical)
- **L** - Show/hide legal moves overlay
- **ESC** - Quit the game

### Notation Display
All positions and moves are displayed in standard Quoridor notation:
- **Pawn positions**: Column letter + row number (e.g., "e5")
- **Wall placements**: Position + orientation (e.g., "e4h", "d5v")
- **Move history**: Scrollable list of all moves in notation format

This makes it easy to verify game correctness and reproduce games.

## Usage

### Human vs Human

Play a local game with two human players:

```bash
# Standard 9x9 board
uv run python examples/play_pygame.py

# Small 5x5 board
uv run python examples/play_pygame.py --config small

# Tiny 3x3 board (for quick testing)
uv run python examples/play_pygame.py --config tiny
```

### Human vs AI

Play against an AI opponent:

```bash
# Play as Player 1 against baseline AI
uv run python examples/play_pygame_vs_ai.py

# Play as Player 2
uv run python examples/play_pygame_vs_ai.py --human-player 2

# Play against random agent
uv run python examples/play_pygame_vs_ai.py --opponent random

# Small board
uv run python examples/play_pygame_vs_ai.py --config small
```

## Game Rules Reminder

### Objective
- **Player 1** (Blue): Start at row 1, reach row 9
- **Player 2** (Red): Start at row 9, reach row 1

### Pawn Movement
- Move one square in any cardinal direction (up, down, left, right)
- **Jump over opponent** if adjacent (straight or diagonal based on walls)

### Wall Placement
- Each player starts with 10 walls (standard game)
- Walls block movement between cells
- **Horizontal walls** (e.g., "e4h"): Block vertical movement
- **Vertical walls** (e.g., "d5v"): Block horizontal movement
- Walls must not completely block a player's path to their goal

### Win Conditions
- First player to reach their goal row wins
- Game can end in draw if maximum moves reached

## Technical Details

### Architecture

**PygameUI Class** (`src/ui/pygame_ui.py`):
- Renders game board and UI elements
- Handles user input (mouse and keyboard)
- Maintains move history
- Provides notation tooltips and overlays

**Integration:**
- Uses `State` class for game logic
- Converts coordinates using `notation.converter`
- Compatible with all agent types (Human, Random, Baseline)

### Layout
- **Board area**: Left side with coordinate labels
- **Sidebar**: Right side (350px) with game info and controls
- **Total window size**: 1200x800px (adjustable)
- **Cell size**: Auto-calculated based on board size

### Colors
- Board: Light gray background with black borders
- Grid: Gray lines
- Legal moves: Semi-transparent green
- Walls: Dark brown with black outline
- Tooltips: Gold background with black border

## Development

### Adding New Features

To add new UI features, modify `src/ui/pygame_ui.py`:

1. **Rendering**: Add methods in the `_render_*` family
2. **Interaction**: Extend `_handle_click`, `_handle_hover`, or `_handle_keypress`
3. **UI State**: Add properties to track new state

### Testing

Run the UI with different configurations:

```bash
# Quick test with tiny board
uv run python examples/play_pygame.py --config tiny

# Full game
uv run python examples/play_pygame.py --config standard

# Against AI
uv run python examples/play_pygame_vs_ai.py --config small
```

### Known Limitations

- Undo feature not yet implemented (planned)
- No game save/load functionality
- Move history shows last 15 moves only (scrolling planned)
- AI moves appear instantly (could add animation)

## Troubleshooting

### "Module not found" errors
Make sure you run with `uv run`:
```bash
uv run python examples/play_pygame.py
```

### Pygame window doesn't appear
- Check that pygame is installed: `uv pip list | grep pygame`
- Try reinstalling: `uv pip install pygame --force-reinstall`

### Game runs but nothing happens on click
- Ensure you're clicking within the board area (not the sidebar)
- Check terminal for error messages
- Verify you're in the correct mode (pawn vs wall)

## Screenshots

The UI displays:
- Clear board grid with labeled coordinates (a-i, 1-9)
- Player pawns (blue and red circles with numbers)
- Placed walls (brown bars)
- Current game state in sidebar
- Legal moves when hovering/pressing L
- Move history with notation

## Future Enhancements

Potential improvements:
- [ ] Undo/redo functionality
- [ ] Game save/load
- [ ] Replay mode for recorded games
- [ ] Animation for moves
- [ ] Sound effects
- [ ] Policy heatmap overlay (for AI analysis)
- [ ] Value estimates display
- [ ] Scrollable move history
- [ ] Game timer
- [ ] Move validation hints

---

For more information about the Quoridor implementation, see the main [README.md](../README.md) and [SPECIFICATION.md](SPECIFICATION.md).

"""Board rendering and visualization utilities for Quoridor.

Provides ASCII and graphical rendering of game states for debugging and display.
"""

from typing import Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def render_board_ascii(game_state) -> str:
    """Render board state as ASCII art.

    Args:
        game_state: GameState object to render

    Returns:
        Multi-line string representing the board

    Example output for 9x9 board:
      a b c d e f g h i
    9 . . . . 2 . . . . 9
      +---+           +
    8 . | . . . . . . . 8
      +   +---+       +
    7 . . . | . . . . . 7
      ...
    1 . . . . 1 . . . . 1
      a b c d e f g h i
    """
    board_size = game_state.config.board_size
    lines = []

    # Top column labels (with wider spacing to match cell spacing)
    col_labels = '   ' + '   '.join(chr(ord('a') + i) for i in range(board_size))
    lines.append(col_labels)

    # Render board from top to bottom (high row numbers to low)
    for row in range(board_size - 1, -1, -1):
        # Cell row
        cell_line = f"{row + 1} "  # Row label (1-indexed)
        if row + 1 < 10:
            cell_line += " "  # Extra space for single-digit row numbers

        for col in range(board_size):
            # Check if player pawn is here
            if game_state.player1_pos.row == row and game_state.player1_pos.col == col:
                cell_line += "1"
            elif game_state.player2_pos.row == row and game_state.player2_pos.col == col:
                cell_line += "2"
            else:
                cell_line += "."

            # Add space or vertical wall (with extra spacing)
            if col < board_size - 1:
                # Check for vertical wall between col and col+1
                # Wall at (row, col) or (row-1, col) blocks this
                if ((row, col) in game_state.v_walls or
                    (row > 0 and (row - 1, col) in game_state.v_walls)):
                    cell_line += " | "
                else:
                    cell_line += "   "

        cell_line += f"  {row + 1}"  # Row label on right
        lines.append(cell_line)

        # Wall row (horizontal walls below this cell row)
        if row > 0:
            wall_line = "   "
            for col in range(board_size):
                # Horizontal wall at position (r, c) spans columns c and c+1
                # When rendering column 'col', show wall if:
                # - Wall starts at this column: (row-1, col) exists, OR
                # - Wall started at previous column and spans here: (row-1, col-1) exists

                # Check if wall starts at this column
                has_wall_here = (row - 1, col) in game_state.h_walls
                # Check if wall started at previous column (and thus spans to this one)
                has_wall_from_left = col > 0 and (row - 1, col - 1) in game_state.h_walls

                wall_line += "═" if (has_wall_here or has_wall_from_left) else " "

                if col < board_size - 1:
                    # Add spacing between columns (must match cell spacing of 3 chars)
                    # Show wall in the gap if wall starts at current column
                    if (row - 1, col) in game_state.h_walls:
                        wall_line += "═══"  # Wall continues from col to col+1 (3 chars to match " | " or "   ")
                    else:
                        wall_line += "   "  # No wall (3 spaces to match cell spacing)

            lines.append(wall_line)

    # Bottom column labels
    lines.append(col_labels)

    return '\n'.join(lines)


def render_board_matplotlib(
    game_state,
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None,
    show: bool = True
):
    """Render board state using matplotlib.

    Args:
        game_state: GameState object to render
        figsize: Figure size (width, height) in inches
        save_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        matplotlib Figure object if not shown, None otherwise
    """
    raise NotImplementedError()


def print_game_info(game_state) -> None:
    """Print game state information to console.

    Args:
        game_state: GameState object

    Prints:
        - Current player
        - Move number
        - Walls remaining for each player
        - Player positions
        - Game status (ongoing/finished)
    """
    raise NotImplementedError()


def render_move_history(move_history: list) -> str:
    """Render move history in notation format.

    Args:
        move_history: List of (action_type, position) tuples

    Returns:
        Formatted string of moves with move numbers

    Example:
        1. e5 e6h
        2. d4 e3v
        3. e4 ...
    """
    raise NotImplementedError()


def save_game_animation(
    game_states: list,
    output_path: str,
    fps: int = 2
) -> None:
    """Create animated visualization of game progression.

    Args:
        game_states: List of GameState objects in sequence
        output_path: Path to save animation (e.g., "game.gif" or "game.mp4")
        fps: Frames per second for animation

    Raises:
        ImportError: If required animation libraries not available
    """
    raise NotImplementedError()


def plot_training_metrics(
    metrics_dict: dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot training metrics over time.

    Args:
        metrics_dict: Dictionary mapping metric names to lists of values
                     e.g., {'loss': [...], 'win_rate': [...]}
        save_path: If provided, save figure to this path
        show: If True, display the figure
    """
    raise NotImplementedError()


def visualize_policy_heatmap(
    game_state,
    policy_probs: list,
    action_space,
    save_path: Optional[str] = None
) -> None:
    """Visualize policy distribution as heatmap overlay on board.

    Args:
        game_state: Current GameState
        policy_probs: List of action probabilities (length = action_space size)
        action_space: ActionSpace object for decoding
        save_path: If provided, save figure to this path
    """
    raise NotImplementedError()


def visualize_value_estimates(
    game_state,
    value: float,
    policy_value_pairs: list = None
) -> None:
    """Visualize value estimate and top policy actions.

    Args:
        game_state: Current GameState
        value: Value estimate for current position
        policy_value_pairs: List of (action, probability) tuples to highlight
    """
    raise NotImplementedError()

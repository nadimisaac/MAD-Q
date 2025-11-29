"""Notation conversion for Quoridor moves.

Converts between internal representation and standard Quoridor notation.

Standard notation:
- Pawn moves: "e5", "d4" (column letter a-i, row number 1-9)
- Horizontal walls: "e3h" - wall above e3 and f3 (e3 is bottom-left reference)
- Vertical walls: "e3v" - wall to the right of e3 and e4 (e3 is bottom-left reference)

Wall placement convention:
The cell notation (e.g., "e3") refers to the BOTTOM-LEFT cell adjacent to the wall.

- "e3h" (horizontal wall):
  * Reference cell: e3 (bottom-left)
  * Wall placement: ABOVE e3 and f3
  * Blocks vertical movement between these cells and the cells above them

- "e3v" (vertical wall):
  * Reference cell: e3 (bottom-left)
  * Wall placement: TO THE RIGHT of e3 and e4
  * Blocks horizontal movement between these cells and the cells to their right

For a 9x9 board:
- Pawn positions: rows 0-8, cols 0-8 (notation: a1-i9)
- Wall positions: rows 0-7, cols 0-7 (notation: a1-h8 + h/v suffix)
"""

from typing import Tuple


def position_to_notation(row: int, col: int, is_wall: bool = False) -> str:
    """Convert (row, col) position to algebraic notation.

    Convention for wall positions:
    - The position refers to the BOTTOM-LEFT reference cell
    - For "e3h": position (2, 4) means wall ABOVE cells e3 and f3
    - For "e3v": position (2, 4) means wall to RIGHT of cells e3 and e4

    Args:
        row: Row index (0-8 for pawn moves, 0-7 for walls on 9x9 board)
        col: Column index (0-8 for pawn moves, 0-7 for walls on 9x9 board)
        is_wall: True if position refers to a wall placement

    Returns:
        Algebraic notation string without orientation suffix
        Column letters: a-i, Row numbers: 1-9 (pawn) or 1-8 (wall)

    Examples:
        position_to_notation(4, 4, False) -> "e5" (pawn)
        position_to_notation(2, 4, True) -> "e3" (wall reference cell)
    """
    col_letter = chr(ord('a') + col)
    row_number = row + 1  # Convert to 1-indexed
    return f"{col_letter}{row_number}"


def notation_to_position(notation: str) -> Tuple[int, int]:
    """Convert algebraic notation to (row, col) position.

    Handles both pawn moves ("e5") and walls ("e4h", "e4v").
    Strips orientation suffix if present.

    Convention for wall notation:
    - "e3h" -> (2, 4): wall ABOVE cells e3 and f3
    - "e3v" -> (2, 4): wall to RIGHT of cells e3 and e4
    - The returned position is the BOTTOM-LEFT reference cell

    Args:
        notation: Algebraic notation (e.g., "e5", "e4h", "e4v")

    Returns:
        Tuple of (row, col) indices

    Raises:
        ValueError: If notation is invalid

    Examples:
        notation_to_position("e5") -> (4, 4)
        notation_to_position("e3h") -> (2, 4)
        notation_to_position("e3v") -> (2, 4)
    """
    if not notation or len(notation) < 2:
        raise ValueError(f"Invalid notation: {notation}")

    # Strip orientation suffix if present
    base_notation = notation.rstrip('hv')

    if len(base_notation) < 2:
        raise ValueError(f"Invalid notation: {notation}")

    col_letter = base_notation[0].lower()
    row_str = base_notation[1:]

    # Validate column letter
    if col_letter < 'a' or col_letter > 'z':
        raise ValueError(f"Invalid column letter: {col_letter}")

    col = ord(col_letter) - ord('a')

    # Validate row number
    try:
        row_number = int(row_str)
    except ValueError:
        raise ValueError(f"Invalid row number: {row_str}")

    if row_number < 1:
        raise ValueError(f"Row number must be at least 1, got {row_number}")

    row = row_number - 1  # Convert to 0-indexed

    return (row, col)


def action_to_notation(action_type: str, position: Tuple[int, int]) -> str:
    """Convert action to standard notation.

    Args:
        action_type: Type of action ('move', 'h_wall', 'v_wall')
        position: (row, col) position

    Returns:
        Notation string:
        - 'move': "e5"
        - 'h_wall': "e4h" (wall ABOVE e4 and f4)
        - 'v_wall': "e4v" (wall to RIGHT of e4 and e5)
    """
    row, col = position
    base = position_to_notation(row, col, is_wall=(action_type != 'move'))

    if action_type == 'move':
        return base
    elif action_type == 'h_wall':
        return base + 'h'
    elif action_type == 'v_wall':
        return base + 'v'
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def notation_to_action(notation: str) -> Tuple[str, Tuple[int, int]]:
    """Parse notation string into action type and position.

    Args:
        notation: Notation string (e.g., "e5", "e4h", "e4v")

    Returns:
        Tuple of (action_type, position)
        action_type: 'move', 'h_wall', or 'v_wall'
        position: (row, col) tuple

    Raises:
        ValueError: If notation is invalid

    Examples:
        "e5" -> ('move', (4, 4))
        "e3h" -> ('h_wall', (2, 4))  # wall ABOVE e3 and f3
        "e3v" -> ('v_wall', (2, 4))  # wall to RIGHT of e3 and e4
    """
    if not notation:
        raise ValueError("Empty notation string")

    notation = notation.strip().lower()

    # Check for wall orientation suffix
    if notation.endswith('h'):
        action_type = 'h_wall'
    elif notation.endswith('v'):
        action_type = 'v_wall'
    else:
        action_type = 'move'

    position = notation_to_position(notation)
    return (action_type, position)


def move_list_to_notation(moves: list) -> str:
    """Convert list of moves to space-separated notation string.

    Args:
        moves: List of (action_type, position) tuples

    Returns:
        Space-separated notation string (e.g., "e5 e6h d4")
    """
    raise NotImplementedError()


def notation_to_move_list(notation_string: str) -> list:
    """Parse notation string into list of moves.

    Args:
        notation_string: Space-separated notation (e.g., "e5 e6h d4")

    Returns:
        List of (action_type, position) tuples
    """
    raise NotImplementedError()


def validate_notation(notation: str) -> bool:
    """Check if notation string is valid.

    Args:
        notation: Notation string to validate

    Returns:
        True if notation is valid
    """
    raise NotImplementedError()

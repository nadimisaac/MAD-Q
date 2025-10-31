"""Low-level, stateless helper functions for move generation and validation.

These are pure functions that perform geometry and rule-based calculations.
They are used internally by State class methods. Do not import these directly -
use State.get_legal_pawn_moves() and related public methods instead.
"""

from typing import List, Tuple, Set


def _get_adjacent_positions(pos: Tuple[int, int], board_size: int) -> List[Tuple[int, int]]:
    """Get adjacent positions in 4 cardinal directions (stateless helper).

    Args:
        pos: Current position as (row, col) - 0-indexed
        board_size: Size of board (e.g., 9 for 9x9)

    Returns:
        List of adjacent positions within board bounds as (row, col) tuples
    """
    row, col = pos
    adjacent = []

    # Up, Down, Right, Left
    if row < board_size - 1:
        adjacent.append((row + 1, col))
    if row > 0:
        adjacent.append((row - 1, col))
    if col < board_size - 1:
        adjacent.append((row, col + 1))
    if col > 0:
        adjacent.append((row, col - 1))

    return adjacent


def _can_move_between(
    pos1: Tuple[int, int],
    pos2: Tuple[int, int],
    h_walls: Set[Tuple[int, int]],
    v_walls: Set[Tuple[int, int]]
) -> bool:
    """Check if movement between two adjacent positions is blocked by a wall (stateless helper).

    Wall convention (0-indexed, bottom-left reference):
    - Horizontal wall at (r, c): blocks vertical movement between row r and r+1, spans cols c and c+1
    - Vertical wall at (r, c): blocks horizontal movement between col c and c+1, spans rows r and r+1

    Args:
        pos1: Starting position (row, col) - 0-indexed
        pos2: Ending position (row, col) - must be adjacent, 0-indexed
        h_walls: Set of horizontal wall positions (bottom-left reference cell)
        v_walls: Set of vertical wall positions (bottom-left reference cell)

    Returns:
        True if movement is not blocked by any wall
    """
    r1, c1 = pos1
    r2, c2 = pos2

    # Moving vertically (same column)
    if c1 == c2:
        if r2 > r1:  # Moving up (from r1 to r1+1)
            # Horizontal wall at (r1, c1) or (r1, c1-1) blocks this
            if (r1, c1) in h_walls or (c1 > 0 and (r1, c1 - 1) in h_walls):
                return False
        else:  # Moving down (from r1 to r1-1)
            # Horizontal wall at (r2, c2) or (r2, c2-1) blocks this
            if (r2, c2) in h_walls or (c2 > 0 and (r2, c2 - 1) in h_walls):
                return False

    # Moving horizontally (same row)
    elif r1 == r2:
        if c2 > c1:  # Moving right (from c1 to c1+1)
            # Vertical wall at (r1, c1) or (r1-1, c1) blocks this
            if (r1, c1) in v_walls or (r1 > 0 and (r1 - 1, c1) in v_walls):
                return False
        else:  # Moving left (from c1 to c1-1)
            # Vertical wall at (r2, c2) or (r2-1, c2) blocks this
            if (r2, c2) in v_walls or (r2 > 0 and (r2 - 1, c2) in v_walls):
                return False

    return True


def _is_straight_jump(
    from_pos: Tuple[int, int],
    over_pos: Tuple[int, int],
    to_pos: Tuple[int, int]
) -> bool:
    """Check if jump is straight (same row or column) over opponent (stateless helper).

    Args:
        from_pos: Starting position (row, col) - 0-indexed
        over_pos: Position being jumped over (opponent) - 0-indexed
        to_pos: Landing position (row, col) - 0-indexed

    Returns:
        True if this is a valid straight jump (2 cells in one direction)
    """
    fr, fc = from_pos
    or_, oc = over_pos
    tr, tc = to_pos

    # Check if all three positions are in a straight line
    # Vertical jump (same column)
    if fc == oc == tc:
        # Check if opponent is adjacent and to_pos is 2 cells away in same direction
        if or_ == fr + 1 and tr == fr + 2:  # Jump up
            return True
        if or_ == fr - 1 and tr == fr - 2:  # Jump down
            return True

    # Horizontal jump (same row)
    if fr == or_ == tr:
        # Check if opponent is adjacent and to_pos is 2 cells away in same direction
        if oc == fc + 1 and tc == fc + 2:  # Jump right
            return True
        if oc == fc - 1 and tc == fc - 2:  # Jump left
            return True

    return False


def _is_diagonal_jump(
    from_pos: Tuple[int, int],
    over_pos: Tuple[int, int],
    to_pos: Tuple[int, int]
) -> bool:
    """Check if jump is diagonal (when opponent blocked from behind) (stateless helper).

    Args:
        from_pos: Starting position (row, col) - 0-indexed
        over_pos: Position being jumped over (opponent) - 0-indexed
        to_pos: Landing position (row, col) - 0-indexed

    Returns:
        True if this is a valid diagonal jump (1 forward toward opponent, 1 sideways)
    """
    fr, fc = from_pos
    or_, oc = over_pos
    tr, tc = to_pos

    # Diagonal jump: must be adjacent to opponent and move diagonally
    # Check if opponent is adjacent in one direction
    # and landing is adjacent to opponent in perpendicular direction

    # Opponent directly above/below
    if fc == oc and abs(or_ - fr) == 1:
        # Landing should be same row as opponent, one column away
        if tr == or_ and abs(tc - oc) == 1:
            return True

    # Opponent directly left/right
    if fr == or_ and abs(oc - fc) == 1:
        # Landing should be same column as opponent, one row away
        if tc == oc and abs(tr - or_) == 1:
            return True

    return False


def _walls_overlap(
    row1: int,
    col1: int,
    orientation1: str,
    row2: int,
    col2: int,
    orientation2: str
) -> bool:
    """Check if two walls overlap or cross (stateless helper).

    Wall spans (0-indexed):
    - Horizontal wall at (r, c): occupies cells (r, c) and (r, c+1)
    - Vertical wall at (r, c): occupies cells (r, c) and (r+1, c)

    Args:
        row1, col1: First wall position (bottom-left reference)
        orientation1: 'h' or 'v' for first wall
        row2, col2: Second wall position (bottom-left reference)
        orientation2: 'h' or 'v' for second wall

    Returns:
        True if walls overlap or create a cross pattern
    """
    # Same orientation - check for overlap
    if orientation1 == orientation2:
        if orientation1 == 'h':
            # Both horizontal: check if they share any cells
            if row1 == row2:
                # Same row: check if columns overlap
                # Wall 1: cols [col1, col1+1], Wall 2: cols [col2, col2+1]
                if col1 == col2 or col1 == col2 + 1 or col1 + 1 == col2:
                    return True
        else:  # Both vertical
            # Both vertical: check if they share any cells
            if col1 == col2:
                # Same col: check if rows overlap
                # Wall 1: rows [row1, row1+1], Wall 2: rows [row2, row2+1]
                if row1 == row2 or row1 == row2 + 1 or row1 + 1 == row2:
                    return True
    else:
        # Different orientations - check for cross pattern
        # A cross occurs when the reference cells of the two walls are the same
        if row1 == row2 and col1 == col2:
            return True

    return False

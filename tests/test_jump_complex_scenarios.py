"""Test more complex jump scenarios with multiple walls."""

from src.game.state import State
from src.game.config import GameConfig

def test_multiple_walls_blocking_jumps():
    """Test scenario with walls blocking some but not all jump options."""
    config = GameConfig(
        board_size=5,
        walls_per_player=5,
        max_moves=100,
        no_progress_limit=20,
        history_length=4,
        draw_penalty=-0.5
    )
    state = State(config)

    # Set up scenario:
    # Player 1 at (2, 2)
    # Player 2 at (3, 2) - directly above Player 1
    # Wall blocking straight jump to (4, 2)
    # Wall blocking diagonal jump to (3, 1) - left side
    # Only (3, 3) should be available as diagonal jump
    state.player1_pos.row = 2
    state.player1_pos.col = 2
    state.player2_pos.row = 3
    state.player2_pos.col = 2

    # Block straight jump: horizontal wall at (3, 2)
    state.h_walls.add((3, 2))

    # Block left diagonal: vertical wall at (3, 1)
    state.v_walls.add((3, 1))

    state.current_player = 1

    print("Test: Multiple walls blocking some jump directions")
    print(f"Player 1 at: {state.player1_pos}")
    print(f"Player 2 at: {state.player2_pos}")
    print(f"Horizontal walls: {state.h_walls}")
    print(f"Vertical walls: {state.v_walls}")
    print()

    legal_moves = state.get_legal_pawn_moves()
    print(f"Legal moves for Player 1: {legal_moves}")
    print()

    # Check straight jump (should be blocked)
    straight_jump = (4, 2)
    if straight_jump not in legal_moves:
        print(f"✓ Straight jump to {straight_jump} is correctly blocked by wall")
    else:
        print(f"✗ BUG: Straight jump to {straight_jump} should be blocked")

    # Check left diagonal (should be blocked)
    left_diagonal = (3, 1)
    if left_diagonal not in legal_moves:
        print(f"✓ Left diagonal jump to {left_diagonal} is correctly blocked by wall")
    else:
        print(f"✗ BUG: Left diagonal jump to {left_diagonal} should be blocked")

    # Check right diagonal (should be allowed)
    right_diagonal = (3, 3)
    if right_diagonal in legal_moves:
        print(f"✓ Right diagonal jump to {right_diagonal} is correctly allowed")
    else:
        print(f"✗ BUG: Right diagonal jump to {right_diagonal} should be allowed")

    print()


def test_all_jumps_blocked():
    """Test scenario where all jump options are blocked."""
    config = GameConfig(
        board_size=5,
        walls_per_player=5,
        max_moves=100,
        no_progress_limit=20,
        history_length=4,
        draw_penalty=-0.5
    )
    state = State(config)

    # Set up scenario:
    # Player 1 at (2, 2)
    # Player 2 at (3, 2) - directly above Player 1
    # Walls blocking ALL possible jumps
    state.player1_pos.row = 2
    state.player1_pos.col = 2
    state.player2_pos.row = 3
    state.player2_pos.col = 2

    # Block straight jump
    state.h_walls.add((3, 2))

    # Block both diagonals
    state.v_walls.add((3, 1))  # Block left diagonal
    state.v_walls.add((3, 2))  # Block right diagonal

    state.current_player = 1

    print("Test: All jump directions blocked")
    print(f"Player 1 at: {state.player1_pos}")
    print(f"Player 2 at: {state.player2_pos}")
    print(f"Horizontal walls: {state.h_walls}")
    print(f"Vertical walls: {state.v_walls}")
    print()

    legal_moves = state.get_legal_pawn_moves()
    print(f"Legal moves for Player 1: {legal_moves}")
    print()

    # No jumps should be possible
    possible_jumps = [(4, 2), (3, 1), (3, 3)]
    all_blocked = True
    for jump in possible_jumps:
        if jump in legal_moves:
            print(f"✗ BUG: Jump to {jump} should be blocked but is allowed")
            all_blocked = False

    if all_blocked:
        print(f"✓ All jumps correctly blocked")
        print(f"✓ Player can only move to other adjacent squares")

    print()


def test_corner_jump_scenario():
    """Test jumping when opponent is in a corner."""
    config = GameConfig(
        board_size=5,
        walls_per_player=5,
        max_moves=100,
        no_progress_limit=20,
        history_length=4,
        draw_penalty=-0.5
    )
    state = State(config)

    # Set up scenario:
    # Player 1 at (0, 3)
    # Player 2 at (0, 4) - in the corner
    # Can only jump diagonally up
    state.player1_pos.row = 0
    state.player1_pos.col = 3
    state.player2_pos.row = 0
    state.player2_pos.col = 4

    state.current_player = 1

    print("Test: Jump when opponent is in corner")
    print(f"Player 1 at: {state.player1_pos}")
    print(f"Player 2 at: {state.player2_pos}")
    print()

    legal_moves = state.get_legal_pawn_moves()
    print(f"Legal moves for Player 1: {legal_moves}")
    print()

    # Straight jump should not be possible (off board)
    # Only vertical diagonal should be available
    diagonal_jump = (1, 4)
    if diagonal_jump in legal_moves:
        print(f"✓ Diagonal jump to {diagonal_jump} is allowed")
    else:
        print(f"✗ BUG: Diagonal jump to {diagonal_jump} should be allowed")

    print()


if __name__ == "__main__":
    test_multiple_walls_blocking_jumps()
    test_all_jumps_blocked()
    test_corner_jump_scenario()

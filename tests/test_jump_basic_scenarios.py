"""Test script to verify jump scenarios and identify the bug."""

from src.game.state import State
from src.game.config import GameConfig

def test_diagonal_jump_when_blocked_behind():
    """Test that diagonal jumps are allowed when opponent is blocked from behind."""
    config = GameConfig(
        board_size=5,
        walls_per_player=3,
        max_moves=100,
        no_progress_limit=20,
        history_length=4,
        draw_penalty=-0.5
    )
    state = State(config)

    # Set up scenario:
    # Player 1 at (2, 2)
    # Player 2 at (2, 3) - to the right of Player 1
    # Horizontal wall blocking Player 2 from behind (can't jump to (2, 4))
    state.player1_pos.row = 2
    state.player1_pos.col = 2
    state.player2_pos.row = 2
    state.player2_pos.col = 3

    # Place vertical wall at (2, 3) to block movement from (2,3) to (2,4)
    # This blocks the straight jump
    state.v_walls.add((2, 3))

    state.current_player = 1

    print("Test 1: Diagonal jump when opponent blocked from behind")
    print(f"Player 1 at: {state.player1_pos}")
    print(f"Player 2 at: {state.player2_pos}")
    print(f"Vertical walls: {state.v_walls}")

    legal_moves = state.get_legal_pawn_moves()
    print(f"Legal moves for Player 1: {legal_moves}")

    # Expected: Should be able to jump diagonally to (1, 3) or (3, 3)
    expected_diagonal_jumps = [(1, 3), (3, 3)]
    for jump in expected_diagonal_jumps:
        if jump in legal_moves:
            print(f"✓ Diagonal jump to {jump} is allowed")
        else:
            print(f"✗ BUG: Diagonal jump to {jump} is NOT allowed (should be)")

    print()


def test_straight_jump_when_not_blocked():
    """Test that straight jump works when opponent is not blocked."""
    config = GameConfig(
        board_size=5,
        walls_per_player=3,
        max_moves=100,
        no_progress_limit=20,
        history_length=4,
        draw_penalty=-0.5
    )
    state = State(config)

    # Set up scenario:
    # Player 1 at (2, 2)
    # Player 2 at (2, 3) - to the right of Player 1
    # No walls blocking
    state.player1_pos.row = 2
    state.player1_pos.col = 2
    state.player2_pos.row = 2
    state.player2_pos.col = 3
    state.current_player = 1

    print("Test 2: Straight jump when opponent not blocked")
    print(f"Player 1 at: {state.player1_pos}")
    print(f"Player 2 at: {state.player2_pos}")

    legal_moves = state.get_legal_pawn_moves()
    print(f"Legal moves for Player 1: {legal_moves}")

    # Expected: Should be able to jump straight to (2, 4)
    straight_jump = (2, 4)
    if straight_jump in legal_moves:
        print(f"✓ Straight jump to {straight_jump} is allowed")
    else:
        print(f"✗ BUG: Straight jump to {straight_jump} is NOT allowed (should be)")

    # Diagonal jumps should NOT be allowed when straight jump is possible
    diagonal_jumps = [(1, 3), (3, 3)]
    for jump in diagonal_jumps:
        if jump not in legal_moves:
            print(f"✓ Diagonal jump to {jump} is correctly NOT allowed")
        else:
            print(f"✗ BUG: Diagonal jump to {jump} IS allowed (should not be)")

    print()


def test_diagonal_jump_at_board_edge():
    """Test diagonal jumps when opponent is at board edge."""
    config = GameConfig(
        board_size=5,
        walls_per_player=3,
        max_moves=100,
        no_progress_limit=20,
        history_length=4,
        draw_penalty=-0.5
    )
    state = State(config)

    # Set up scenario:
    # Player 1 at (2, 3)
    # Player 2 at (2, 4) - at the right edge
    # Can't jump straight because it's off the board
    state.player1_pos.row = 2
    state.player1_pos.col = 3
    state.player2_pos.row = 2
    state.player2_pos.col = 4
    state.current_player = 1

    print("Test 3: Diagonal jump when opponent at board edge")
    print(f"Player 1 at: {state.player1_pos}")
    print(f"Player 2 at: {state.player2_pos}")

    legal_moves = state.get_legal_pawn_moves()
    print(f"Legal moves for Player 1: {legal_moves}")

    # Expected: Should be able to jump diagonally to (1, 4) or (3, 4)
    expected_diagonal_jumps = [(1, 4), (3, 4)]
    for jump in expected_diagonal_jumps:
        if jump in legal_moves:
            print(f"✓ Diagonal jump to {jump} is allowed")
        else:
            print(f"✗ BUG: Diagonal jump to {jump} is NOT allowed (should be)")

    print()


def test_both_jumps_possible():
    """Test the scenario you described: wall on OTHER side allows diagonal."""
    config = GameConfig(
        board_size=5,
        walls_per_player=3,
        max_moves=100,
        no_progress_limit=20,
        history_length=4,
        draw_penalty=-0.5
    )
    state = State(config)

    # Set up scenario:
    # Player 1 at (2, 1)
    # Player 2 at (2, 2)
    # Wall at (2, 3) blocking the right side of Player 2
    # Straight jump to (2, 3) is possible (no wall between (2,2) and (2,3))
    # But there's a wall on the other side at (2, 3)
    state.player1_pos.row = 2
    state.player1_pos.col = 1
    state.player2_pos.row = 2
    state.player2_pos.col = 2

    # This might be what you mean - wall on far side of opponent
    state.v_walls.add((2, 3))  # Wall to the right of (2,3)

    state.current_player = 1

    print("Test 4: Checking if wall on far side affects jump options")
    print(f"Player 1 at: {state.player1_pos}")
    print(f"Player 2 at: {state.player2_pos}")
    print(f"Vertical walls: {state.v_walls}")

    legal_moves = state.get_legal_pawn_moves()
    print(f"Legal moves for Player 1: {legal_moves}")

    straight_jump = (2, 3)
    print(f"Can move straight to {straight_jump}: {straight_jump in legal_moves}")

    diagonal_jumps = [(1, 2), (3, 2)]
    for jump in diagonal_jumps:
        print(f"Can jump diagonally to {jump}: {jump in legal_moves}")

    print()


if __name__ == "__main__":
    test_diagonal_jump_when_blocked_behind()
    test_straight_jump_when_not_blocked()
    test_diagonal_jump_at_board_edge()
    test_both_jumps_possible()

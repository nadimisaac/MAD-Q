"""Reproduce the exact game state from the screenshot."""

from src.game.state import State
from src.game.config import load_game_config
from src.notation.converter import notation_to_action

# Play through the exact moves from the screenshot
config = load_game_config("standard")
state = State(config)

moves = ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e6h", "d5v", "e4h"]

print("Playing through moves from screenshot:")
print("=" * 60)

for i, move in enumerate(moves):
    print(f"\nMove {i+1}: {move}")

    action_type, pos = notation_to_action(move)

    if action_type == 'move':
        # Pawn move
        legal_moves = state.get_legal_pawn_moves()
        if pos in legal_moves:
            state.player1_pos.row = pos[0] if state.current_player == 1 else state.player1_pos.row
            state.player1_pos.col = pos[1] if state.current_player == 1 else state.player1_pos.col
            state.player2_pos.row = pos[0] if state.current_player == 2 else state.player2_pos.row
            state.player2_pos.col = pos[1] if state.current_player == 2 else state.player2_pos.col
            print(f"  Player {state.current_player} moved to {pos}")
        else:
            print(f"  ERROR: Move {pos} is not legal!")
            print(f"  Legal moves: {legal_moves}")
            break
    elif action_type == 'h_wall':
        state.h_walls.add(pos)
        print(f"  Player {state.current_player} placed horizontal wall at {pos}")
    elif action_type == 'v_wall':
        state.v_walls.add(pos)
        print(f"  Player {state.current_player} placed vertical wall at {pos}")

    # Switch player
    state.current_player = 2 if state.current_player == 1 else 1
    state.move_count += 1

print("\n" + "=" * 60)
print("Final game state:")
print(f"Player 1 at: {state.player1_pos}")
print(f"Player 2 at: {state.player2_pos}")
print(f"Horizontal walls: {state.h_walls}")
print(f"Vertical walls: {state.v_walls}")
print(f"Current player: {state.current_player}")

print("\n" + "=" * 60)
print("Legal moves for current player:")
legal_moves = state.get_legal_pawn_moves()

def pos_to_notation(pos):
    row, col = pos
    col_letter = chr(ord('a') + col)
    row_num = row + 1
    return f"{col_letter}{row_num}"

for move in legal_moves:
    print(f"  - {pos_to_notation(move)} {move}")

print("\n" + "=" * 60)
print("Analysis:")
print()

if state.current_player == 1:
    current_pos = state.player1_pos
    opponent_pos = state.player2_pos
else:
    current_pos = state.player2_pos
    opponent_pos = state.player1_pos

print(f"Current player {state.current_player} is at {current_pos} = {pos_to_notation((current_pos.row, current_pos.col))}")
print(f"Opponent is at {opponent_pos} = {pos_to_notation((opponent_pos.row, opponent_pos.col))}")
print()

# Check if opponent is adjacent
if abs(current_pos.row - opponent_pos.row) + abs(current_pos.col - opponent_pos.col) == 1:
    print("Opponent is adjacent - jump logic will be triggered")
    print()

    # Calculate straight jump position
    dr = opponent_pos.row - current_pos.row
    dc = opponent_pos.col - current_pos.col
    straight_jump = (opponent_pos.row + dr, opponent_pos.col + dc)

    print(f"Straight jump would be to {straight_jump} = {pos_to_notation(straight_jump)}")
    print(f"Is straight jump in legal moves? {straight_jump in legal_moves}")

    # Check diagonal jumps
    diagonal_positions = []
    for dr_diag, dc_diag in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        diag_pos = (opponent_pos.row + dr_diag, opponent_pos.col + dc_diag)
        if (diag_pos != (current_pos.row, current_pos.col) and
            0 <= diag_pos[0] < config.board_size and
            0 <= diag_pos[1] < config.board_size):
            diagonal_positions.append(diag_pos)

    print()
    print("Possible diagonal jumps:")
    for diag_pos in diagonal_positions:
        is_legal = diag_pos in legal_moves
        print(f"  - {pos_to_notation(diag_pos)} {diag_pos}: {'✓ LEGAL' if is_legal else '✗ NOT LEGAL'}")

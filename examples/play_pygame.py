#!/usr/bin/env python3
"""Play Quoridor with Pygame GUI - Human vs Human.

Launch a graphical Quoridor game with two human players.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.state import State
from src.game.config import load_game_config
from src.ui.pygame_ui import PygameUI


def main():
    """Run human vs human game with Pygame UI."""
    parser = argparse.ArgumentParser(description="Play Quoridor with Pygame GUI")
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "small", "tiny"],
        help="Game configuration (default: standard 9x9)"
    )
    args = parser.parse_args()

    # Load game configuration
    config = load_game_config(args.config)
    print(f"Starting {config.board_size}x{config.board_size} Quoridor game")
    print(f"Each player has {config.walls_per_player} walls")
    print()
    print("Controls:")
    print("  - Click on a cell to move your pawn")
    print("  - Press 'W' to toggle wall placement mode")
    print("  - Press 'R' to rotate wall orientation (H/V)")
    print("  - Press 'L' to show/hide legal moves")
    print("  - Press 'ESC' to quit")
    print()

    # Initialize game state
    state = State(config)

    # Create UI
    ui = PygameUI(state)

    # Define move callback
    def on_move(action_type: str, position: tuple):
        nonlocal state

        # Attempt to make the move
        try:
            new_state = state.make_move(action_type, position)
            state = new_state
            ui.update_state(state)

            # Check for game over
            result = state.check_termination()
            if result is not None:
                if result.winner is not None:
                    print(f"\n=== Player {result.winner} wins! ===")
                else:
                    print(f"\n=== Game ended in a draw: {result.reason} ===")

        except ValueError as e:
            print(f"Illegal move: {e}")

    # Run the UI
    ui.run(on_move=on_move)

    print("\nGame ended.")
    print(f"Final move count: {state.move_count}")


if __name__ == "__main__":
    main()

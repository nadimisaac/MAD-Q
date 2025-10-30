#!/usr/bin/env python3
"""Play Quoridor with Pygame GUI - Human vs AI.

Launch a graphical Quoridor game against an AI opponent.
"""

import sys
from pathlib import Path
import argparse
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.state import State
from src.game.config import load_game_config
from src.ui.pygame_ui import PygameUI
from src.agents.random_agent import RandomAgent
from src.agents.baseline_agent import BaselineAgent


def main():
    """Run human vs AI game with Pygame UI."""
    parser = argparse.ArgumentParser(description="Play Quoridor vs AI with Pygame GUI")
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "small", "tiny"],
        help="Game configuration (default: standard 9x9)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="baseline",
        choices=["random", "baseline"],
        help="AI opponent type (default: baseline)"
    )
    parser.add_argument(
        "--human-player",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which player the human controls (default: 1)"
    )
    args = parser.parse_args()

    # Load game configuration
    config = load_game_config(args.config)
    print(f"Starting {config.board_size}x{config.board_size} Quoridor game")
    print(f"Each player has {config.walls_per_player} walls")
    print(f"You are Player {args.human_player}")
    print(f"Opponent: {args.opponent}")
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

    # Determine which player is AI
    ai_player = 2 if args.human_player == 1 else 1

    # Create AI opponent with player number
    if args.opponent == "random":
        ai_agent = RandomAgent(ai_player)
    else:
        ai_agent = BaselineAgent(ai_player)

    # Create UI
    ui = PygameUI(state)

    # Track if we're waiting for AI move
    ai_move_pending = (state.current_player == ai_player)

    # Define move callback
    def on_move(action_type: str, position: tuple):
        nonlocal state, ai_move_pending

        # Only process if it's human's turn
        if state.current_player != args.human_player:
            return

        # Attempt to make the move
        try:
            new_state = state.make_move(action_type, position)
            state = new_state
            ui.update_state(state)

            # Check for game over
            result = state.check_termination()
            if result is not None:
                if result.winner is not None:
                    winner_name = "You" if result.winner == args.human_player else "AI"
                    print(f"\n=== {winner_name} win! ===")
                else:
                    print(f"\n=== Game ended in a draw: {result.reason} ===")
                return

            # Mark that AI should move next
            ai_move_pending = True

        except ValueError as e:
            print(f"Illegal move: {e}")

    # Custom game loop to handle AI moves
    import pygame

    running = True
    clock = pygame.time.Clock()

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                ui._handle_click(event.pos, on_move)
            elif event.type == pygame.MOUSEMOTION:
                ui._handle_hover(event.pos)
            elif event.type == pygame.KEYDOWN:
                ui._handle_keypress(event.key)

        # Handle AI move if pending
        if ai_move_pending and not state.game_over:
            # Small delay so user can see the move
            time.sleep(0.5)

            # Get AI move
            action_type, position = ai_agent.select_action(state)

            # Make the move
            try:
                state = state.make_move(action_type, position)
                ui.update_state(state)

                # Add to move history
                from src.notation.converter import position_to_notation
                notation = position_to_notation(position[0], position[1], is_wall=(action_type != 'move'))
                if action_type == 'h_wall':
                    notation += 'h'
                elif action_type == 'v_wall':
                    notation += 'v'
                ui.move_history.append(notation)

                # Check for game over
                result = state.check_termination()
                if result is not None:
                    if result.winner is not None:
                        winner_name = "You" if result.winner == args.human_player else "AI"
                        print(f"\n=== {winner_name} win! ===")
                    else:
                        print(f"\n=== Game ended in a draw: {result.reason} ===")

                ai_move_pending = False

            except ValueError as e:
                print(f"AI made illegal move: {e}")
                ai_move_pending = False

        # Render
        ui.render()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("\nGame ended.")
    print(f"Final move count: {state.move_count}")


if __name__ == "__main__":
    main()

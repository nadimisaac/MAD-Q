#!/usr/bin/env python3
"""Human vs Human Quoridor game."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.config import load_game_config
from src.game.state import create_initial_state
from src.agents.human_agent import HumanAgent
from src.utils.visualization import render_board_ascii


def play_game(config_name: str = "standard"):
    """Play a human vs human game.

    Args:
        config_name: Name of game configuration to use
    """
    # Load configuration
    config = load_game_config(config_name)
    print(f"\n{'=' * 60}")
    print(f"QUORIDOR - Human vs Human")
    print(f"{'=' * 60}")
    print(f"\nBoard size: {config.board_size}x{config.board_size}")
    print(f"Walls per player: {config.walls_per_player}")
    print(f"\nPlayer 1 starts at bottom (row 1), goal is top (row {config.board_size})")
    print(f"Player 2 starts at top (row {config.board_size}), goal is bottom (row 1)")
    print(f"\nMove notation:")
    print(f"  - Pawn moves: e5 (column + row)")
    print(f"  - Horizontal walls: e4h (blocks vertical movement)")
    print(f"  - Vertical walls: e4v (blocks horizontal movement)")
    print(f"\n{'=' * 60}\n")

    input("Press Enter to start...")

    # Create initial game state
    state = create_initial_state(config)

    # Create agents
    player1 = HumanAgent(1)
    player2 = HumanAgent(2)
    agents = {1: player1, 2: player2}

    # Game loop
    while not state.game_over:
        # Get current player's agent
        current_agent = agents[state.current_player]

        # Get action from agent
        try:
            action_type, position = current_agent.select_action(state)

            # Execute the move
            state = state.make_move(action_type, position)

        except KeyboardInterrupt:
            print("\n\nGame interrupted by user.")
            return
        except Exception as e:
            print(f"\nError during move: {e}")
            print("Game terminated.")
            return

    # Game over - display final state
    print("\n" + "=" * 60)
    print("GAME OVER!")
    print("=" * 60)
    print(render_board_ascii(state))

    result = state.check_termination()
    if result:
        if result.winner:
            print(f"\n🎉 Player {result.winner} wins!")
            print(f"Reason: {result.reason}")
        else:
            print(f"\nGame ended in a draw")
            print(f"Reason: {result.reason}")
        print(f"Total moves: {result.num_moves}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play Quoridor human vs human")
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "small", "tiny"],
        help="Game configuration to use (default: standard)"
    )

    args = parser.parse_args()

    try:
        play_game(args.config)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
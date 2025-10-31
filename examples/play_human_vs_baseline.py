#!/usr/bin/env python3
"""Human vs Baseline agent Quoridor game."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.config import load_game_config
from src.game.state import create_initial_state
from src.agents.human_agent import HumanAgent
from src.agents.baseline_agent import BaselineAgent
from src.utils.visualization import render_board_ascii
from src.notation.converter import action_to_notation


def play_game(config_name: str = "standard", human_player: int = 1):
    """Play a human vs baseline agent game.

    Args:
        config_name: Name of game configuration to use
        human_player: Which player is human (1 or 2)
    """
    # Load configuration
    config = load_game_config(config_name)
    print(f"\n{'=' * 60}")
    print(f"QUORIDOR - Human vs Baseline Agent")
    print(f"{'=' * 60}")
    print(f"\nBoard size: {config.board_size}x{config.board_size}")
    print(f"Walls per player: {config.walls_per_player}")
    print(f"\nYou are Player {human_player}")
    print(f"Player 1 starts at bottom (row 1), goal is top (row {config.board_size})")
    print(f"Player 2 starts at top (row {config.board_size}), goal is bottom (row 1)")
    print(f"\nBaseline Agent Strategy:")
    print(f"  - 50-50 chance between pawn move or wall placement")
    print(f"  - Pawn moves: Follows shortest path to goal")
    print(f"  - Wall placements: Blocks opponent's next move on their shortest path")
    print(f"\nMove notation:")
    print(f"  - Pawn moves: e5 (column + row)")
    print(f"  - Horizontal walls: e4h (blocks vertical movement)")
    print(f"  - Vertical walls: e4v (blocks horizontal movement)")
    print(f"\n{'=' * 60}\n")

    input("Press Enter to start...")

    # Create initial game state
    state = create_initial_state(config)

    # Create agents
    human = HumanAgent(human_player)
    baseline_agent = BaselineAgent(3 - human_player)  # Other player
    agents = {human_player: human, 3 - human_player: baseline_agent}

    # Game loop
    while not state.game_over:
        # Get current player's agent
        current_agent = agents[state.current_player]

        # Get action from agent
        try:
            action_type, position = current_agent.select_action(state)

            # If baseline agent, show what it chose
            if isinstance(current_agent, BaselineAgent):
                move_str = action_to_notation(action_type, position)
                move_type = "pawn move" if action_type == 'move' else "wall placement"
                print(f"\nBaseline agent (Player {state.current_player}) plays: {move_str} ({move_type})")

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
            if result.winner == human_player:
                print(f"\n🎉 You win!")
            else:
                print(f"\n😞 Baseline agent wins!")
            print(f"Reason: {result.reason}")
        else:
            print(f"\nGame ended in a draw")
            print(f"Reason: {result.reason}")
        print(f"Total moves: {result.num_moves}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play Quoridor human vs baseline agent")
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "small", "tiny"],
        help="Game configuration to use (default: standard)"
    )
    parser.add_argument(
        "--human-player",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which player is human (1 or 2, default: 1)"
    )

    args = parser.parse_args()

    try:
        play_game(args.config, args.human_player)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

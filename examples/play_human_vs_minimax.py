#!/usr/bin/env python3
"""Human vs Minimax agent Quoridor game."""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.config import load_game_config
from src.game.state import create_initial_state
from src.agents.human_agent import HumanAgent
from src.agents.minimax_agent import MinimaxAgent
from src.utils.visualization import render_board_ascii
from src.notation.converter import action_to_notation


def play_game(
    config_name: str = "standard",
    human_player: int = 1,
    depth: int = 3,
    max_wall_moves: Optional[int] = 8,
) -> None:
    """Play a human vs minimax agent game."""
    config = load_game_config(config_name)

    print(f"\n{'=' * 60}")
    print("QUORIDOR - Human vs Minimax Agent")
    print(f"{'=' * 60}")
    print(f"\nBoard size: {config.board_size}x{config.board_size}")
    print(f"Walls per player: {config.walls_per_player}")
    wall_limit_label = max_wall_moves if max_wall_moves is not None else "unbounded"
    print(f"Search depth: {depth} plies")
    print(f"Wall branching limit: {wall_limit_label}")
    print(f"\nYou are Player {human_player}")
    print(f"Player 1 starts at bottom (row 1), goal is top (row {config.board_size})")
    print(f"Player 2 starts at top (row {config.board_size}), goal is bottom (row 1)")
    print("\nMinimax Agent Highlights:")
    print("  - Alpha-beta pruning with move ordering")
    print("  - Heuristic blends path distance, progress, and wall stock")
    print("  - Optional pruning of low-priority wall placements")
    print("\nMove notation:")
    print("  - Pawn moves: e5 (column + row)")
    print("  - Horizontal walls: e4h (blocks vertical movement)")
    print("  - Vertical walls: e4v (blocks horizontal movement)")
    print(f"\n{'=' * 60}\n")

    input("Press Enter to start...")

    state = create_initial_state(config)

    human = HumanAgent(human_player)
    minimax_agent = MinimaxAgent(
        player_number=3 - human_player,
        depth=depth,
        max_wall_moves=max_wall_moves,
    )
    agents = {human_player: human, 3 - human_player: minimax_agent}

    while not state.game_over:
        current_agent = agents[state.current_player]

        try:
            action_type, position = current_agent.select_action(state)

            if isinstance(current_agent, MinimaxAgent):
                move_str = action_to_notation(action_type, position)
                move_type = "pawn move" if action_type == "move" else "wall placement"
                print(f"\nMinimax agent (Player {state.current_player}) plays: {move_str} ({move_type})")

            state = state.make_move(action_type, position)

        except KeyboardInterrupt:
            print("\n\nGame interrupted by user.")
            return
        except Exception as exc:
            print(f"\nError during move: {exc}")
            print("Game terminated.")
            return

    print("\n" + "=" * 60)
    print("GAME OVER!")
    print("=" * 60)
    print(render_board_ascii(state))

    result = state.check_termination()
    if result:
        if result.winner:
            if result.winner == human_player:
                print("\n🎉 You win!")
            else:
                print("\n🤖 Minimax agent wins!")
            print(f"Reason: {result.reason}")
        else:
            print("\nGame ended in a draw")
            print(f"Reason: {result.reason}")
        print(f"Total moves: {result.num_moves}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play Quoridor human vs minimax agent")
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "small", "tiny"],
        help="Game configuration to use (default: standard)",
    )
    parser.add_argument(
        "--human-player",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which player is human (1 or 2, default: 1)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Search depth in plies for the minimax agent (default: 3)",
    )
    parser.add_argument(
        "--max-wall-moves",
        type=int,
        default=8,
        help="Limit on wall placements to explore per node (default: 8, use 0 for no limit)",
    )

    args = parser.parse_args()

    max_walls = None if args.max_wall_moves <= 0 else args.max_wall_moves

    try:
        play_game(
            config_name=args.config,
            human_player=args.human_player,
            depth=args.depth,
            max_wall_moves=max_walls,
        )
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        import traceback

        traceback.print_exc()

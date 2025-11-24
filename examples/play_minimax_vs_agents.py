#!/usr/bin/env python3
"""Automated Minimax agent matches against other Quoridor agents."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.config import load_game_config
from src.game.state import GameResult, create_initial_state
from src.agents.baseline_agent import BaselineAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.notation.converter import action_to_notation
from src.utils.visualization import render_board_ascii

Action = Tuple[str, Tuple[int, int]]

AUTOMATED_OPPONENTS: Dict[str, Tuple[str, type]] = {
    "random": ("Random", RandomAgent),
    "baseline": ("Baseline", BaselineAgent),
}


def main() -> None:
    """Entry point for the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run automated matches for the Minimax agent."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["standard", "small", "tiny"],
        help="Game configuration to use (default: standard)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "baseline", "all"],
        help="Opponent agent to face (default: random, use 'all' to play every automated agent)",
    )
    parser.add_argument(
        "--minimax-player",
        type=int,
        default=1,
        choices=[1, 2],
        help="Player number controlled by the Minimax agent (default: 1)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Search depth in plies for the Minimax agent (default: 3)",
    )
    parser.add_argument(
        "--max-wall-moves",
        type=int,
        default=8,
        help="Limit on wall placements explored per node (default: 8, use 0 for no limit)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(__file__).parent / "match_logs"),
        help="Directory where match logs will be stored (default: examples/match_logs)",
    )
    parser.add_argument(
        "--show-board",
        action="store_true",
        help="Render the board after every move (default: disabled)",
    )

    args = parser.parse_args()
    max_walls = None if args.max_wall_moves <= 0 else args.max_wall_moves
    log_dir = Path(args.log_dir)

    opponent_keys = (
        list(AUTOMATED_OPPONENTS.keys())
        if args.opponent == "all"
        else [args.opponent]
    )

    summary: List[Tuple[str, GameResult]] = []
    for opponent_key in opponent_keys:
        result = play_minimax_match(
            config_name=args.config,
            minimax_player=args.minimax_player,
            opponent_key=opponent_key,
            depth=args.depth,
            max_wall_moves=max_walls,
            log_dir=log_dir,
            show_board=args.show_board,
        )

        if result is None:
            print("\nMatch interrupted. Aborting remaining games.")
            break
        summary.append((opponent_key, result))

    if len(summary) > 1:
        print("\nSummary of automated matches:")
        for opponent_key, result in summary:
            label = AUTOMATED_OPPONENTS[opponent_key][0]
            if result.winner is None:
                outcome = "Draw"
            elif result.winner == args.minimax_player:
                outcome = "Minimax agent wins"
            else:
                outcome = f"{label} agent wins"
            print(
                f"- vs {label}: {outcome} "
                f"(reason: {result.reason}, total moves: {result.num_moves})"
            )


def play_minimax_match(
    config_name: str,
    minimax_player: int,
    opponent_key: str,
    depth: int,
    max_wall_moves: Optional[int],
    log_dir: Path,
    show_board: bool = False,
) -> Optional[GameResult]:
    """Run a single automated match and save its move history."""
    opponent_label, opponent_cls = AUTOMATED_OPPONENTS[opponent_key]
    config = load_game_config(config_name)

    print(f"\n{'=' * 60}")
    print(f"QUORIDOR - Minimax vs {opponent_label} Agent")
    print(f"{'=' * 60}")
    print(f"\nBoard size: {config.board_size}x{config.board_size}")
    print(f"Walls per player: {config.walls_per_player}")
    wall_limit_label = max_wall_moves if max_wall_moves is not None else "unbounded"
    print(f"Minimax search depth: {depth} plies")
    print(f"Wall branching limit: {wall_limit_label}")
    print(f"\nMinimax controls Player {minimax_player}")
    print(f"{opponent_label} agent controls Player {3 - minimax_player}")
    print("\nMove-by-move updates:")
    print(f"{'=' * 60}\n")

    state = create_initial_state(config)
    minimax_agent = MinimaxAgent(
        player_number=minimax_player,
        depth=depth,
        max_wall_moves=max_wall_moves,
    )
    opponent_agent = opponent_cls(3 - minimax_player)
    agents = {minimax_player: minimax_agent, 3 - minimax_player: opponent_agent}
    labels = {
        minimax_player: "Minimax agent",
        3 - minimax_player: f"{opponent_label} agent",
    }
    agent_keys = {
        minimax_player: "minimax",
        3 - minimax_player: opponent_key,
    }

    move_history: List[Dict[str, object]] = []

    while not state.game_over:
        current_player = state.current_player
        current_agent = agents[current_player]

        try:
            decision_start = perf_counter()
            action_type, position = current_agent.select_action(state)
            decision_ms = (perf_counter() - decision_start) * 1000.0
        except KeyboardInterrupt:
            print("\nMatch interrupted by user.")
            return None
        except Exception as exc:  # pragma: no cover - informational for CLI
            print(f"\nError during move: {exc}")
            print("Match terminated.")
            return None

        notation = action_to_notation(action_type, position)
        move_type = describe_move(action_type)
        turn_number = len(move_history) + 1
        print(
            f"Turn {turn_number:03d} - {labels[current_player]} "
            f"(Player {current_player}) plays: {notation} ({move_type}) "
            f"[{decision_ms:.1f} ms]"
        )

        move_history.append(
            {
                "turn": turn_number,
                "player": current_player,
                "agent": labels[current_player],
                "agent_key": agent_keys[current_player],
                "decision_time_ms": round(decision_ms, 3),
                "action_type": action_type,
                "position": [position[0], position[1]],
                "notation": notation,
            }
        )

        state = state.make_move(action_type, position)
        if show_board:
            print(render_board_ascii(state))

    result = state.check_termination()
    final_board = render_board_ascii(state)
    print("\n" + "=" * 60)
    print("GAME OVER!")
    print("=" * 60)
    print(final_board)

    if result:
        if result.winner is None:
            print("\nResult: Draw")
        else:
            winner_label = labels[result.winner]
            print(f"\nResult: {winner_label} wins!")
        print(f"Reason: {result.reason}")
        print(f"Total moves: {result.num_moves}")
    print("\n" + "=" * 60 + "\n")

    log_path = save_match_log(
        config_name=config_name,
        config_info={
            "board_size": config.board_size,
            "walls_per_player": config.walls_per_player,
        },
        minimax_player=minimax_player,
        opponent_key=opponent_key,
        opponent_label=opponent_label,
        depth=depth,
        max_wall_moves=max_wall_moves,
        move_history=move_history,
        result=result,
        final_board=final_board,
        log_dir=log_dir,
    )
    print(f"Match log saved to {log_path}")

    return result


def save_match_log(
    *,
    config_name: str,
    config_info: Dict[str, int],
    minimax_player: int,
    opponent_key: str,
    opponent_label: str,
    depth: int,
    max_wall_moves: Optional[int],
    move_history: List[Dict[str, object]],
    result: Optional[GameResult],
    final_board: str,
    log_dir: Path,
) -> Path:
    """Persist match metadata and move history as JSON."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = (
        f"{timestamp}_config-{config_name}_minimax-{minimax_player}_vs_{opponent_key}.json"
    )
    log_path = log_dir / filename

    log_payload = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "name": config_name,
            **config_info,
        },
        "minimax_player": minimax_player,
        "opponent": {
            "key": opponent_key,
            "label": opponent_label,
        },
        "search_parameters": {
            "depth": depth,
            "max_wall_moves": max_wall_moves,
        },
        "result": {
            "winner": result.winner if result else None,
            "reason": result.reason if result else None,
            "num_moves": result.num_moves if result else len(move_history),
        },
        "moves": move_history,
        "final_board_ascii": final_board,
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        json.dump(log_payload, log_file, indent=2)

    return log_path


def describe_move(action_type: str) -> str:
    """Return a human-readable description for an action type."""
    return "pawn move" if action_type == "move" else "wall placement"


if __name__ == "__main__":
    main()

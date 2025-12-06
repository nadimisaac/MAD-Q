"""Minimax agent with alpha-beta pruning for Quoridor."""

from __future__ import annotations

import math
import random
import numpy as np
from typing import List, Optional, Tuple

from .base_agent import BaseAgent
from ..game import State
from ..pathfinding.astar import astar_find_path

Action = Tuple[str, Tuple[int, int]]


class MinimaxAgent(BaseAgent):
    """Game-tree search agent that uses minimax with alpha-beta pruning."""

    def __init__(
        self,
        player_number: int,
        depth: int = 2,
        max_wall_moves: Optional[int] = 8,
        path_weight: float = 12.0,
        progress_weight: float = 1.5,
        wall_weight: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            player_number: Controlled player (1 or 2).
            depth: Search depth in plies (must be >= 1).
            max_wall_moves: Optional cap on how many wall placements to explore per node.
            path_weight: Weight for shortest-path difference heuristic.
            progress_weight: Weight for row-progress heuristic.
            wall_weight: Weight for wall-count advantage heuristic.
            seed: Random seed used for deterministic tie-breaking.
        """
        if depth < 1:
            raise ValueError("depth must be >= 1")

        super().__init__(player_number)
        self.max_depth = depth
        self.max_wall_moves = max_wall_moves
        self.path_weight = path_weight
        self.progress_weight = progress_weight
        self.wall_weight = wall_weight

        self._win_score = 1_000_000.0
        self.rng = random.Random(seed)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def select_action(self, game_state: State) -> Action:
        """Choose an action for the current state."""
        legal_moves = list(game_state.get_legal_moves())
        if not legal_moves:
            raise ValueError("No legal actions available")

        # Early exit if only one option.
        if len(legal_moves) == 1:
            return legal_moves[0]

        value, best_move = self._search(game_state, self.max_depth, -math.inf, math.inf)
        if best_move is None:
            # Fallback to deterministic but reproducible choice.
            legal_moves.sort()
            return legal_moves[0]
        return best_move

    # --------------------------------------------------------------------- #
    # Feature Extractor for TD Learning Eval Function
    # --------------------------------------------------------------------- #

    def extract_td_features(self, state: State) -> np.ndarray:
        opponent = state.get_opponent(self.player_number)
        board_size = state.config.board_size
        walls_per_player = state.config.walls_per_player

        my_path = self._shortest_path_length(state, self.player_number)
        opp_path = self._shortest_path_length(state, opponent)
        path_score = opp_path - my_path

        progress_score = self._progress_to_goal(state, self.player_number) - self._progress_to_goal(state, opponent)
        wall_score = state.walls_remaining[self.player_number] - state.walls_remaining[opponent]
        
        # normalize our eval function features
        normalized_path_score = path_score / board_size
        normalized_progress_score = progress_score / board_size
        normalized_wall_score = wall_score / walls_per_player if walls_per_player > 0 else 0.0
        
        td_features = [normalized_path_score, normalized_progress_score, normalized_wall_score]

        return np.array(td_features, dtype=np.float64)


        

    # --------------------------------------------------------------------- #
    # Core Minimax Search
    # --------------------------------------------------------------------- #

    def _search(
        self,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Tuple[float, Optional[Action]]:
        if depth == 0 or state.game_over:
            return self._evaluate_state(state, depth), None

        moves = self._get_candidate_moves(state)
        if not moves:
            return self._evaluate_state(state, depth), None

        maximizing = state.current_player == self.player_number

        if maximizing:
            best_score = -math.inf
            best_action: Optional[Action] = None

            for move in moves:
                child_state = state.make_move(*move)
                score, _ = self._search(child_state, depth - 1, alpha, beta)
                if score > best_score or (score == best_score and self._prefer_move(move, best_action)):
                    best_score = score
                    best_action = move
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score, best_action

        # Minimizing branch (opponent to move)
        best_score = math.inf
        best_action = None

        for move in moves:
            child_state = state.make_move(*move)
            score, _ = self._search(child_state, depth - 1, alpha, beta)
            if score < best_score or (score == best_score and self._prefer_move(move, best_action)):
                best_score = score
                best_action = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break

        return best_score, best_action

    # --------------------------------------------------------------------- #
    # Evaluation Heuristics
    # --------------------------------------------------------------------- #

    def _evaluate_state(self, state: State, depth_remaining: int) -> float:
        if state.game_over:
            if state.winner == self.player_number:
                return self._win_score + depth_remaining
            if state.winner is None:
                return state.config.draw_penalty
            return -self._win_score - depth_remaining

        opponent = state.get_opponent(self.player_number)

        my_path = self._shortest_path_length(state, self.player_number)
        opp_path = self._shortest_path_length(state, opponent)
        path_score = opp_path - my_path

        progress_score = self._progress_to_goal(state, self.player_number) - self._progress_to_goal(state, opponent)

        wall_score = state.walls_remaining[self.player_number] - state.walls_remaining[opponent]

        return (
            self.path_weight * path_score
            + self.progress_weight * progress_score
            + self.wall_weight * wall_score
        )

    def _shortest_path_length(self, state: State, player: int) -> int:
        position = state.get_pawn_position(player).to_tuple()
        goal_row = state.config.board_size - 1 if player == 1 else 0

        _, distance = astar_find_path(
            position,
            goal_row,
            state.config.board_size,
            state.h_walls,
            state.v_walls,
        )

        if distance == -1:
            # Fallback: treat impossible paths as very long to discourage them.
            return state.config.board_size * 4

        return distance

    def _progress_to_goal(self, state: State, player: int) -> int:
        pos = state.get_pawn_position(player)
        if player == 1:
            return pos.row
        return state.config.board_size - 1 - pos.row

    # --------------------------------------------------------------------- #
    # Move Ordering Utilities
    # --------------------------------------------------------------------- #

    def _get_candidate_moves(self, state: State) -> List[Action]:
        moves = list(state.get_legal_moves())
        if not moves:
            return []

        self.rng.shuffle(moves)  # Avoid deterministic ordering before sorting.

        moving_player = state.current_player
        pawn_moves = [move for move in moves if move[0] == "move"]
        wall_moves = [move for move in moves if move[0] != "move"]

        pawn_moves.sort(key=lambda m: self._pawn_move_priority(moving_player, m[1], state))
        wall_moves.sort(key=lambda m: self._wall_move_priority(moving_player, m, state))

        if self.max_wall_moves is not None and len(wall_moves) > self.max_wall_moves:
            wall_moves = wall_moves[: self.max_wall_moves]

        return pawn_moves + wall_moves

    def _pawn_move_priority(self, player: int, destination: Tuple[int, int], state: State) -> float:
        goal_row = state.config.board_size - 1 if player == 1 else 0
        return abs(goal_row - destination[0])

    def _wall_move_priority(self, player: int, move: Action, state: State) -> float:
        opponent = state.get_opponent(player)
        opp_row, opp_col = state.get_pawn_position(opponent).to_tuple()
        (wall_row, wall_col) = move[1]

        center_row = wall_row + 0.5
        center_col = wall_col + 0.5

        return abs(center_row - opp_row) + abs(center_col - opp_col)

    def _prefer_move(self, candidate: Action, current_best: Optional[Action]) -> bool:
        """Tie-breaker used to maintain deterministic choices."""
        if current_best is None:
            return True

        # Prefer pawn moves over wall placements in ties.
        if candidate[0] == "move" and current_best[0] != "move":
            return True
        if candidate[0] != "move" and current_best[0] == "move":
            return False

        # Fall back to lexicographic order for reproducibility.
        return candidate < current_best

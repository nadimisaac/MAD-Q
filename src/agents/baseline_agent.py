"""Baseline agent with path-based strategy."""

import random
from typing import Tuple, List, Optional
from .base_agent import BaseAgent


class BaselineAgent(BaseAgent):
    """Agent that uses shortest path information for decision making.

    Selection strategy:
    1. First, randomly decide between pawn move or wall placement (50-50 chance)
    2. If pawn move: Move in the direction of the shortest path to goal
    3. If wall placement: Block opponent's next move on their shortest path
    4. Falls back to random selection if strategy fails

    This provides a simple baseline that uses pathfinding but with predictable behavior.
    """

    def __init__(self, player_number: int, seed: int = None):
        """Initialize baseline agent.

        Args:
            player_number: Player number (1 or 2)
            seed: Random seed for reproducibility
        """
        super().__init__(player_number)
        self.rng = random.Random(seed)

    def select_action(self, game_state) -> Tuple[str, Tuple[int, int]]:
        """Select an action using path-based strategy.

        Args:
            game_state: Current GameState object

        Returns:
            Selected (action_type, position) tuple

        Raises:
            ValueError: If no legal actions available
        """
        from ..pathfinding.astar import astar_find_path

        # Get all legal moves once
        legal_moves = game_state.get_legal_moves()

        # Separate by type
        pawn_moves = [(act, pos) for act, pos in legal_moves if act == 'move']
        wall_moves = [(act, pos) for act, pos in legal_moves if act in ('h_wall', 'v_wall')]

        # Check if we have any legal moves at all
        if not pawn_moves and not wall_moves:
            raise ValueError("No legal actions available")

        # If only one type available, choose from it
        if not wall_moves:
            return self._select_best_pawn_move(game_state, pawn_moves)
        if not pawn_moves:
            return self._select_best_wall_move(game_state, wall_moves)

        # Both types available - do 50-50 coin flip
        if self.rng.random() < 0.5:
            return self._select_best_pawn_move(game_state, pawn_moves)
        else:
            return self._select_best_wall_move(game_state, wall_moves)

    def _select_best_pawn_move(
        self,
        game_state,
        pawn_moves: List[Tuple[str, Tuple[int, int]]]
    ) -> Tuple[str, Tuple[int, int]]:
        """Select the best pawn move based on shortest path.

        Strategy: Move toward the goal following the shortest path.

        Args:
            game_state: Current GameState object
            pawn_moves: List of legal pawn moves

        Returns:
            Best pawn move or random if strategy fails
        """
        from ..pathfinding.astar import astar_find_path

        # Get current position and goal
        current_pos = game_state.get_pawn_position(self.player_number)
        goal_row = game_state.config.board_size - 1 if self.player_number == 1 else 0

        # Find shortest path to goal
        path, distance = astar_find_path(
            current_pos.to_tuple(),
            goal_row,
            game_state.config.board_size,
            game_state.h_walls,
            game_state.v_walls
        )

        # If we have a path and it has at least 2 positions, move to next position
        if path and len(path) >= 2:
            next_pos = path[1]  # path[0] is current position, path[1] is next

            # Find this move in our legal moves
            for action_type, pos in pawn_moves:
                if pos == next_pos:
                    return (action_type, pos)

        # Fallback: choose random pawn move
        return self.rng.choice(pawn_moves)

    def _select_best_wall_move(
        self,
        game_state,
        wall_moves: List[Tuple[str, Tuple[int, int]]]
    ) -> Tuple[str, Tuple[int, int]]:
        """Select the best wall move to block opponent.

        Strategy: Block the opponent's next move on their shortest path.

        Args:
            game_state: Current GameState object
            wall_moves: List of legal wall moves

        Returns:
            Best wall move or random if strategy fails
        """
        from ..pathfinding.astar import astar_find_path

        # Get opponent info
        opponent = game_state.get_opponent(self.player_number)
        opponent_pos = game_state.get_pawn_position(opponent)
        opponent_goal = game_state.config.board_size - 1 if opponent == 1 else 0

        # Find opponent's shortest path
        path, distance = astar_find_path(
            opponent_pos.to_tuple(),
            opponent_goal,
            game_state.config.board_size,
            game_state.h_walls,
            game_state.v_walls
        )

        # If opponent has a path with at least 2 positions, try to block their next move
        if path and len(path) >= 2:
            opp_current = path[0]
            opp_next = path[1]

            # Find walls that would block movement from opp_current to opp_next
            blocking_walls = self._find_blocking_walls(opp_current, opp_next, wall_moves)

            if blocking_walls:
                return self.rng.choice(blocking_walls)

        # Fallback: choose random wall move
        return self.rng.choice(wall_moves)

    def _find_blocking_walls(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        wall_moves: List[Tuple[str, Tuple[int, int]]]
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """Find walls that would block movement between two positions.

        Args:
            from_pos: Starting position (row, col)
            to_pos: Ending position (row, col)
            wall_moves: List of legal wall moves

        Returns:
            List of wall moves that would block this movement
        """
        blocking = []
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # Determine direction of movement
        row_diff = to_row - from_row
        col_diff = to_col - from_col

        for action_type, wall_pos in wall_moves:
            wall_row, wall_col = wall_pos

            # Check if this wall would block the movement
            if action_type == 'h_wall':
                # Horizontal wall blocks vertical movement
                # Wall at (r, c) blocks movement between rows r and r+1
                # for columns c and c+1
                if row_diff != 0:  # Vertical movement
                    # Check if wall is between the two rows
                    min_row = min(from_row, to_row)
                    if wall_row == min_row:
                        # Check if wall covers the column
                        if wall_col == from_col or wall_col == from_col - 1:
                            blocking.append((action_type, wall_pos))

            elif action_type == 'v_wall':
                # Vertical wall blocks horizontal movement
                # Wall at (r, c) blocks movement between cols c and c+1
                # for rows r and r+1
                if col_diff != 0:  # Horizontal movement
                    # Check if wall is between the two columns
                    min_col = min(from_col, to_col)
                    if wall_col == min_col:
                        # Check if wall covers the row
                        if wall_row == from_row or wall_row == from_row - 1:
                            blocking.append((action_type, wall_pos))

        return blocking

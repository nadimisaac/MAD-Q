"""Action space encoding and decoding for Quoridor RL.

This module handles the conversion between human-readable actions and
integer action indices for RL training.

Action space structure:
- Pawn moves: 0-127 (9x9 grid = 81 possible positions, but fewer are actually reachable)
- Horizontal walls: 128-191 (8x8 grid of possible wall positions)
- Vertical walls: 192-255 (8x8 grid of possible wall positions)

Total action space: 256 actions for a 9x9 board
"""

from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class Action:
    """Represents a Quoridor action.

    Attributes:
        action_type: Type of action ('move', 'wall_h', 'wall_v')
        position: For moves: (row, col) destination
                 For walls: (row, col) top-left corner of wall
    """
    action_type: str
    position: Tuple[int, int]


class ActionSpace:
    """Handles encoding/decoding of actions for RL.

    Maps between Action objects and integer action indices.
    """

    def __init__(self, board_size: int = 9):
        """Initialize action space for given board size.

        Args:
            board_size: Size of the Quoridor board (default 9)
        """
        self.board_size = board_size
        self.wall_grid_size = board_size - 1

        # Calculate action space boundaries
        self.num_move_actions = board_size * board_size
        self.num_wall_actions = self.wall_grid_size * self.wall_grid_size
        self.total_actions = self.num_move_actions + 2 * self.num_wall_actions

        # Action index boundaries
        self.move_start = 0
        self.wall_h_start = self.num_move_actions
        self.wall_v_start = self.wall_h_start + self.num_wall_actions

    def encode_action(self, action: Action) -> int:
        """Convert Action object to integer action index.

        Args:
            action: Action object to encode

        Returns:
            Integer action index

        Raises:
            ValueError: If action type is invalid or position is out of bounds
        """
        raise NotImplementedError()

    def decode_action(self, action_idx: int) -> Action:
        """Convert integer action index to Action object.

        Args:
            action_idx: Integer action index

        Returns:
            Action object

        Raises:
            ValueError: If action index is out of bounds
        """
        raise NotImplementedError()

    def get_legal_action_mask(
        self,
        legal_moves: List[Tuple[int, int]],
        legal_walls_h: List[Tuple[int, int]],
        legal_walls_v: List[Tuple[int, int]]
    ) -> List[bool]:
        """Create a boolean mask of legal actions.

        Args:
            legal_moves: List of legal pawn move destinations
            legal_walls_h: List of legal horizontal wall positions
            legal_walls_v: List of legal vertical wall positions

        Returns:
            Boolean list of length total_actions, True for legal actions
        """
        raise NotImplementedError()

    def action_to_string(self, action: Action) -> str:
        """Convert Action to human-readable string.

        Args:
            action: Action object

        Returns:
            String representation (e.g., "move to (4,5)", "h-wall at (3,2)")
        """
        raise NotImplementedError()


def create_action_lookup_tables(board_size: int = 9) -> Tuple[Dict, Dict]:
    """Pre-compute lookup tables for fast encoding/decoding.

    Args:
        board_size: Size of the Quoridor board

    Returns:
        Tuple of (action_to_idx, idx_to_action) dictionaries
    """
    raise NotImplementedError()

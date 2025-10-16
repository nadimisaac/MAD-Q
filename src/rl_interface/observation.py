"""Observation encoding for Quoridor neural networks.

Converts State objects into tensor representations suitable for neural network input.
Uses spatial channels for board state (with temporal history) and scalar features for metadata.
"""

from typing import Tuple, List
import torch
from torch import Tensor


class Observation:
    """Encodes State into neural network input tensors (observations).

    Converts board state into:
    1. Spatial features: (C, H, W) tensor with temporal history
    2. Scalar features: (S,) tensor with game metadata

    Spatial channels (for 9x9 board with history_length=4):
    For each of the last 4 game states:
    - Channel 0-3: Current player pawn position (4 temporal frames)
    - Channel 4-7: Opponent pawn position (4 temporal frames)
    - Channel 8-11: Horizontal walls (4 temporal frames)
    - Channel 12-15: Vertical walls (4 temporal frames)
    Total: 16 spatial channels

    Scalar features:
    - Current player (0 or 1)
    - Current player walls remaining (normalized 0-1)
    - Opponent walls remaining (normalized 0-1)
    - Move number (normalized 0-1)
    - Distance to goal for current player (normalized)
    - Distance to goal for opponent (normalized)
    """

    def __init__(
        self,
        board_size: int = 9,
        max_walls: int = 10,
        history_length: int = 4,
        device: str = 'cpu'
    ):
        """Initialize observation encoder.

        Args:
            board_size: Size of the Quoridor board
            max_walls: Maximum walls per player
            history_length: Number of past states to include (temporal depth)
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.board_size = board_size
        self.max_walls = max_walls
        self.history_length = history_length
        self.device = device
        self.num_base_channels = 4  # player1, player2, walls_h, walls_v
        self.num_spatial_channels = self.num_base_channels * history_length
        self.num_scalar_features = 6

    def encode(
        self,
        state,
        state_history: List = None
    ) -> Tuple[Tensor, Tensor]:
        """Encode State (with history) into spatial and scalar tensors.

        Args:
            state: Current State object to encode
            state_history: List of past State objects (most recent last)
                          If None or shorter than history_length, will pad with zeros

        Returns:
            Tuple of (spatial_features, scalar_features)
            - spatial_features: (C, H, W) torch tensor where C = num_base_channels * history_length
            - scalar_features: (S,) torch tensor
        """
        raise NotImplementedError()

    def encode_spatial_features(
        self,
        state,
        state_history: List = None
    ) -> Tensor:
        """Encode spatial board features with temporal history.

        Args:
            state: Current State object
            state_history: List of past State objects

        Returns:
            (C, H, W) torch tensor of spatial features with temporal dimension
        """
        raise NotImplementedError()

    def encode_single_state_spatial(self, state) -> Tensor:
        """Encode a single game state's spatial features (4 base channels).

        Args:
            state: State object

        Returns:
            (4, H, W) torch tensor
        """
        raise NotImplementedError()

    def encode_scalar_features(self, state) -> Tensor:
        """Encode scalar game metadata.

        Args:
            state: State object

        Returns:
            (S,) torch tensor of scalar features
        """
        raise NotImplementedError()

    def encode_pawn_position(
        self,
        position: Tuple[int, int],
        board_size: int
    ) -> Tensor:
        """Encode pawn position as binary spatial map.

        Args:
            position: (row, col) position
            board_size: Size of board

        Returns:
            (H, W) binary tensor with 1 at position
        """
        raise NotImplementedError()

    def encode_walls(self, state, orientation: str) -> Tensor:
        """Encode wall positions as binary spatial map.

        Args:
            state: State object
            orientation: 'h' for horizontal, 'v' for vertical

        Returns:
            (H, W) binary tensor with 1s where walls exist
        """
        raise NotImplementedError()

    def compute_distance_to_goal(self, state, player: int) -> float:
        """Compute shortest path distance to goal for a player.

        Uses A* pathfinding to compute actual distance considering walls.

        Args:
            state: State object
            player: Player number (1 or 2)

        Returns:
            Normalized distance value in [0, 1]
        """
        raise NotImplementedError()

    def normalize_walls_remaining(self, walls_remaining: int) -> float:
        """Normalize wall count to [0, 1].

        Args:
            walls_remaining: Number of walls remaining

        Returns:
            Normalized value in [0, 1]
        """
        raise NotImplementedError()

    def normalize_move_number(self, move_number: int, max_moves: int = 200) -> float:
        """Normalize move number to [0, 1].

        Args:
            move_number: Current move number
            max_moves: Expected maximum moves per game

        Returns:
            Normalized value in [0, 1]
        """
        raise NotImplementedError()


def batch_encode_states(
    states: List,
    state_histories: List[List],
    observation: Observation
) -> Tuple[Tensor, Tensor]:
    """Encode a batch of game states with their histories.

    Args:
        states: List of current State objects
        state_histories: List of state history lists (one per state)
        observation: Observation encoder instance

    Returns:
        Tuple of (spatial_batch, scalar_batch)
        - spatial_batch: (N, C, H, W) torch tensor
        - scalar_batch: (N, S) torch tensor
    """
    raise NotImplementedError()

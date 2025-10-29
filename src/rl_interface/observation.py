"""Observation encoding for Quoridor neural networks.

Converts State objects into fully spatial tensor representations suitable for CNN input.
Uses 8 spatial planes (no scalar features, no temporal history) with canonicalization.
"""

from typing import Tuple
import torch
from torch import Tensor
from ..pathfinding.astar import astar_find_path

class Observation:
    """Encodes State into neural network input tensors (9-plane spatial representation).
    
    The 9 planes (for 9x9 board) are:
    - Plane 0: Current player pawn position (binary, one-hot)
    - Plane 1: Opponent pawn position (binary, one-hot)
    - Plane 2: Current player walls remaining (constant, normalized [0,1])
    - Plane 3: Opponent walls remaining (constant, normalized [0,1])
    - Plane 4: Horizontal walls (binary - both cells marked per wall)
    - Plane 5: Vertical walls (binary - both cells marked per wall)
    - Plane 6: Current player distance to goal (constant, normalized A* distance, broadcasted)
    - Plane 7: Opponent distance to goal (constant, normalized A* distance, broadcasted)
    - Plane 8: Turn Indicator (binary, 1 represents P1, 0 represents P2)
    
    TENTATIVE: Canonicalization (Do we want a turn indicator and canonicalization ...): 
    - Always encodes from current player's perspective
    - Current player at bottom, moving toward top
    - This way the network doesn't have the additional complexity in its learning 
    - Network should not care whose turn it is and how they will play based on the turn
    - Just learn how to get to the top row when you are the one playing
    - AlphaZero paper canonicalizes the state I believe ? but we are not sure if we want to do this ..
    """

    def __init__(
        self,
        board_size: int = 9,
        max_walls: int = 10,
        device: str = 'cpu'
    ):
        """Initialize observation encoder.

        Args:
            board_size: Size of the Quoridor board
            max_walls: Maximum walls per player (for normalization)
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.board_size = board_size
        self.max_walls = max_walls
        self.device = device
        self.num_planes = 9

    def encode(self, state) -> Tensor:
        """Encode State into spatial tensor with canonicalization(?).

        Args:
            state: Current State object to encode

        Returns:
            (9, H, W) torch tensor of spatial features
        """
        # Canonicalize: if Player 2's turn, flip perspective
        # This makes the current player always appear at the bottom
        # TODO: Tentative: Decide if we want to canonicalize the state?
        # Would need to implement the flip_perspective method in the State class

        # canonical_state = state if state.current_player == 1 else state.flip_perspective()
        
        # Initialize output tensor
        observation = torch.zeros(
            self.num_planes, 
            self.board_size, 
            self.board_size,
            dtype=torch.float32,
            device=self.device
        )
        
        # Plane 0: Current player pawn position (one-hot)
        observation[0] = self._encode_pawn_position(
            state.player1_pos.to_tuple()
        )
        
        # Plane 1: Opponent pawn position (one-hot)
        observation[1] = self._encode_pawn_position(
            state.player2_pos.to_tuple()
        )
        
        # Plane 2: Current player walls remaining (constant)
        observation[2] = self._encode_walls_remaining(
            state.walls_remaining[1]
        )
        
        # Plane 3: Opponent walls remaining (constant)
        observation[3] = self._encode_walls_remaining(
            state.walls_remaining[2]
        )
        
        # Plane 4: Horizontal walls (binary with both cells marked)
        observation[4] = self._encode_walls(state, 'h')
        
        # Plane 5: Vertical walls (binary with both cells marked)
        observation[5] = self._encode_walls(state, 'v')
        
        # Plane 6: Current player distance to goal (constant)
        observation[6] = self._encode_distance_to_goal(state, player=1)
        
        # Plane 7: Opponent distance to goal (constant)
        observation[7] = self._encode_distance_to_goal(state, player=2)

        # Plane 8: Turn indicator (binary, 1 represents P1, 0 represents P2)
        observation[8] = self._encode_turn_indicator(state.current_player)
        
        return observation

    def _encode_pawn_position(self, position: Tuple[int, int]) -> Tensor:
        """Encode pawn position as binary spatial map (one-hot).

        Args:
            position: (row, col) position

        Returns:
            (H, W) binary tensor with 1 at position, 0 elsewhere
        """
        plane = torch.zeros(
            self.board_size, 
            self.board_size,
            dtype=torch.float32,
            device=self.device
        )
        row, col = position
        plane[row, col] = 1.0
        return plane

    def _encode_walls_remaining(self, walls_remaining: int) -> Tensor:
        """Encode walls remaining as constant plane.

        Args:
            walls_remaining: Number of walls remaining

        Returns:
            (H, W) constant tensor with normalized value [0, 1]
        """
        normalized_value = walls_remaining / self.max_walls
        return torch.full(
            (self.board_size, self.board_size),
            normalized_value,
            dtype=torch.float32,
            device=self.device
        )

    def _encode_walls(self, state, orientation: str) -> Tensor:
        """Encode wall positions as binary spatial map.

        Each wall spans 2 cells - both cells are marked with 1.
        
        Args:
            state: State object
            orientation: 'h' for horizontal, 'v' for vertical

        Returns:
            (H, W) binary tensor with 1s where walls exist
        """
        plane = torch.zeros(
            self.board_size,
            self.board_size,
            dtype=torch.float32,
            device=self.device
        )
        
        walls = state.h_walls if orientation == 'h' else state.v_walls
        
        for row, col in walls:
            if orientation == 'h':
                # Horizontal wall spans 2 columns
                if row < self.board_size and col < self.board_size:
                    plane[row, col] = 1.0
                if row < self.board_size and col + 1 < self.board_size:
                    plane[row, col + 1] = 1.0
            else:  # vertical
                # Vertical wall spans 2 rows
                if row < self.board_size and col < self.board_size:
                    plane[row, col] = 1.0
                if row + 1 < self.board_size and col < self.board_size:
                    plane[row + 1, col] = 1.0
        
        return plane

    def _encode_distance_to_goal(self, state, player: int) -> Tensor:
        """Encode shortest path distance to goal as constant plane.

        Uses A* pathfinding to compute actual distance considering walls.

        Args:
            state: State object
            player: Player number (1 or 2)

        Returns:
            (H, W) constant tensor with normalized distance [0, 1]
        """
        
        # Get player position and goal
        if player == 1:
            start_pos = state.player1_pos.to_tuple()
            goal_row = self.board_size - 1  # Player 1 moves toward top
        else:
            start_pos = state.player2_pos.to_tuple()
            goal_row = 0  # Player 2 moves toward bottom
        
        # Compute A* distance
        distance = astar_find_path(
            start_pos,
            goal_row,
            self.board_size,
            state.h_walls,
            state.v_walls
        )

        # If no path exists, our A* function returns -1
        # We need to handle this in our encoded state input to the network
        # Ideally, we want a -1 distance to be normalized to 1
        # This will represent the longest possible normalized distance

        max_distance = self.board_size ** 2
        if distance == -1:
            normalized_distance = 1.0
        else:
            normalized_distance = min(distance / max_distance, 1.0)
        
        
        return torch.full(
            (self.board_size, self.board_size),
            normalized_distance,
            dtype=torch.float32,
            device=self.device
        )

    def _encode_turn_indicator(self, player: int) -> Tensor:
        """Encode turn indicator as binary plane.

        Args:
            player: Player number (1 or 2)

        Returns:
            (H, W) binary tensor with 1 at position, 0 elsewhere
        """
        return torch.full(
            (self.board_size, self.board_size),
            1 if player == 1 else 0,
            dtype=torch.float32,
            device=self.device
        )

def batch_encode_states(
    states: list,
    observation: Observation
) -> Tensor:
    """Encode a batch of game states.

    Args:
        states: List of State objects
        observation: Observation encoder instance

    Returns:
        (N, 9, H, W) torch tensor batch
    """
    encoded = [observation.encode(state) for state in states]
    return torch.stack(encoded, dim=0)

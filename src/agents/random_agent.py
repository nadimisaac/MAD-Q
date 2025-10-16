"""Random agent with 50-50 move type selection strategy."""

import random
from typing import Tuple
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that randomly selects between move types, then picks randomly within that type.

    Selection strategy:
    1. First, randomly decide between pawn move or wall placement (50-50 chance)
    2. Then, randomly select from available legal moves of that type
    3. If chosen type has no legal moves, selects from the other type

    Note: This creates a bias toward move types rather than uniform selection over
    all legal actions. For example, if there are 3 pawn moves and 30 wall placements,
    each pawn move has ~16.67% probability while each wall has ~1.67% probability.

    Useful as a baseline for evaluation.
    """

    def __init__(self, player_number: int, seed: int = None):
        """Initialize random agent.

        Args:
            player_number: Player number (1 or 2)
            seed: Random seed for reproducibility
        """
        super().__init__(player_number)
        self.rng = random.Random(seed)

    def select_action(self, game_state) -> Tuple[str, Tuple[int, int]]:
        """Select a random legal action.

        First randomly decides between pawn move or wall placement (50-50).
        Then randomly selects from the available moves of that type.
        Falls back to the other type if the chosen type has no legal moves.

        Args:
            game_state: Current GameState object

        Returns:
            Randomly selected (action_type, position) tuple

        Raises:
            ValueError: If no legal actions available
        """
        # Get all legal moves once (performance optimization)
        legal_moves = game_state.get_legal_moves()

        # Separate by type
        pawn_moves = [(act, pos) for act, pos in legal_moves if act == 'move']
        wall_moves = [(act, pos) for act, pos in legal_moves if act in ('h_wall', 'v_wall')]

        # Check if we have any legal moves at all
        if not pawn_moves and not wall_moves:
            raise ValueError("No legal actions available")

        # If only one type available, choose from it
        if not wall_moves:
            return self.rng.choice(pawn_moves)
        if not pawn_moves:
            return self.rng.choice(wall_moves)

        # Both types available - do 50-50 coin flip
        if self.rng.random() < 0.5:
            return self.rng.choice(pawn_moves)
        else:
            return self.rng.choice(wall_moves)

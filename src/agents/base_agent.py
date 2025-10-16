"""Base agent interface for Quoridor."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseAgent(ABC):
    """Abstract base class for Quoridor agents.

    All agents must implement the select_action method.
    """

    def __init__(self, player_number: int):
        """Initialize agent.

        Args:
            player_number: Player number (1 or 2)
        """
        self.player_number = player_number

    @abstractmethod
    def select_action(self, game_state) -> Tuple[str, Tuple[int, int]]:
        """Select an action given the current game state.

        Args:
            game_state: Current GameState object

        Returns:
            Tuple of (action_type, position)
            - action_type: 'move', 'wall_h', or 'wall_v'
            - position: (row, col) tuple

        Raises:
            ValueError: If no legal actions available
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset agent state (for stateful agents).

        Optional method - stateless agents can skip implementation.
        """
        pass

    def game_over(self, result) -> None:
        """Notify agent that game has ended.

        Args:
            result: GameResult object

        Optional method for learning agents to update from game outcome.
        """
        pass

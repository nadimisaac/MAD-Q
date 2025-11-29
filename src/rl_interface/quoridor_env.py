"""Gymnasium environment for Quoridor.

Provides a standard Gymnasium interface for training RL agents.
Uses the unified State representation and Observation encoder.
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class QuoridorEnv(gym.Env):
    """Gymnasium environment for Quoridor.

    Observation space:
        - spatial: Box (C, H, W) where C = num_base_channels * history_length
        - scalar: Box (S,) where S = num_scalar_features

    Action space:
        - Discrete(N) where N = board_size^2 + 2 * (board_size-1)^2
    """

    metadata = {"render_modes": ["human", "ascii", "rgb_array"], "render_fps": 2}

    def __init__(
        self,
        config_name: str = "standard",
        render_mode: Optional[str] = None,
        opponent=None
    ):
        """Initialize Quoridor environment.

        Args:
            config_name: Name of game configuration to use
            render_mode: Rendering mode ('human', 'ascii', 'rgb_array', or None)
            opponent: Optional opponent agent (if None, requires external control)
        """
        raise NotImplementedError()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
            - observation: Dict with 'spatial' and 'scalar' keys
            - info: Dict with additional information
        """
        raise NotImplementedError()

    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Integer action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            - observation: Dict with 'spatial' and 'scalar' keys
            - reward: Reward for this step
            - terminated: Whether episode ended naturally (win/loss)
            - truncated: Whether episode was truncated (max steps)
            - info: Dict with additional information
        """
        raise NotImplementedError()

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode='rgb_array', otherwise None
        """
        raise NotImplementedError()

    def close(self) -> None:
        """Clean up environment resources."""
        raise NotImplementedError()

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation from game state.

        Uses Observation encoder to convert State → tensor observations.

        Returns:
            Dict with 'spatial' and 'scalar' observation components
        """
        raise NotImplementedError()

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state.

        Returns:
            Info dict with legal actions, move number, etc.
        """
        raise NotImplementedError()

    def _calculate_reward(
        self,
        previous_state,
        action: int,
        current_state
    ) -> float:
        """Calculate reward for a transition.

        Args:
            previous_state: State before action
            action: Action taken
            current_state: State after action

        Returns:
            Reward value
        """
        raise NotImplementedError()

    def get_legal_actions_mask(self) -> np.ndarray:
        """Get boolean mask of legal actions.

        Returns:
            Boolean array of length action_space.n
        """
        raise NotImplementedError()


class QuoridorSelfPlayEnv(QuoridorEnv):
    """Environment for self-play training.

    Automatically alternates between two agents/policies.
    """

    def __init__(
        self,
        config_name: str = "standard",
        render_mode: Optional[str] = None
    ):
        """Initialize self-play environment.

        Args:
            config_name: Name of game configuration to use
            render_mode: Rendering mode
        """
        raise NotImplementedError()

    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step with automatic opponent response.

        Args:
            action: Integer action index for current player

        Returns:
            Observation after both players have moved (if game continues)
        """
        raise NotImplementedError()

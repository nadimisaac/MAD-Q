"""Game logic module for Quoridor."""

from .state import State, Position, GameResult, create_initial_state
from .config import GameConfig, load_game_config, load_all_configs

# Note: moves.py functions are internal helpers, not exported
# Use State.get_legal_moves() and related methods instead

__all__ = [
    'State',
    'Position',
    'GameResult',
    'create_initial_state',
    'GameConfig',
    'load_game_config',
    'load_all_configs',
]

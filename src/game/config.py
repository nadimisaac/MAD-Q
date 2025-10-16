"""Game configuration loader and data structures."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import yaml


@dataclass
class GameConfig:
    """Configuration for a Quoridor game variant.

    Attributes:
        board_size: Size of the board (e.g., 9 for 9x9)
        walls_per_player: Number of walls each player starts with
        max_moves: Maximum number of moves before draw
        no_progress_limit: Max moves without wall placement before draw
        history_length: Number of past states to include in encoding
        draw_penalty: Reward penalty for draws (typically negative)
    """
    board_size: int
    walls_per_player: int
    max_moves: int
    no_progress_limit: int
    history_length: int
    draw_penalty: float

    @property
    def action_space_size(self) -> int:
        """Calculate total action space size.

        Returns:
            Total number of possible actions (moves + walls)
        """
        # Pawn moves: board_size^2 (all cells)
        pawn_moves = self.board_size ** 2
        # Wall placements: (board_size-1)^2 positions * 2 orientations
        wall_placements = 2 * (self.board_size - 1) ** 2
        return pawn_moves + wall_placements

    @property
    def wall_grid_size(self) -> int:
        """Size of wall placement grid.

        Returns:
            Wall grid size (board_size - 1)
        """
        return self.board_size - 1


def load_game_config(config_name: str = "standard") -> GameConfig:
    """Load game configuration from YAML file.

    Args:
        config_name: Name of configuration ("standard", "small", "tiny")

    Returns:
        GameConfig object with loaded parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If config_name not found in file
        ValueError: If config values are invalid
    """
    # Find config file relative to this module
    config_path = Path(__file__).parent.parent.parent / "config" / "game_configs.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)

    if config_name not in all_configs:
        raise KeyError(f"Config '{config_name}' not found. Available: {list(all_configs.keys())}")

    config_data = all_configs[config_name]

    # Validate required fields
    required_fields = ['board_size', 'walls_per_player', 'no_progress_limit',
                       'history_length', 'draw_penalty']
    missing = [f for f in required_fields if f not in config_data]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # Add max_moves if not present (default based on no_progress_limit)
    if 'max_moves' not in config_data:
        config_data['max_moves'] = config_data['no_progress_limit'] * 4

    # Validate values
    if config_data['board_size'] < 3 or config_data['board_size'] > 21:
        raise ValueError(f"board_size must be between 3 and 21, got {config_data['board_size']}")

    if config_data['walls_per_player'] < 0:
        raise ValueError(f"walls_per_player must be non-negative, got {config_data['walls_per_player']}")

    return GameConfig(
        board_size=config_data['board_size'],
        walls_per_player=config_data['walls_per_player'],
        max_moves=config_data['max_moves'],
        no_progress_limit=config_data['no_progress_limit'],
        history_length=config_data['history_length'],
        draw_penalty=config_data['draw_penalty']
    )


def load_all_configs() -> Dict[str, GameConfig]:
    """Load all available game configurations.

    Returns:
        Dictionary mapping config names to GameConfig objects
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "game_configs.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)

    return {name: load_game_config(name) for name in all_configs.keys()}

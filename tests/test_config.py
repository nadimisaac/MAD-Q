"""Tests for game configuration loading."""

import pytest
from src.game.config import GameConfig, load_game_config, load_all_configs


class TestGameConfig:
    """Test GameConfig dataclass properties."""

    def test_action_space_size_standard(self):
        """Test action space size calculation for standard 9x9 board."""
        config = GameConfig(
            board_size=9,
            walls_per_player=10,
            max_moves=200,
            no_progress_limit=50,
            history_length=4,
            draw_penalty=-0.05
        )
        # 9^2 pawn moves + 2 * 8^2 wall placements
        assert config.action_space_size == 81 + 2 * 64
        assert config.action_space_size == 209

    def test_action_space_size_small(self):
        """Test action space size calculation for small 5x5 board."""
        config = GameConfig(
            board_size=5,
            walls_per_player=5,
            max_moves=120,
            no_progress_limit=30,
            history_length=4,
            draw_penalty=-0.05
        )
        # 5^2 pawn moves + 2 * 4^2 wall placements
        assert config.action_space_size == 25 + 2 * 16
        assert config.action_space_size == 57

    def test_wall_grid_size(self):
        """Test wall grid size calculation."""
        config = GameConfig(
            board_size=9,
            walls_per_player=10,
            max_moves=200,
            no_progress_limit=50,
            history_length=4,
            draw_penalty=-0.05
        )
        assert config.wall_grid_size == 8

        config_small = GameConfig(
            board_size=5,
            walls_per_player=5,
            max_moves=120,
            no_progress_limit=30,
            history_length=4,
            draw_penalty=-0.05
        )
        assert config_small.wall_grid_size == 4


class TestLoadGameConfig:
    """Test loading configurations from YAML file."""

    def test_load_standard_config(self):
        """Test loading standard 9x9 configuration."""
        config = load_game_config("standard")
        assert config.board_size == 9
        assert config.walls_per_player == 10
        assert config.no_progress_limit == 50
        assert config.history_length == 4
        assert config.draw_penalty == -0.05
        assert config.max_moves == 200  # Should be auto-calculated

    def test_load_small_config(self):
        """Test loading small 5x5 configuration."""
        config = load_game_config("small")
        assert config.board_size == 5
        assert config.walls_per_player == 5
        assert config.no_progress_limit == 30
        assert config.max_moves == 120

    def test_load_tiny_config(self):
        """Test loading tiny 3x3 configuration."""
        config = load_game_config("tiny")
        assert config.board_size == 3
        assert config.walls_per_player == 2
        assert config.no_progress_limit == 20
        assert config.max_moves == 80

    def test_load_default_config(self):
        """Test loading default configuration (should be 'standard')."""
        config = load_game_config()
        assert config.board_size == 9

    def test_load_invalid_config_name(self):
        """Test that loading invalid config raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            load_game_config("invalid_config_name")
        assert "invalid_config_name" in str(exc_info.value)

    def test_action_space_size_loaded_configs(self):
        """Test that loaded configs have correct action space sizes."""
        standard = load_game_config("standard")
        assert standard.action_space_size == 209

        small = load_game_config("small")
        assert small.action_space_size == 57

        tiny = load_game_config("tiny")
        assert tiny.action_space_size == 17  # 3^2 + 2*2^2


class TestLoadAllConfigs:
    """Test loading all configurations at once."""

    def test_load_all_configs(self):
        """Test loading all available configurations."""
        all_configs = load_all_configs()
        assert isinstance(all_configs, dict)
        assert "standard" in all_configs
        assert "small" in all_configs
        assert "tiny" in all_configs
        assert len(all_configs) >= 3

    def test_all_configs_valid(self):
        """Test that all loaded configs are valid GameConfig objects."""
        all_configs = load_all_configs()
        for name, config in all_configs.items():
            assert isinstance(config, GameConfig)
            assert config.board_size >= 3
            assert config.walls_per_player >= 0
            assert config.max_moves > 0
            assert config.no_progress_limit > 0

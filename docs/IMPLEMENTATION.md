# Quoridor Gumbel AlphaZero - Implementation Guide

## Project Overview

Building a Quoridor implementation for training with Gumbel AlphaZero. The game uses a novel state representation with spatial features (16-channel CNN input) + scalar features (5 values), and an action space of 209 actions for standard 9×9 board.

## Repository Structure

```
GAZE-Quoridor/
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── config/
│   └── game_configs.yaml
├── src/
│   ├── __init__.py
│   ├── game/
│   │   ├── __init__.py
│   │   ├── board.py              # Board state (pawns, walls)
│   │   ├── game_state.py         # Complete game state with history
│   │   ├── moves.py              # Move generation and validation
│   │   └── config.py             # Config loader
│   ├── pathfinding/
│   │   ├── __init__.py
│   │   └── astar.py              # A* for wall validation
│   ├── rl_interface/
│   │   ├── __init__.py
│   │   ├── state_encoder.py     # GameState -> (spatial, scalar) tensors
│   │   ├── action_space.py      # Action encoding/decoding
│   │   └── quoridor_env.py      # Gymnasium environment
│   ├── network/
│   │   ├── __init__.py
│   │   └── quoridor_net.py      # Neural network architecture
│   ├── notation/
│   │   ├── __init__.py
│   │   └── converter.py         # Notation <-> internal representation
│   ├── mcts/
│   │   ├── __init__.py
│   │   └── mcts_interface.py    # Interface for MCTS algorithms
│   └── utils/
│       ├── __init__.py
│       └── visualization.py     # Board rendering
├── tests/
│   ├── __init__.py
│   ├── test_board.py
│   ├── test_moves.py
│   ├── test_pathfinding.py
│   ├── test_game_state.py
│   ├── test_state_encoder.py
│   ├── test_action_space.py
│   ├── test_environment.py
│   ├── test_network.py
│   └── test_integration.py
├── examples/
│   ├── play_random.py           # Random agent demo
│   └── play_human.py            # Human vs agent
├── docs/
│   ├── SPECIFICATION.md         # Complete state/action space spec
│   └── IMPLEMENTATION.md        # This file - setup and build guide
└── notebooks/
    └── visualization.ipynb      # Jupyter notebook for analysis
```

## Initial Files to Create

### 1. .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Trained models
*.pth
*.pt
*.h5
checkpoints/

# Data
data/
logs/
```

### 2. requirements.txt

```
numpy>=1.24.0
torch>=2.0.0
pyyaml>=6.0
gymnasium>=0.29.0
pytest>=7.4.0
```

### 3. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "quoridor-gaz"
version = "0.1.0"
description = "Quoridor implementation for Gumbel AlphaZero training"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24.0",
    "torch>=2.0.0",
    "pyyaml>=6.0",
    "gymnasium>=0.29.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]
```

### 4. config/game_configs.yaml

```yaml
standard:
  board_size: 9
  walls_per_player: 10
  no_progress_limit: 50
  history_length: 4
  draw_penalty: -0.05

small:
  board_size: 5
  walls_per_player: 5
  no_progress_limit: 30
  history_length: 4
  draw_penalty: -0.05

tiny:
  board_size: 3
  walls_per_player: 2
  no_progress_limit: 20
  history_length: 4
  draw_penalty: -0.05
```

### 5. README.md

See the main [README.md](../README.md) in the repository root.

## Key Technical Details for Implementation

### State Representation

```python
state = {
    'spatial': np.ndarray(shape=(16, board_size, board_size), dtype=np.float32),
    'scalars': np.ndarray(shape=(5,), dtype=np.float32)
}

# Spatial channels (per timestep, 4 timesteps):
# 0: Current player pawn (one-hot)
# 1: Opponent pawn (one-hot)
# 2: Horizontal walls (binary, marks BOTH cells of 2-cell wall)
# 3: Vertical walls (binary, marks BOTH cells of 2-cell wall)

# Scalars:
# 0: Current player walls remaining (normalized 0-1)
# 1: Opponent walls remaining (normalized 0-1)
# 2: Current player ID (0 or 1)
# 3: Move count (normalized 0-1)
# 4: Moves since last wall (normalized 0-1)
```

### Action Encoding

```python
# Action index ranges for 9×9 board:
# 0-80: Pawn moves (index = row * 9 + col)
# 81-144: Horizontal walls (64 positions on 8×8 grid)
# 145-208: Vertical walls (64 positions on 8×8 grid)
```

### Critical Implementation Requirements

1. **Wall Encoding**: Each wall is 2 cells long. When placing horizontal wall at (row, col), mark BOTH [row, col] and [row, col+1] as 1 in the h_walls plane.

2. **Player Perspective**: State always encoded from current player's view. When Player 2's turn, flip board vertically and swap player labels.

3. **A* Validation**: Every wall placement MUST be validated with A* pathfinding to ensure both players can still reach their goals.

4. **Termination Conditions**:
   - Win: Player reaches opposite row (+1.0 / -1.0)
   - Draw: Move limit reached (-0.05 / -0.05)
   - Draw: No progress limit reached (-0.05 / -0.05)

5. **Action Masking**: Must return binary mask of legal actions at every state. This is critical for MCTS efficiency.

## Build Order

### Phase 1-5 (Core Game)

Build in order:

1. Board state representation
2. Pawn move generation
3. Wall placement (basic validation)
4. A* pathfinding
5. Complete game state with history

### Phase 6-8 (RL Interface)

6. State encoding to tensors
7. Action space encoding/decoding
8. Gymnasium environment wrapper

### Phase 9-10 (Network & Tools)

9. Neural network architecture
10. Visualization and debugging tools

## Testing Strategy

- Unit test each module as you build it
- Run 1000 random games in integration tests
- Verify no illegal moves slip through
- Check all termination conditions trigger correctly

## Documentation Reference

Complete specification with all formulas, network architecture, and design decisions is in [SPECIFICATION.md](SPECIFICATION.md).
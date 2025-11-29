# Quoridor Implementation Specification
## For Gumbel AlphaZero Training

---

## Overview

This document specifies the complete state representation, action space, and neural network architecture for a Quoridor implementation designed for reinforcement learning with Gumbel AlphaZero.

**Key Principles:**
- State always from current player's perspective
- History-based encoding (4 timesteps)
- Efficient separation of spatial and scalar features
- Configurable board sizes
- Termination rules to prevent infinite games

---

## State Space Representation

### 1. Spatial Features: `(16, H, W)` tensor

Where `H = W = board_size` (typically 9).

**Structure:** 4 channels per timestep × 4 timesteps = 16 channels

Always encoded from **current player's perspective**:
- Current player starts at bottom row (row 0)
- Current player's goal is top row (row 8 for 9×9)
- When Player 2's turn: board is flipped vertically and players swapped

#### Per Timestep (t ∈ {0, -1, -2, -3}):

| Channel Offset | Feature | Encoding | Example |
|----------------|---------|----------|---------|
| 0 | Current player pawn | One-hot: single 1 at pawn position | `pawn[4,4] = 1`, rest = 0 |
| 1 | Opponent pawn | One-hot: single 1 at pawn position | `pawn[4,2] = 1`, rest = 0 |
| 2 | Horizontal walls | Binary: **both cells** marked for each wall | See wall encoding below |
| 3 | Vertical walls | Binary: **both cells** marked for each wall | See wall encoding below |

**Channel Layout:**
```
Channels 0-3:   t=0 (current state)
Channels 4-7:   t=-1 (1 move ago)
Channels 8-11:  t=-2 (2 moves ago)  
Channels 12-15: t=-3 (3 moves ago)
```

#### Wall Encoding Details:

**Horizontal Wall** at position `(row, col)`:
- Blocks movement between `row` and `row+1`
- Spans columns `col` and `col+1` (2 cells wide)
- Encoding:
  ```python
  h_walls[row, col] = 1
  h_walls[row, col+1] = 1
  ```

**Vertical Wall** at position `(row, col)`:
- Blocks movement between `col` and `col+1`
- Spans rows `row` and `row+1` (2 cells tall)
- Encoding:
  ```python
  v_walls[row, col] = 1
  v_walls[row+1, col] = 1
  ```

**Example:** Wall notation `e3h` (horizontal wall at column e=4, row 3):
```python
h_walls[3, 4] = 1  # Mark both cells the wall spans
h_walls[3, 5] = 1
```

### 2. Scalar Features: `(5,)` vector

Global game state information, not spatially dependent.

| Index | Feature | Range | Normalization | Description |
|-------|---------|-------|---------------|-------------|
| 0 | Current player walls remaining | [0, 1] | `count / max_walls` | e.g., 8/10 = 0.8 |
| 1 | Opponent walls remaining | [0, 1] | `count / max_walls` | e.g., 7/10 = 0.7 |
| 2 | Current player ID | {0, 1} | None | 0 = Player 1, 1 = Player 2 |
| 3 | Move count | [0, 1] | `moves / max_moves` | Progress toward move limit |
| 4 | Moves since last wall | [0, 1] | `moves / no_progress_limit` | Progress toward no-progress draw |

### 3. Complete State Representation

```python
state = {
    'spatial': np.ndarray(shape=(16, board_size, board_size), dtype=np.float32),
    'scalars': np.ndarray(shape=(5,), dtype=np.float32)
}
```

**Memory footprint (9×9 board):**
- Spatial: 16 × 9 × 9 = 1,296 floats (5.2 KB)
- Scalars: 5 floats (20 bytes)
- **Total: 1,301 floats ≈ 5.2 KB per state**

---

## Action Space

### Action Space Size

**Formula:** `board_size² + 2 × (board_size - 1)²`

| Board Size | Pawn Moves | H-Walls | V-Walls | Total Actions |
|------------|------------|---------|---------|---------------|
| 3×3 | 9 | 4 | 4 | **17** |
| 5×5 | 25 | 16 | 16 | **57** |
| 7×7 | 49 | 36 | 36 | **121** |
| 9×9 | 81 | 64 | 64 | **209** |

### Action Encoding

Actions are encoded as integer indices from 0 to `action_space_size - 1`.

#### Action Index Ranges (9×9 board):

| Action Type | Index Range | Count | Description |
|-------------|-------------|-------|-------------|
| Pawn Move | 0 - 80 | 81 | Move to cell `(row, col)` |
| Horizontal Wall | 81 - 144 | 64 | Place wall at `(row, col)` in 8×8 grid |
| Vertical Wall | 145 - 208 | 64 | Place wall at `(row, col)` in 8×8 grid |

#### Encoding Functions:

```python
def encode_action(action_type: str, row: int, col: int, board_size: int = 9) -> int:
    """
    Convert (action_type, row, col) to action index.
    
    Args:
        action_type: 'move', 'h_wall', or 'v_wall'
        row, col: Position on board (0-indexed)
        board_size: Size of board
        
    Returns:
        Action index in range [0, action_space_size)
    """
    if action_type == 'move':
        return row * board_size + col
    elif action_type == 'h_wall':
        base = board_size ** 2
        return base + row * (board_size - 1) + col
    else:  # 'v_wall'
        base = board_size ** 2 + (board_size - 1) ** 2
        return base + row * (board_size - 1) + col

def decode_action(action_idx: int, board_size: int = 9) -> tuple[str, int, int]:
    """
    Convert action index to (action_type, row, col).
    
    Returns:
        (action_type, row, col) where action_type in {'move', 'h_wall', 'v_wall'}
    """
    pawn_actions = board_size ** 2
    wall_actions = (board_size - 1) ** 2
    
    if action_idx < pawn_actions:
        # Pawn move
        row = action_idx // board_size
        col = action_idx % board_size
        return ('move', row, col)
    elif action_idx < pawn_actions + wall_actions:
        # Horizontal wall
        offset = action_idx - pawn_actions
        row = offset // (board_size - 1)
        col = offset % (board_size - 1)
        return ('h_wall', row, col)
    else:
        # Vertical wall
        offset = action_idx - (pawn_actions + wall_actions)
        row = offset // (board_size - 1)
        col = offset % (board_size - 1)
        return ('v_wall', row, col)
```

### Action Masking

**Critical:** Action masking is required at every state.

```python
def get_action_mask(state: GameState) -> np.ndarray:
    """
    Returns binary mask of shape (action_space_size,)
    where mask[i] = 1 if action i is legal, 0 otherwise.
    """
    mask = np.zeros(action_space_size)
    
    # 1. Legal pawn moves (reachable adjacent cells or jumps)
    for move_pos in get_legal_pawn_moves(state):
        action_idx = encode_action('move', move_pos[0], move_pos[1])
        mask[action_idx] = 1
    
    # 2. Legal wall placements (if walls remaining)
    if state.current_player_walls > 0:
        for wall_type in ['h_wall', 'v_wall']:
            for (row, col) in get_legal_wall_positions(state, wall_type):
                # Check: no overlap, no cross pattern, both players can reach goal
                if is_wall_legal(state, row, col, wall_type):
                    action_idx = encode_action(wall_type, row, col)
                    mask[action_idx] = 1
    
    return mask
```

**Legality checks for walls:**
1. Wall doesn't overlap existing walls
2. Wall doesn't create a "cross" pattern with perpendicular wall
3. After placing wall, both players can still reach their goal (A* pathfinding check)
4. Current player has walls remaining

---

## Neural Network Architecture

### Input Processing: Late Fusion

```python
class QuoridorNet(nn.Module):
    def __init__(self, board_size=9, n_spatial_channels=16, n_scalars=5, n_res_blocks=10):
        super().__init__()
        
        self.board_size = board_size
        self.action_size = board_size**2 + 2*(board_size-1)**2
        
        # Convolutional trunk for spatial features
        self.conv_input = nn.Conv2d(n_spatial_channels, 256, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(256)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(n_res_blocks)
        ])
        
        # Flatten spatial features
        self.conv_output_size = 256 * board_size * board_size
        
        # Late fusion: combine flattened spatial + scalar features
        self.fc_combined = nn.Linear(self.conv_output_size + n_scalars, 512)
        self.bn_combined = nn.BatchNorm1d(512)
        
        # Policy head
        self.policy_fc1 = nn.Linear(512, 256)
        self.policy_bn1 = nn.BatchNorm1d(256)
        self.policy_fc2 = nn.Linear(256, self.action_size)
        
        # Value head
        self.value_fc1 = nn.Linear(512, 256)
        self.value_bn1 = nn.BatchNorm1d(256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, spatial, scalars):
        """
        Args:
            spatial: (batch, 16, board_size, board_size)
            scalars: (batch, 5)
            
        Returns:
            policy_logits: (batch, action_size) - raw logits before softmax
            value: (batch, 1) - value estimate in [-1, 1]
        """
        # Process spatial features through CNN
        x = F.relu(self.bn_input(self.conv_input(spatial)))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Flatten spatial features
        x_spatial = x.view(x.size(0), -1)
        
        # Concatenate spatial and scalar features (late fusion)
        x = torch.cat([x_spatial, scalars], dim=1)
        x = F.relu(self.bn_combined(self.fc_combined(x)))
        
        # Policy head
        p = F.relu(self.policy_bn1(self.policy_fc1(x)))
        policy_logits = self.policy_fc2(p)
        
        # Value head
        v = F.relu(self.value_bn1(self.value_fc1(x)))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
```

### Key Design Decisions:

1. **Late Fusion:** Scalars concatenated after CNN processing, not broadcast across spatial dimensions
2. **Residual Blocks:** Standard ResNet architecture for deep feature learning
3. **Separate Heads:** Policy and value heads diverge after shared trunk
4. **Policy Output:** Raw logits (before softmax) - softmax + masking applied externally
5. **Value Output:** Tanh activation for range [-1, 1]

---

## Game Rules & Termination

### Win/Loss/Draw Conditions

| Condition | Outcome | Reward (Winner/Loser) |
|-----------|---------|----------------------|
| Player reaches goal row | **Win** | +1.0 / -1.0 |
| Move count ≥ `max_moves` | **Draw** | -0.05 / -0.05 |
| Moves since last wall ≥ `no_progress_limit` | **Draw** | -0.05 / -0.05 |

**Rationale for draw penalty (-0.05):**
- Discourages stalling and infinite loops
- Small enough not to dominate training signal
- Agent prefers: Win > Draw > Loss
- Encourages active play toward the goal

### Game Configuration

```yaml
standard:
  board_size: 9
  walls_per_player: 10
  max_moves: 200
  no_progress_limit: 50
  history_length: 4
  draw_penalty: -0.05

small:
  board_size: 5
  walls_per_player: 5
  max_moves: 100
  no_progress_limit: 30
  history_length: 4
  draw_penalty: -0.05

tiny:
  board_size: 3
  walls_per_player: 2
  max_moves: 50
  no_progress_limit: 20
  history_length: 4
  draw_penalty: -0.05
```

### Move Validation

**Pawn Moves:**
- Adjacent cell (4 cardinal directions)
- Jump over opponent if opponent is adjacent and no wall blocks
- If opponent is adjacent and blocked behind, diagonal jump is allowed

**Wall Placements:**
1. Player must have walls remaining
2. Wall position must be on valid grid: `(board_size-1) × (board_size-1)`
3. Wall must not overlap existing walls
4. Wall must not create perpendicular cross with existing wall
5. **Critical:** After placement, both players must have path to goal (A* check)

---

## A* Pathfinding for Wall Validation

Every wall placement must be validated using A* to ensure connectivity.

```python
def is_wall_placement_legal(state: GameState, row: int, col: int, 
                            direction: str) -> bool:
    """
    Check if wall placement is legal.
    
    Returns False if wall would block either player from reaching goal.
    """
    # Create temporary state with wall added
    temp_state = state.copy()
    temp_state.add_wall(row, col, direction)
    
    # Check if both players can still reach their goals
    player1_can_reach = astar_can_reach_goal(
        temp_state, 
        temp_state.player1_pos, 
        goal_row=temp_state.board_size - 1
    )
    
    player2_can_reach = astar_can_reach_goal(
        temp_state,
        temp_state.player2_pos,
        goal_row=0
    )
    
    return player1_can_reach and player2_can_reach


def astar_can_reach_goal(state: GameState, start_pos: tuple, 
                         goal_row: int) -> bool:
    """
    A* pathfinding to check if position can reach goal row.
    
    Args:
        state: Current game state
        start_pos: (row, col) starting position
        goal_row: Target row to reach
        
    Returns:
        True if path exists, False otherwise
    """
    def heuristic(pos):
        return abs(pos[0] - goal_row)
    
    open_set = [(heuristic(start_pos), start_pos)]
    closed_set = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current[0] == goal_row:
            return True
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor in get_reachable_neighbors(state, current):
            if neighbor not in closed_set:
                heapq.heappush(open_set, (heuristic(neighbor), neighbor))
    
    return False
```

---

## Notation System Integration

### Official Quoridor Notation

- Columns: `a-i` (0-8 in code)
- Rows: `1-9` (0-8 in code)
- Pawn moves: `e5` (column e, row 5)
- Wall placements: `e3h` (horizontal wall at e3), `d4v` (vertical wall at d4)

### Conversion Functions

```python
def position_to_notation(row: int, col: int) -> str:
    """Convert (row, col) to notation like 'e5'"""
    return f"{chr(ord('a') + col)}{row + 1}"

def notation_to_position(notation: str) -> tuple[int, int]:
    """Convert 'e5' to (row=4, col=4)"""
    col = ord(notation[0]) - ord('a')
    row = int(notation[1]) - 1
    return (row, col)

def action_to_notation(action_idx: int, board_size: int = 9) -> str:
    """Convert action index to notation"""
    action_type, row, col = decode_action(action_idx, board_size)
    
    if action_type == 'move':
        return position_to_notation(row, col)
    elif action_type == 'h_wall':
        return position_to_notation(row, col) + 'h'
    else:  # v_wall
        return position_to_notation(row, col) + 'v'

def notation_to_action(notation: str, board_size: int = 9) -> int:
    """Convert notation to action index"""
    if len(notation) == 2:
        # Pawn move
        row, col = notation_to_position(notation)
        return encode_action('move', row, col, board_size)
    else:
        # Wall placement
        row, col = notation_to_position(notation[:2])
        direction = 'h_wall' if notation[2] == 'h' else 'v_wall'
        return encode_action(direction, row, col, board_size)
```

---

## Implementation Checklist

### Core Game Logic
- [ ] `GameState` class with state representation
- [ ] Move generation (legal pawn moves)
- [ ] Wall placement validation
- [ ] A* pathfinding for connectivity checks
- [ ] Win/draw/loss detection
- [ ] History tracking (last 4 states)
- [ ] No-progress counter (moves since last wall)
- [ ] Move limit counter

### State Encoding
- [ ] `encode_state()` function returning `(spatial, scalars)` tuple
- [ ] Wall encoding (mark both cells)
- [ ] Board orientation (always current player's perspective)
- [ ] History stacking (4 timesteps)
- [ ] Scalar normalization

### Action Space
- [ ] `encode_action()` and `decode_action()` functions
- [ ] Action masking generation
- [ ] Notation conversion (to/from official format)

### Neural Network
- [ ] `QuoridorNet` with late fusion architecture
- [ ] Policy head (raw logits output)
- [ ] Value head (tanh activation)
- [ ] Residual blocks for deep feature learning

### Game Environment
- [ ] Gymnasium-compatible environment wrapper
- [ ] `reset()`, `step()`, `render()` methods
- [ ] Action masking in info dict
- [ ] Configurable board sizes

### Testing
- [ ] Unit tests for move validation
- [ ] Unit tests for wall placement
- [ ] Unit tests for A* pathfinding
- [ ] Unit tests for state encoding
- [ ] Unit tests for action encoding/decoding
- [ ] Integration tests for full games
- [ ] Test different board sizes (3×3, 5×5, 9×9)

---

## Summary Statistics

| Board | State Size | Action Space | Avg Branching | Typical Game Length |
|-------|------------|--------------|---------------|---------------------|
| 3×3 | 149 floats | 17 actions | ~8 | 10-20 moves |
| 5×5 | 405 floats | 57 actions | ~25 | 20-40 moves |
| 9×9 | 1,301 floats | 209 actions | ~70 | 40-100 moves |

**Key Advantages:**
- ✅ Efficient state representation (1.3KB for 9×9)
- ✅ Proper wall encoding (2-cell spans)
- ✅ CNN-friendly spatial structure
- ✅ Scalars separated (no redundancy)
- ✅ History for temporal learning
- ✅ Configurable for multiple board sizes
- ✅ Draw penalties prevent infinite games
- ✅ Action masking for efficient training

---

## Relationship Between State and Action Spaces

**Important:** The neural network does **not** directly "understand" the connection between states and actions. This relationship is **learned through self-play**:

1. **Initially:** Network outputs random probabilities for all actions
2. **Action Masking:** Illegal actions are masked to probability 0
3. **MCTS Exploration:** Monte Carlo Tree Search tries legal actions
4. **Outcome Observation:** Network observes which actions led to wins/losses
5. **Gradient Updates:** Backpropagation adjusts network to prefer winning actions
6. **Learned Understanding:** Over time, network learns patterns like:
   - "When opponent close, move forward"
   - "When ahead, block opponent's shortest path"
   - "When low on walls, prioritize movement"

The action space is a fixed encoding scheme. The network learns which actions are good in which states through experience, not through explicit programming.

---

*Document Version: 1.0*  
*Last Updated: 2025*  
*For Gumbel AlphaZero Quoridor Implementation*
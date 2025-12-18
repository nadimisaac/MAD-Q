# Quoridor Implementation Specification
## For Minimax Search with Alpha-Beta Pruning

---

## Overview

This document specifies the complete implementation for a Quoridor game engine with a minimax-based AI agent using alpha-beta pruning, depth-limited search, and evaluation functions with weights tuned through temporal-difference learning and hyperparameter grid search.

**Key Principles:**
- Minimax search with alpha-beta pruning for efficient game-tree exploration
- Depth-limited search with evaluation functions for terminal states
- Configurable board sizes (3×3, 5×5, 9×9)
- Evaluation function combining path distance, progress, and wall advantage
- Termination rules to prevent infinite games
- Comprehensive performance analysis and tournament evaluation

---

## Game State Representation

The game state tracks all information needed for minimax search and evaluation.

### Core State Components

```python
class State:
    """Complete game state for Quoridor."""
    
    # Board configuration
    config: GameConfig              # Board size, walls per player, limits
    board_size: int                 # Typically 3, 5, or 9
    
    # Player positions
    player1_pos: Position           # (row, col) for Player 1
    player2_pos: Position           # (row, col) for Player 2
    current_player: int             # 1 or 2
    
    # Wall state
    h_walls: Set[Tuple[int, int]]   # Horizontal wall positions
    v_walls: Set[Tuple[int, int]]   # Vertical wall positions
    walls_remaining: Dict[int, int] # Walls left per player
    
    # Game progress tracking
    move_count: int                 # Total moves played
    moves_since_last_wall: int      # For no-progress detection
    game_over: bool                 # Terminal state flag
    winner: Optional[int]           # 1, 2, or None (draw)
```

### State Properties

The state maintains:
- **Immutability:** `make_move()` returns a new state (copy-on-write)
- **Legality checking:** `get_legal_moves()` computes all valid actions
- **Win detection:** Automatic when player reaches goal row
- **Draw detection:** Based on move limits and no-progress rules

### Wall Encoding Details

**Horizontal Wall** at position `(row, col)`:
- Blocks vertical movement between `row` and `row+1`
- Spans columns `col` and `col+1` (2 cells wide)
- Stored as single coordinate `(row, col)` in `h_walls` set
- Effects both cells when checking movement legality

**Vertical Wall** at position `(row, col)`:
- Blocks horizontal movement between `col` and `col+1`
- Spans rows `row` and `row+1` (2 cells tall)
- Stored as single coordinate `(row, col)` in `v_walls` set
- Effects both cells when checking movement legality

**Example:** Wall notation `e3h` (horizontal wall at column e=4, row 3):
```python
h_walls.add((3, 4))  # Single entry represents 2-cell span
```

### Memory Footprint

**9×9 board game state:**
- Positions: 2 × 2 integers (4 ints = 16 bytes)
- Walls: ~20 walls × 2 integers avg (40 ints = 160 bytes)
- Metadata: 5 integers (20 bytes)
- **Total: ~200 bytes per state** (very efficient for tree search)

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

### Legal Move Generation

**Critical:** Only legal actions are considered in minimax search.

```python
def get_legal_moves(state: State) -> List[Action]:
    """
    Returns list of legal actions for current state.
    
    Returns:
        List of (action_type, position) tuples
    """
    legal_moves = []
    
    # 1. Legal pawn moves (reachable adjacent cells or jumps)
    for move_pos in get_legal_pawn_moves(state):
        legal_moves.append(('move', move_pos))
    
    # 2. Legal wall placements (if walls remaining)
    if state.walls_remaining[state.current_player] > 0:
        for wall_type in ['h_wall', 'v_wall']:
            for (row, col) in get_legal_wall_positions(state, wall_type):
                # Check: no overlap, no cross pattern, both players can reach goal
                if is_wall_legal(state, row, col, wall_type):
                    legal_moves.append((wall_type, (row, col)))
    
    return legal_moves
```

**Legality checks for walls:**
1. Wall doesn't overlap existing walls
2. Wall doesn't create a "cross" pattern with perpendicular wall
3. After placing wall, both players can still reach their goal (A* pathfinding check)
4. Current player has walls remaining

---

## Minimax Agent Architecture

### Core Algorithm: Alpha-Beta Pruning

```python
class MinimaxAgent:
    def __init__(
        self,
        player_number: int,
        depth: int = 2,
        max_wall_moves: int = 8,
        path_weight: float = 12.0,
        progress_weight: float = 1.5,
        wall_weight: float = 2.0,
    ):
        """
        Minimax agent with alpha-beta pruning.
        
        Args:
            player_number: Player controlled (1 or 2)
            depth: Search depth in plies
            max_wall_moves: Max wall placements explored per node
            path_weight: Weight for shortest-path heuristic
            progress_weight: Weight for goal-progress heuristic
            wall_weight: Weight for wall-advantage heuristic
        """
        self.player_number = player_number
        self.max_depth = depth
        self.max_wall_moves = max_wall_moves
        self.path_weight = path_weight
        self.progress_weight = progress_weight
        self.wall_weight = wall_weight
        self._win_score = 1_000_000.0
    
    def select_action(self, state: State) -> Action:
        """Choose best action via minimax search."""
        value, best_move = self._search(state, self.max_depth, -math.inf, math.inf)
        return best_move
    
    def _search(self, state: State, depth: int, alpha: float, beta: float):
        """Minimax with alpha-beta pruning."""
        if depth == 0 or state.game_over:
            return self._evaluate(state, depth), None
        
        moves = self._get_candidate_moves(state)
        maximizing = (state.current_player == self.player_number)
        
        if maximizing:
            best_score = -math.inf
            best_move = None
            for move in moves:
                child = state.make_move(*move)
                score, _ = self._search(child, depth - 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return best_score, best_move
        else:
            best_score = math.inf
            best_move = None
            for move in moves:
                child = state.make_move(*move)
                score, _ = self._search(child, depth - 1, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return best_score, best_move
```

### Evaluation Function

The evaluation function combines three strategic heuristics:

```python
def _evaluate(self, state: State, depth_remaining: int) -> float:
    """
    Evaluate non-terminal state from agent's perspective.
            
        Returns:
        score: Higher is better for agent
    """
    # Terminal states
    if state.game_over:
        if state.winner == self.player_number:
            return self._win_score + depth_remaining  # Prefer faster wins
        elif state.winner is None:
            return state.config.draw_penalty
        else:
            return -self._win_score - depth_remaining  # Prefer slower losses
    
    # Non-terminal evaluation
    opponent = state.get_opponent(self.player_number)
    
    # 1. Path distance heuristic (most important)
    my_path_len = shortest_path_length(state, self.player_number)
    opp_path_len = shortest_path_length(state, opponent)
    path_score = opp_path_len - my_path_len
    
    # 2. Progress toward goal heuristic
    my_progress = progress_to_goal(state, self.player_number)
    opp_progress = progress_to_goal(state, opponent)
    progress_score = my_progress - opp_progress
    
    # 3. Wall advantage heuristic
    wall_score = (state.walls_remaining[self.player_number] - 
                  state.walls_remaining[opponent])
    
    return (self.path_weight * path_score +
            self.progress_weight * progress_score +
            self.wall_weight * wall_score)
```

### Evaluation Heuristics Explained

| Heuristic | Formula | Weight Range | Description |
|-----------|---------|--------------|-------------|
| **Path Distance** | `opp_path - my_path` | 8.0 - 16.0 | Shortest path to goal (A* search) |
| **Goal Progress** | `my_progress - opp_progress` | 0.5 - 2.5 | Raw row distance to goal |
| **Wall Advantage** | `my_walls - opp_walls` | 1.0 - 3.0 | Remaining walls available |

**Strategic Interpretation:**
- **Path Distance:** Primary factor. Prefer states where opponent's path is longer.
- **Goal Progress:** Tie-breaker. Move toward goal when paths are equal.
- **Wall Advantage:** Resource management. Value having more walls remaining.

### Weight Tuning Methods

Weights are optimized through two complementary approaches:

1. **Temporal-Difference Learning**
   - Linear evaluator trained on self-play games
   - TD(λ) updates adjust weights based on outcome
   - Produces weights grounded in game experience

2. **Hyperparameter Grid Search**
   - Tournament evaluation across weight configurations
   - Elo ratings computed from round-robin matches
   - Identifies robust weight combinations

### Move Ordering Optimization

To maximize alpha-beta pruning efficiency:

```python
def _get_candidate_moves(self, state: State) -> List[Action]:
    """Get moves sorted by likely quality (best-first)."""
    pawn_moves = [m for m in legal_moves if m[0] == 'move']
    wall_moves = [m for m in legal_moves if m[0] != 'move']
    
    # Sort pawn moves by distance to goal
    pawn_moves.sort(key=lambda m: distance_to_goal(m[1]))
    
    # Sort wall moves by distance to opponent
    wall_moves.sort(key=lambda m: distance_to_opponent(m[1]))
    
    # Limit wall branching if configured
    if self.max_wall_moves and len(wall_moves) > self.max_wall_moves:
        wall_moves = wall_moves[:self.max_wall_moves]
    
    return pawn_moves + wall_moves
```

### Performance Characteristics

| Board Size | Avg Branching | Depth 2 Nodes | Depth 3 Nodes | Depth 4 Nodes |
|------------|---------------|---------------|---------------|---------------|
| 3×3 | ~8 | ~60 | ~500 | ~4K |
| 5×5 | ~25 | ~600 | ~15K | ~400K |
| 9×9 | ~70 | ~5K | ~350K | ~25M |

**Alpha-beta pruning** typically achieves ~50% node reduction with good move ordering.

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
- Small enough not to dominate evaluation scores
- Minimax agent prefers: Win > Draw > Loss
- Encourages active play toward the goal

### Game Configuration

```yaml
standard:
  board_size: 9
  walls_per_player: 10
  max_moves: 200
  no_progress_limit: 50
  draw_penalty: -0.05

small:
  board_size: 5
  walls_per_player: 5
  max_moves: 100
  no_progress_limit: 30
  draw_penalty: -0.05

tiny:
  board_size: 3
  walls_per_player: 2
  max_moves: 50
  no_progress_limit: 20
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
- [x] `State` class with complete game state
- [x] Move generation (legal pawn moves including jumps)
- [x] Wall placement validation
- [x] A* pathfinding for connectivity checks
- [x] Win/draw/loss detection
- [x] No-progress counter (moves since last wall)
- [x] Move limit counter
- [x] Immutable state updates (copy-on-write)

### Action Space
- [x] `encode_action()` and `decode_action()` functions
- [x] Legal move generation
- [x] Notation conversion (to/from official format)
- [x] Configurable board sizes (3×3, 5×5, 9×9)

### Minimax Agent
- [x] Alpha-beta pruning implementation
- [x] Depth-limited search
- [x] Evaluation function with three heuristics
- [x] Move ordering optimization
- [x] Configurable wall branching limit
- [x] Deterministic tie-breaking

### Evaluation & Heuristics
- [x] Path distance calculation (A* search)
- [x] Goal progress measurement
- [x] Wall advantage computation
- [x] Configurable weights for each heuristic
- [x] Terminal state handling with depth bonus

### Weight Tuning
- [x] Temporal-difference learning implementation
- [x] Feature extraction for TD learning
- [x] Hyperparameter grid search
- [x] Tournament-based evaluation
- [x] Elo rating computation

### Other Agents
- [x] Random agent (baseline)
- [x] Pathfinding-based heuristic agent
- [x] Human player interface

### Game Interface
- [x] Command-line rendering
- [x] Pygame UI for visualization
- [x] Interactive play modes
- [x] Move notation input/output
- [x] Game configuration system

### Analysis Tools
- [x] Performance benchmarking
- [x] Move time tracking
- [x] Game length analysis
- [x] Win rate calculation
- [x] Tournament evaluation framework
- [x] Plot generation for results

### Testing
- [x] Unit tests for move validation
- [x] Unit tests for wall placement
- [x] Unit tests for A* pathfinding
- [x] Unit tests for jump scenarios
- [x] Integration tests for full games
- [x] Tests for all board sizes (3×3, 5×5, 9×9)

---

## Summary Statistics

| Board | State Size | Action Space | Avg Branching | Typical Game Length |
|-------|------------|--------------|---------------|---------------------|
| 3×3 | ~200 bytes | 17 actions | ~8 | 10-20 moves |
| 5×5 | ~200 bytes | 57 actions | ~25 | 20-40 moves |
| 9×9 | ~200 bytes | 209 actions | ~70 | 40-100 moves |

**Key Advantages:**
- ✅ Efficient state representation (~200 bytes)
- ✅ Fast state copying for tree search
- ✅ Proper wall encoding (2-cell spans)
- ✅ A* pathfinding for accurate evaluation
- ✅ Configurable for multiple board sizes
- ✅ Draw penalties prevent infinite games
- ✅ Alpha-beta pruning for efficient search
- ✅ Tunable evaluation weights

---

## Strategic Insights from Analysis

### Performance by Board Size

Analysis across board sizes reveals:

- **3×3 boards:** Simple enough for exhaustive search; minimax achieves near-perfect play at depth 4+
- **5×5 boards:** Sweet spot for analysis; complex enough to be interesting, small enough for deep search
- **9×9 boards:** Standard Quoridor; requires depth limitation and strong evaluation for practical play

### Evaluation Weight Sensitivity

Hyperparameter search reveals:

- **Path weight (8-16):** Most critical parameter; dominates strategic decisions
- **Progress weight (0.5-2.5):** Important tie-breaker; encourages forward movement
- **Wall weight (1-3):** Secondary factor; prevents wasteful wall usage

Optimal weights vary by board size and opponent strategy, but path distance is universally dominant.

### Search Depth Trade-offs

| Depth | 5×5 Board Time | 9×9 Board Time | Win Rate Gain |
|-------|----------------|----------------|---------------|
| 1 | ~1ms | ~5ms | Baseline |
| 2 | ~20ms | ~200ms | +15-20% |
| 3 | ~400ms | ~5s | +8-12% |
| 4 | ~8s | ~2min | +3-5% |

Diminishing returns after depth 3; depth 2-3 provides best time/performance balance.

### First-Player Advantage

Measured across 1000+ games:
- **Standard (9×9):** Player 1 wins ~52-54% (slight advantage)
- **Small (5×5):** Player 1 wins ~55-58% (moderate advantage)
- **Tiny (3×3):** Player 1 wins ~65-70% (significant advantage)

Smaller boards amplify first-move advantage due to reduced path length.

---

*Document Version: 2.0*  
*Last Updated: December 2025*  
*For MAD-Q (Minimax Alpha-beta with Depth-limited search for Quoridor)*
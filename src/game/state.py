"""Complete game state representation for Quoridor.

This module contains the State class which represents the complete state of a Quoridor game,
including pawn positions, walls, turn information, and game history.
"""

from typing import List, Optional, Tuple, Set, Dict, Deque
from dataclasses import dataclass
from collections import deque
from .config import GameConfig


@dataclass
class Position:
    """A position on the board.

    Attributes:
        row: Row index (0-indexed)
        col: Column index (0-indexed)
    """
    row: int
    col: int

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Position):
            return False
        return self.row == other.row and self.col == other.col

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple representation."""
        return (self.row, self.col)


@dataclass
class GameResult:
    """Result of a completed game.

    Attributes:
        winner: Player number (1 or 2) or None for draw
        reason: Reason for game end ('goal_reached', 'max_moves', 'no_progress')
        num_moves: Total number of moves made
    """
    winner: Optional[int]
    reason: str
    num_moves: int


class State:
    """Complete Quoridor game state.

    Represents everything about the current game position:
    - Pawn positions for both players
    - All placed walls (horizontal and vertical)
    - Current player turn
    - Walls remaining for each player
    - Move history
    - Game configuration

    This is a unified class that contains both the board state and game-level
    information. No separate Board class is needed.

    Attributes:
        config: GameConfig object
        player1_pos: Position of player 1's pawn
        player2_pos: Position of player 2's pawn
        h_walls: Set of horizontal wall positions (row, col)
        v_walls: Set of vertical wall positions (row, col)
        current_player: Player whose turn it is (1 or 2)
        walls_remaining: Dict mapping player (1, 2) to walls remaining
        move_count: Total number of moves made (for max_moves termination)
        moves_since_last_wall: Moves since last wall placed (for no-progress termination)
        history: Deque of last 4 states (for neural network temporal features)
        game_over: Whether game has ended
        winner: Winner (1, 2) or None if ongoing/draw
    """

    def __init__(self, config: GameConfig):
        """Initialize a new game state.

        Args:
            config: GameConfig object with game parameters
        """
        self.config = config

        # Pawn positions
        self.player1_pos = Position(row=0, col=config.board_size // 2)
        self.player2_pos = Position(row=config.board_size - 1, col=config.board_size // 2)

        # Walls - sets of (row, col) tuples
        self.h_walls: Set[Tuple[int, int]] = set()
        self.v_walls: Set[Tuple[int, int]] = set()

        # Game state
        self.current_player = 1
        self.walls_remaining: Dict[int, int] = {
            1: config.walls_per_player,
            2: config.walls_per_player
        }

        # Move tracking
        self.move_count = 0
        self.moves_since_last_wall = 0

        # History for neural network (stores last N states)
        self.history: Deque = deque(maxlen=config.history_length)

        # Game status
        self.game_over = False
        self.winner: Optional[int] = None

    # ============================================================================
    # Action Execution
    # ============================================================================

    def make_move(self, action_type: str, position: Tuple[int, int]) -> 'State':
        """Execute a move and return new game state.

        Args:
            action_type: Type of action ('move', 'h_wall', 'v_wall')
            position: Position for the action

        Returns:
            New State after applying the action

        Raises:
            ValueError: If move is illegal
        """
        # Create a copy of the state
        new_state = self.copy()

        # Execute the action
        if action_type == 'move':
            # Move pawn
            if not new_state.is_legal_pawn_move(position):
                raise ValueError(f"Illegal pawn move to {position}")

            if new_state.current_player == 1:
                new_state.player1_pos = Position(position[0], position[1])
            else:
                new_state.player2_pos = Position(position[0], position[1])

            new_state.moves_since_last_wall += 1

        elif action_type == 'h_wall':
            # Place horizontal wall
            if not new_state.is_legal_wall_placement(position[0], position[1], 'h'):
                raise ValueError(f"Illegal horizontal wall placement at {position}")

            new_state.h_walls.add(position)
            new_state.walls_remaining[new_state.current_player] -= 1
            new_state.moves_since_last_wall = 0

        elif action_type == 'v_wall':
            # Place vertical wall
            if not new_state.is_legal_wall_placement(position[0], position[1], 'v'):
                raise ValueError(f"Illegal vertical wall placement at {position}")

            new_state.v_walls.add(position)
            new_state.walls_remaining[new_state.current_player] -= 1
            new_state.moves_since_last_wall = 0

        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # Update move count
        new_state.move_count += 1

        # Check for game termination
        result = new_state.check_termination()
        if result:
            new_state.game_over = True
            new_state.winner = result.winner

        # Switch player
        new_state.switch_player()

        return new_state

    # ============================================================================
    # Legal Move Generation
    # ============================================================================

    def get_legal_moves(self) -> List[Tuple[str, Tuple[int, int]]]:
        """Get all legal actions (moves + walls) for current player.

        Returns:
            List of (action_type, position) tuples
        """
        legal_actions = []

        # Add pawn moves
        for pos in self.get_legal_pawn_moves():
            legal_actions.append(('move', pos))

        # Add wall placements
        for pos in self.get_legal_wall_placements('h'):
            legal_actions.append(('h_wall', pos))
        for pos in self.get_legal_wall_placements('v'):
            legal_actions.append(('v_wall', pos))

        return legal_actions

    def get_legal_pawn_moves(self) -> List[Tuple[int, int]]:
        """Get all legal pawn moves for current player.

        Includes:
        - Adjacent moves (up, down, left, right)
        - Jump moves over opponent
        - Diagonal jumps when opponent is blocked from behind

        Returns:
            List of legal pawn destination positions as (row, col) tuples
        """
        from .moves import _get_adjacent_positions, _can_move_between, _is_straight_jump, _is_diagonal_jump

        current_pos = self.get_pawn_position(self.current_player)
        opponent_pos = self.get_pawn_position(self.get_opponent(self.current_player))
        legal_moves = []

        # Get all adjacent positions
        adjacent = _get_adjacent_positions(current_pos.to_tuple(), self.config.board_size)

        for adj_pos in adjacent:
            # Check if wall blocks movement
            if not _can_move_between(current_pos.to_tuple(), adj_pos, self.h_walls, self.v_walls):
                continue

            # Check if opponent is in this adjacent position
            if adj_pos == opponent_pos.to_tuple():
                # Need to jump over opponent
                # Try straight jump first (2 cells in same direction)
                dr = adj_pos[0] - current_pos.row
                dc = adj_pos[1] - current_pos.col
                straight_land = (adj_pos[0] + dr, adj_pos[1] + dc)

                # Check if straight landing position is on board
                if (0 <= straight_land[0] < self.config.board_size and
                    0 <= straight_land[1] < self.config.board_size):
                    # Check if no wall blocks the jump
                    if _can_move_between(adj_pos, straight_land, self.h_walls, self.v_walls):
                        if _is_straight_jump(current_pos.to_tuple(), adj_pos, straight_land):
                            legal_moves.append(straight_land)
                            continue  # Found straight jump, don't check diagonal

                # Straight jump not possible, try diagonal jumps
                # Get positions adjacent to opponent
                opp_adjacent = _get_adjacent_positions(adj_pos, self.config.board_size)
                for diag_land in opp_adjacent:
                    # Skip the position we came from
                    if diag_land == current_pos.to_tuple():
                        continue
                    # Skip the straight-ahead position (already checked)
                    if diag_land == straight_land:
                        continue
                    # Check if no wall blocks movement from opponent to diagonal landing
                    if _can_move_between(adj_pos, diag_land, self.h_walls, self.v_walls):
                        if _is_diagonal_jump(current_pos.to_tuple(), adj_pos, diag_land):
                            legal_moves.append(diag_land)
            else:
                # No opponent in the way, this is a legal adjacent move
                legal_moves.append(adj_pos)

        return legal_moves

    def get_legal_wall_placements(self, orientation: str) -> List[Tuple[int, int]]:
        """Get all legal wall placements for given orientation.

        Args:
            orientation: 'h' for horizontal, 'v' for vertical

        Returns:
            List of legal wall positions (passes all legality checks including connectivity)
        """
        legal_walls = []
        wall_grid_size = self.config.wall_grid_size

        for row in range(wall_grid_size):
            for col in range(wall_grid_size):
                if self.is_legal_wall_placement(row, col, orientation):
                    legal_walls.append((row, col))

        return legal_walls

    def is_legal_pawn_move(self, position: Tuple[int, int]) -> bool:
        """Check if a pawn move to position is legal.

        Args:
            position: Destination position (row, col)

        Returns:
            True if pawn move is legal
        """
        return position in self.get_legal_pawn_moves()

    def is_legal_wall_placement(self, row: int, col: int, orientation: str) -> bool:
        """Check if a wall placement is legal.

        Checks:
        1. Player has walls remaining
        2. No overlap with existing walls
        3. No perpendicular cross pattern
        4. Both players can still reach their goals (A* connectivity check)

        Args:
            row: Row index for wall placement
            col: Column index for wall placement
            orientation: 'h' for horizontal, 'v' for vertical

        Returns:
            True if wall placement is legal
        """
        # Check if player has walls remaining
        if self.walls_remaining[self.current_player] <= 0:
            return False

        # Check basic wall placement validity (no overlaps/crosses)
        if not self._is_wall_placement_valid(row, col, orientation):
            return False

        # Check A* connectivity - both players must still be able to reach their goals
        # Temporarily add this wall and check if paths exist
        temp_h_walls = self.h_walls.copy()
        temp_v_walls = self.v_walls.copy()

        if orientation == 'h':
            temp_h_walls.add((row, col))
        else:
            temp_v_walls.add((row, col))

        # Import A* pathfinding
        from ..pathfinding.astar import astar_path_exists

        # Check if player 1 can still reach their goal (top row)
        player1_can_reach = astar_path_exists(
            self.player1_pos.to_tuple(),
            self.config.board_size - 1,  # Goal: top row
            self.config.board_size,
            temp_h_walls,
            temp_v_walls
        )

        if not player1_can_reach:
            return False

        # Check if player 2 can still reach their goal (bottom row)
        player2_can_reach = astar_path_exists(
            self.player2_pos.to_tuple(),
            0,  # Goal: bottom row
            self.config.board_size,
            temp_h_walls,
            temp_v_walls
        )

        return player2_can_reach

    # ============================================================================
    # Game Status
    # ============================================================================

    def check_termination(self) -> Optional[GameResult]:
        """Check if game has ended and return result.

        Checks for:
        1. Goal reached (player wins)
        2. Max moves exceeded (draw)
        3. No progress limit exceeded (draw)

        Returns:
            GameResult if game is over, None if ongoing
        """
        # Check if player 1 reached goal (top row)
        if self.player1_pos.row == self.config.board_size - 1:
            return GameResult(winner=1, reason='goal_reached', num_moves=self.move_count)

        # Check if player 2 reached goal (bottom row)
        if self.player2_pos.row == 0:
            return GameResult(winner=2, reason='goal_reached', num_moves=self.move_count)

        # Check for max moves limit
        if self.move_count >= self.config.max_moves:
            return GameResult(winner=None, reason='max_moves', num_moves=self.move_count)

        # Check for no progress limit
        if self.moves_since_last_wall >= self.config.no_progress_limit:
            return GameResult(winner=None, reason='no_progress', num_moves=self.move_count)

        return None

    # ============================================================================
    # State Queries
    # ============================================================================

    def get_pawn_position(self, player: int) -> Position:
        """Get current position of player's pawn.

        Args:
            player: Player number (1 or 2)

        Returns:
            Current pawn position
        """
        if player == 1:
            return self.player1_pos
        elif player == 2:
            return self.player2_pos
        else:
            raise ValueError(f"Invalid player number: {player}")

    def has_wall(self, row: int, col: int, orientation: str) -> bool:
        """Check if a wall exists at position.

        Args:
            row: Row index
            col: Column index
            orientation: 'h' for horizontal, 'v' for vertical

        Returns:
            True if wall exists
        """
        if orientation == 'h':
            return (row, col) in self.h_walls
        elif orientation == 'v':
            return (row, col) in self.v_walls
        else:
            raise ValueError(f"Invalid orientation: {orientation}")

    def is_blocked(self, pos1: Position, pos2: Position) -> bool:
        """Check if movement between two adjacent positions is blocked by wall.

        Args:
            pos1: Starting position
            pos2: Ending position (must be adjacent)

        Returns:
            True if wall blocks movement
        """
        # Use the helper from moves.py
        from .moves import _can_move_between
        return not _can_move_between(
            pos1.to_tuple(),
            pos2.to_tuple(),
            self.h_walls,
            self.v_walls
        )

    def get_opponent(self, player: int) -> int:
        """Get opponent player number.

        Args:
            player: Player number (1 or 2)

        Returns:
            Opponent player number
        """
        return 3 - player  # 1→2, 2→1

    # ============================================================================
    # State Manipulation
    # ============================================================================

    def switch_player(self) -> None:
        """Switch to the other player's turn."""
        self.current_player = self.get_opponent(self.current_player)

    def copy(self) -> 'State':
        """Create a deep copy of the game state.

        Returns:
            New State instance with same state
        """
        import copy as copy_module
        new_state = State.__new__(State)
        new_state.config = self.config
        new_state.player1_pos = Position(self.player1_pos.row, self.player1_pos.col)
        new_state.player2_pos = Position(self.player2_pos.row, self.player2_pos.col)
        new_state.h_walls = self.h_walls.copy()
        new_state.v_walls = self.v_walls.copy()
        new_state.current_player = self.current_player
        new_state.walls_remaining = self.walls_remaining.copy()
        new_state.move_count = self.move_count
        new_state.moves_since_last_wall = self.moves_since_last_wall
        new_state.history = copy_module.copy(self.history)
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        return new_state

    def flip_perspective(self) -> 'State':
        """Create equivalent game state from opponent's perspective.

        Creates a mirror image of the board:
        - Swaps player 1 and player 2
        - Flips board vertically (row i -> row board_size-1-i)
        - Flips wall positions accordingly
        - Makes current player always appear as player 1

        This is useful for training neural networks with a canonical perspective
        where the current player always starts at the bottom.

        Returns:
            New State with flipped perspective
        """
        raise NotImplementedError()

    # ============================================================================
    # Serialization / Representation
    # ============================================================================

    def to_string(self) -> str:
        """Convert game state to human-readable string.

        Returns:
            Multi-line string representation
        """
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of state
        """
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, data: dict, config: GameConfig) -> 'State':
        """Create state from dictionary.

        Args:
            data: Dictionary representation
            config: GameConfig object

        Returns:
            State object
        """
        raise NotImplementedError()

    # ============================================================================
    # Hashing / Equality (for MCTS transposition tables)
    # ============================================================================

    def __eq__(self, other: 'State') -> bool:
        """Check equality with another state (for transposition tables)."""
        raise NotImplementedError()

    def __hash__(self) -> int:
        """Hash the state (for MCTS transposition tables)."""
        raise NotImplementedError()

    # ============================================================================
    # Internal Wall Validation
    # ============================================================================

    def _is_wall_placement_valid(self, row: int, col: int, orientation: str) -> bool:
        """Check if wall placement is locally valid (no overlaps/crosses).

        Does NOT check connectivity - use is_legal_action for full validation.

        Args:
            row: Row index for wall placement
            col: Column index for wall placement
            orientation: 'h' for horizontal, 'v' for vertical

        Returns:
            True if wall can be placed without overlaps/crosses
        """
        from .moves import _walls_overlap

        # Check if this wall already exists
        if orientation == 'h':
            if (row, col) in self.h_walls:
                return False
        else:
            if (row, col) in self.v_walls:
                return False

        # Check for overlaps/crosses with existing walls
        for (r, c) in self.h_walls:
            if _walls_overlap(row, col, orientation, r, c, 'h'):
                return False

        for (r, c) in self.v_walls:
            if _walls_overlap(row, col, orientation, r, c, 'v'):
                return False

        return True


def create_initial_state(config: GameConfig) -> State:
    """Create initial game state for a new game.

    Args:
        config: GameConfig object

    Returns:
        State with initial board setup (pawns at starting positions, no walls)
    """
    return State(config)

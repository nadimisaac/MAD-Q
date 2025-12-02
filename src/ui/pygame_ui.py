"""Pygame-based GUI for Quoridor game.

Provides an interactive graphical interface with:
- Visual board with grid and coordinates
- Notation display everywhere for verification
- Mouse-based move selection
- Move history panel
- Game state information
"""

import pygame
from typing import Optional, Tuple, List, Callable
from src.game.state import State
from src.notation.converter import position_to_notation


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
GREEN = (50, 200, 50)
LIGHT_GREEN = (150, 255, 150, 100)
GOLD = (255, 215, 0)

# Player colors
LIGHT_BROWN = (210, 180, 140)  # Tan/Light brown for Player 1
DARK_BROWN = (101, 67, 33)     # Dark brown for Player 2

# Wall color
WALL_COLOR = (70, 47, 23)


class PygameUI:
    """Pygame-based UI for Quoridor game."""

    def __init__(self, state: State, window_size: Tuple[int, int] = (1200, 800)):
        """Initialize Pygame UI.

        Args:
            state: Initial game state
            window_size: Window dimensions (width, height)
        """
        pygame.init()

        self.state = state
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Quoridor")

        # Layout constants
        self.sidebar_width = 350
        self.board_area_width = window_size[0] - self.sidebar_width
        self.board_margin = 60  # Margin for labels

        # Calculate cell size to fit the board
        available_size = min(
            self.board_area_width - 2 * self.board_margin,
            window_size[1] - 2 * self.board_margin
        )
        self.cell_size = available_size // state.config.board_size
        self.board_size = self.cell_size * state.config.board_size

        # Center the board in the left area
        self.board_x = (self.board_area_width - self.board_size) // 2
        self.board_y = (window_size[1] - self.board_size) // 2

        # Wall dimensions
        self.wall_thickness = 8
        self.wall_length = self.cell_size * 2

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        self.font_tiny = pygame.font.Font(None, 18)

        # UI state
        self.selected_cell = None
        self.wall_mode = False  # False = pawn mode, True = wall mode
        self.wall_orientation = 'h'  # 'h' or 'v'
        self.hover_cell = None
        self.move_history = []
        self.history_scroll = 0
        self.show_legal_moves = False

        # Clock for frame rate
        self.clock = pygame.time.Clock()

    def run(self, on_move: Optional[Callable[[str, Tuple[int, int]], None]] = None):
        """Run the UI main loop.

        Args:
            on_move: Callback function called when a move is made
                     Signature: on_move(action_type, position)
        """
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos, on_move)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_hover(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self._handle_keypress(event.key)

            # Render
            self.render()
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()

    def render(self):
        """Render the entire UI."""
        self.screen.fill(WHITE)

        # Draw board components
        self._render_board_background()
        self._render_grid()
        self._render_coordinates()
        self._render_walls()
        self._render_legal_moves_overlay()
        self._render_pawns()
        self._render_hover_tooltip()

        # Draw sidebar
        self._render_sidebar()

    def _render_board_background(self):
        """Draw the board background."""
        board_rect = pygame.Rect(
            self.board_x,
            self.board_y,
            self.board_size,
            self.board_size
        )
        pygame.draw.rect(self.screen, LIGHT_GRAY, board_rect)
        pygame.draw.rect(self.screen, BLACK, board_rect, 2)

    def _render_grid(self):
        """Draw the grid lines."""
        for i in range(self.state.config.board_size + 1):
            # Vertical lines
            x = self.board_x + i * self.cell_size
            pygame.draw.line(
                self.screen,
                GRAY,
                (x, self.board_y),
                (x, self.board_y + self.board_size),
                1
            )

            # Horizontal lines
            y = self.board_y + i * self.cell_size
            pygame.draw.line(
                self.screen,
                GRAY,
                (self.board_x, y),
                (self.board_x + self.board_size, y),
                1
            )

    def _render_coordinates(self):
        """Draw coordinate labels (a-i, 1-9) around the board."""
        board_size = self.state.config.board_size

        # Column labels (a-i) - top and bottom
        for col in range(board_size):
            letter = chr(ord('a') + col)
            x = self.board_x + col * self.cell_size + self.cell_size // 2

            # Top label
            text = self.font_medium.render(letter, True, BLACK)
            text_rect = text.get_rect(center=(x, self.board_y - 30))
            self.screen.blit(text, text_rect)

            # Bottom label
            text_rect = text.get_rect(center=(x, self.board_y + self.board_size + 30))
            self.screen.blit(text, text_rect)

        # Row labels (1-9) - left and right
        for row in range(board_size):
            number = str(row + 1)
            y = self.board_y + (board_size - 1 - row) * self.cell_size + self.cell_size // 2

            # Left label
            text = self.font_medium.render(number, True, BLACK)
            text_rect = text.get_rect(center=(self.board_x - 30, y))
            self.screen.blit(text, text_rect)

            # Right label
            text_rect = text.get_rect(center=(self.board_x + self.board_size + 30, y))
            self.screen.blit(text, text_rect)

    def _render_walls(self):
        """Draw all placed walls."""
        # Horizontal walls
        for wall_row, wall_col in self.state.h_walls:
            x = self.board_x + wall_col * self.cell_size
            y = self.board_y + (self.state.config.board_size - 1 - wall_row) * self.cell_size

            # Draw wall (2 cells wide)
            wall_rect = pygame.Rect(
                x,
                y - self.wall_thickness // 2,
                self.wall_length,
                self.wall_thickness
            )
            pygame.draw.rect(self.screen, WALL_COLOR, wall_rect)
            pygame.draw.rect(self.screen, BLACK, wall_rect, 1)

        # Vertical walls
        for wall_row, wall_col in self.state.v_walls:
            # Vertical wall at (r, c) spans rows r and r+1
            # In screen coords (y=0 at top), row r+1 is ABOVE row r
            x = self.board_x + (wall_col + 1) * self.cell_size

            # Calculate y position for the top of the wall (row r+1 in game coords)
            # Row r+1 has screen y = board_y + (board_size - 1 - (r+1)) * cell_size
            #                       = board_y + (board_size - 2 - r) * cell_size
            y_top = self.board_y + (self.state.config.board_size - 2 - wall_row) * self.cell_size

            # Draw wall (2 cells tall, from row r+1 down to row r)
            wall_rect = pygame.Rect(
                x - self.wall_thickness // 2,
                y_top,
                self.wall_thickness,
                self.wall_length
            )
            pygame.draw.rect(self.screen, WALL_COLOR, wall_rect)
            pygame.draw.rect(self.screen, BLACK, wall_rect, 1)

    def _render_pawns(self):
        """Draw player pawns."""
        # Player 1 (light brown)
        p1_x = self.board_x + self.state.player1_pos.col * self.cell_size + self.cell_size // 2
        p1_y = self.board_y + (self.state.config.board_size - 1 - self.state.player1_pos.row) * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, LIGHT_BROWN, (p1_x, p1_y), self.cell_size // 3)
        pygame.draw.circle(self.screen, BLACK, (p1_x, p1_y), self.cell_size // 3, 3)

        # Draw "1" on player 1 pawn
        text = self.font_medium.render("1", True, BLACK)
        text_rect = text.get_rect(center=(p1_x, p1_y))
        self.screen.blit(text, text_rect)

        # Player 2 (dark brown)
        p2_x = self.board_x + self.state.player2_pos.col * self.cell_size + self.cell_size // 2
        p2_y = self.board_y + (self.state.config.board_size - 1 - self.state.player2_pos.row) * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, DARK_BROWN, (p2_x, p2_y), self.cell_size // 3)
        pygame.draw.circle(self.screen, BLACK, (p2_x, p2_y), self.cell_size // 3, 3)

        # Draw "2" on player 2 pawn
        text = self.font_medium.render("2", True, WHITE)
        text_rect = text.get_rect(center=(p2_x, p2_y))
        self.screen.blit(text, text_rect)

    def _render_legal_moves_overlay(self):
        """Highlight legal moves if a cell is selected or show_legal_moves is True."""
        if not self.show_legal_moves and not self.selected_cell:
            return

        if self.wall_mode:
            # Show legal wall placements
            legal_walls = self.state.get_legal_wall_placements(self.wall_orientation)
            for wall_row, wall_col in legal_walls:
                if self.wall_orientation == 'h':
                    x = self.board_x + wall_col * self.cell_size
                    y = self.board_y + (self.state.config.board_size - 1 - wall_row) * self.cell_size
                    highlight_rect = pygame.Rect(
                        x,
                        y - self.wall_thickness,
                        self.wall_length,
                        self.wall_thickness * 2
                    )
                else:  # 'v'
                    x = self.board_x + (wall_col + 1) * self.cell_size
                    # Same calculation as in _render_walls for vertical walls
                    y_top = self.board_y + (self.state.config.board_size - 2 - wall_row) * self.cell_size
                    highlight_rect = pygame.Rect(
                        x - self.wall_thickness,
                        y_top,
                        self.wall_thickness * 2,
                        self.wall_length
                    )

                # Draw semi-transparent overlay
                surf = pygame.Surface((highlight_rect.width, highlight_rect.height))
                surf.set_alpha(100)
                surf.fill(GREEN)
                self.screen.blit(surf, highlight_rect.topleft)
        else:
            # Show legal pawn moves
            legal_moves = self.state.get_legal_pawn_moves()
            for move_row, move_col in legal_moves:
                x = self.board_x + move_col * self.cell_size
                y = self.board_y + (self.state.config.board_size - 1 - move_row) * self.cell_size

                # Draw semi-transparent green overlay
                surf = pygame.Surface((self.cell_size, self.cell_size))
                surf.set_alpha(100)
                surf.fill(GREEN)
                self.screen.blit(surf, (x, y))

                # Draw notation in the cell
                notation = position_to_notation(move_row, move_col, is_wall=False)
                text = self.font_small.render(notation, True, DARK_GRAY)
                text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size - 15))
                self.screen.blit(text, text_rect)

    def _render_hover_tooltip(self):
        """Show notation tooltip on hover."""
        if self.hover_cell is None:
            return

        row, col = self.hover_cell
        notation = position_to_notation(row, col, is_wall=False)

        # Get mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Draw tooltip
        text = self.font_small.render(notation, True, BLACK)
        padding = 5
        tooltip_rect = pygame.Rect(
            mouse_x + 15,
            mouse_y - 25,
            text.get_width() + padding * 2,
            text.get_height() + padding * 2
        )

        pygame.draw.rect(self.screen, GOLD, tooltip_rect)
        pygame.draw.rect(self.screen, BLACK, tooltip_rect, 2)
        self.screen.blit(text, (tooltip_rect.x + padding, tooltip_rect.y + padding))

    def _render_sidebar(self):
        """Draw the information sidebar."""
        sidebar_x = self.board_area_width

        # Background
        sidebar_rect = pygame.Rect(sidebar_x, 0, self.sidebar_width, self.window_size[1])
        pygame.draw.rect(self.screen, LIGHT_GRAY, sidebar_rect)
        pygame.draw.line(self.screen, BLACK, (sidebar_x, 0), (sidebar_x, self.window_size[1]), 2)

        y_offset = 20

        # Title
        title = self.font_large.render("Quoridor", True, BLACK)
        self.screen.blit(title, (sidebar_x + 20, y_offset))
        y_offset += 50

        # Current player
        current_player_text = f"Current Player: {self.state.current_player}"
        color = LIGHT_BROWN if self.state.current_player == 1 else DARK_BROWN
        text = self.font_medium.render(current_player_text, True, color)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 35

        # Player positions with notation
        p1_notation = position_to_notation(self.state.player1_pos.row, self.state.player1_pos.col)
        p2_notation = position_to_notation(self.state.player2_pos.row, self.state.player2_pos.col)

        text = self.font_small.render(f"Player 1: {p1_notation}", True, LIGHT_BROWN)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 25

        text = self.font_small.render(f"Player 2: {p2_notation}", True, DARK_BROWN)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 35

        # Walls remaining
        text = self.font_small.render("Walls Remaining:", True, BLACK)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 25

        text = self.font_small.render(f"  Player 1: {self.state.walls_remaining[1]}", True, LIGHT_BROWN)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 25

        text = self.font_small.render(f"  Player 2: {self.state.walls_remaining[2]}", True, DARK_BROWN)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 35

        # Move count
        text = self.font_small.render(f"Move: {self.state.move_count}", True, BLACK)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 35

        # Game status
        if self.state.game_over:
            if self.state.winner is not None:
                winner_color = LIGHT_BROWN if self.state.winner == 1 else DARK_BROWN
                status_text = f"PLAYER {self.state.winner} WINS!"
                text = self.font_large.render(status_text, True, winner_color)
                self.screen.blit(text, (sidebar_x + 20, y_offset))
                y_offset += 50
            else:
                text = self.font_medium.render("GAME DRAW", True, BLACK)
                self.screen.blit(text, (sidebar_x + 20, y_offset))
                y_offset += 40
        else:
            # Mode indicator (only show when game is ongoing)
            mode_text = "WALL MODE" if self.wall_mode else "PAWN MODE"
            mode_color = WALL_COLOR if self.wall_mode else GREEN
            text = self.font_medium.render(mode_text, True, mode_color)
            self.screen.blit(text, (sidebar_x + 20, y_offset))
            y_offset += 30

        if self.wall_mode and not self.state.game_over:
            orient_text = f"Orientation: {'Horizontal' if self.wall_orientation == 'h' else 'Vertical'}"
            text = self.font_small.render(orient_text, True, DARK_GRAY)
            self.screen.blit(text, (sidebar_x + 20, y_offset))
            y_offset += 30

        # Controls
        y_offset += 10
        text = self.font_small.render("Controls:", True, BLACK)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 25

        controls = [
            "Click: Make move",
            "W: Toggle wall mode",
            "R: Rotate wall (H/V)",
            "L: Show legal moves",
            "U: Undo move",
            "ESC: Quit"
        ]

        for control in controls:
            text = self.font_tiny.render(control, True, DARK_GRAY)
            self.screen.blit(text, (sidebar_x + 25, y_offset))
            y_offset += 20

        # Move history
        y_offset += 20
        text = self.font_small.render("Move History:", True, BLACK)
        self.screen.blit(text, (sidebar_x + 20, y_offset))
        y_offset += 25

        # Draw column headers
        col1_x = sidebar_x + 25
        col2_x = sidebar_x + 185
        header1 = self.font_tiny.render("P1", True, LIGHT_BROWN)
        header2 = self.font_tiny.render("P2", True, DARK_BROWN)
        self.screen.blit(header1, (col1_x + 30, y_offset))
        self.screen.blit(header2, (col2_x, y_offset))
        y_offset += 20

        # Draw move history in 2-column format (last 15 moves = ~7-8 pairs)
        # Group moves into pairs (Player 1, Player 2)
        max_rows = 7  # Show last 7 move pairs
        start_idx = max(0, len(self.move_history) - (max_rows * 2))
        history_display = self.move_history[start_idx:]

        move_number = (start_idx // 2) + 1
        for i in range(0, len(history_display), 2):
            # Move number
            move_num_text = self.font_tiny.render(f"{move_number}.", True, DARK_GRAY)
            self.screen.blit(move_num_text, (col1_x, y_offset))

            # Player 1's move
            p1_move = history_display[i]
            p1_text = self.font_tiny.render(p1_move, True, LIGHT_BROWN)
            self.screen.blit(p1_text, (col1_x + 30, y_offset))

            # Player 2's move (if exists)
            if i + 1 < len(history_display):
                p2_move = history_display[i + 1]
                p2_text = self.font_tiny.render(p2_move, True, DARK_BROWN)
                self.screen.blit(p2_text, (col2_x, y_offset))

            y_offset += 18
            move_number += 1

    def _handle_click(self, pos: Tuple[int, int], on_move: Optional[Callable]):
        """Handle mouse click events.

        Args:
            pos: Mouse position (x, y)
            on_move: Callback for when a move is made
        """
        # Don't accept moves if game is over
        if self.state.game_over:
            return

        cell = self._get_cell_from_pos(pos)

        if cell is None:
            return

        row, col = cell

        if self.wall_mode:
            # Wall placement mode
            if self.state.walls_remaining[self.state.current_player] == 0:
                return  # No walls left

            # Check if this wall placement is legal
            legal_walls = self.state.get_legal_wall_placements(self.wall_orientation)
            if (row, col) in legal_walls:
                # Make the move
                if on_move:
                    action_type = 'h_wall' if self.wall_orientation == 'h' else 'v_wall'
                    on_move(action_type, (row, col))

                    # Record notation
                    notation = position_to_notation(row, col, is_wall=True)
                    notation += self.wall_orientation
                    self.move_history.append(notation)
        else:
            # Pawn movement mode
            legal_moves = self.state.get_legal_pawn_moves()
            if (row, col) in legal_moves:
                # Make the move
                if on_move:
                    on_move('move', (row, col))

                    # Record notation
                    notation = position_to_notation(row, col, is_wall=False)
                    self.move_history.append(notation)

    def _handle_hover(self, pos: Tuple[int, int]):
        """Handle mouse hover events.

        Args:
            pos: Mouse position (x, y)
        """
        self.hover_cell = self._get_cell_from_pos(pos)

    def _handle_keypress(self, key: int):
        """Handle keyboard events.

        Args:
            key: Pygame key constant
        """
        if key == pygame.K_w:
            # Toggle wall mode
            self.wall_mode = not self.wall_mode
        elif key == pygame.K_r:
            # Rotate wall orientation
            self.wall_orientation = 'v' if self.wall_orientation == 'h' else 'h'
        elif key == pygame.K_l:
            # Toggle legal moves display
            self.show_legal_moves = not self.show_legal_moves
        elif key == pygame.K_ESCAPE:
            # Quit
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def _get_cell_from_pos(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert screen position to board cell coordinates.

        Args:
            pos: Screen position (x, y)

        Returns:
            (row, col) tuple in board coordinates, or None if outside board
        """
        x, y = pos

        # Check if within board bounds
        if (x < self.board_x or x >= self.board_x + self.board_size or
            y < self.board_y or y >= self.board_y + self.board_size):
            return None

        # Convert to cell coordinates
        col = (x - self.board_x) // self.cell_size
        row = self.state.config.board_size - 1 - ((y - self.board_y) // self.cell_size)

        # Clamp to valid range
        row = max(0, min(row, self.state.config.board_size - 1))
        col = max(0, min(col, self.state.config.board_size - 1))

        return (row, col)

    def update_state(self, new_state: State):
        """Update the displayed game state.

        Args:
            new_state: New game state to display
        """
        self.state = new_state

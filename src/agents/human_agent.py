"""Human agent that prompts for input via command line."""

from typing import Tuple
from .base_agent import BaseAgent


class HumanAgent(BaseAgent):
    """Agent that prompts human player for moves via CLI.

    Displays the board and legal moves, then prompts for input in notation format.
    """

    def __init__(self, player_number: int):
        """Initialize human agent.

        Args:
            player_number: Player number (1 or 2)
        """
        super().__init__(player_number)

    def select_action(self, game_state) -> Tuple[str, Tuple[int, int]]:
        """Prompt human for action input.

        Args:
            game_state: Current GameState object

        Returns:
            (action_type, position) tuple based on human input

        Raises:
            ValueError: If input is invalid
        """
        self._display_game_state(game_state)
        self._display_legal_moves(game_state)

        while True:
            try:
                user_input = input(f"\nPlayer {self.player_number}, enter your move: ").strip()

                if not user_input:
                    print("Please enter a move.")
                    continue

                # Parse the input
                action_type, position = self._parse_input(user_input)

                # Verify it's a legal move
                legal_moves = game_state.get_legal_moves()
                if (action_type, position) not in legal_moves:
                    print(f"Illegal move: {user_input}")
                    print("Please choose from the legal moves shown above.")
                    continue

                return (action_type, position)

            except ValueError as e:
                print(f"Invalid input: {e}")
                print("Please use notation like: e5 (move), e4h (horizontal wall), e4v (vertical wall)")
            except KeyboardInterrupt:
                print("\nGame interrupted.")
                raise
            except Exception as e:
                print(f"Error: {e}")

    def _display_game_state(self, game_state) -> None:
        """Display current game state to player.

        Args:
            game_state: Current GameState object
        """
        from ..utils.visualization import render_board_ascii

        print("\n" + "=" * 50)
        print(render_board_ascii(game_state))
        print(f"\nPlayer {game_state.current_player}'s turn")
        print(f"Walls remaining - Player 1: {game_state.walls_remaining[1]}, Player 2: {game_state.walls_remaining[2]}")
        print(f"Move count: {game_state.move_count}")

    def _display_legal_moves(self, game_state) -> None:
        """Display legal moves to player.

        Args:
            game_state: Current GameState object
        """
        from ..notation.converter import action_to_notation

        legal_moves = game_state.get_legal_moves()

        # Separate moves by type
        pawn_moves = [action_to_notation(act, pos) for act, pos in legal_moves if act == 'move']
        h_walls = [action_to_notation(act, pos) for act, pos in legal_moves if act == 'h_wall']
        v_walls = [action_to_notation(act, pos) for act, pos in legal_moves if act == 'v_wall']

        print("\nLegal moves:")
        if pawn_moves:
            print(f"  Pawn moves: {', '.join(pawn_moves)}")
        if h_walls and game_state.walls_remaining[game_state.current_player] > 0:
            # Show just a sample if there are many
            if len(h_walls) > 10:
                print(f"  Horizontal walls: {', '.join(h_walls[:10])}... ({len(h_walls)} total)")
            else:
                print(f"  Horizontal walls: {', '.join(h_walls)}")
        if v_walls and game_state.walls_remaining[game_state.current_player] > 0:
            if len(v_walls) > 10:
                print(f"  Vertical walls: {', '.join(v_walls[:10])}... ({len(v_walls)} total)")
            else:
                print(f"  Vertical walls: {', '.join(v_walls)}")

    def _parse_input(self, input_str: str) -> Tuple[str, Tuple[int, int]]:
        """Parse user input into action.

        Args:
            input_str: User input string (e.g., "e5", "e4h", "e4v")

        Returns:
            (action_type, position) tuple

        Raises:
            ValueError: If input format is invalid
        """
        from ..notation.converter import notation_to_action

        return notation_to_action(input_str)

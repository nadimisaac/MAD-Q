"""Elo rating system for agent performance evaluation."""

from typing import Dict, List, Tuple
import math


class EloRatingSystem:
    """Elo rating system for tracking agent performance."""

    def __init__(self, k: int = 32, initial_rating: int = 1500):
        """
        Initialize Elo rating system.

        Args:
            k: K-factor for rating updates (default: 32)
            initial_rating: Initial rating for new players (default: 1500)
        """
        self.k = k
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}

    def get_rating(self, player: str) -> float:
        """Get current rating for a player."""
        if player not in self.ratings:
            self.ratings[player] = self.initial_rating
        return self.ratings[player]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.

        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B

        Returns:
            Expected score (probability of A winning)
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        player_a: str,
        player_b: str,
        actual_score: float
    ) -> Tuple[float, float]:
        """
        Update ratings based on game result.

        Args:
            player_a: Name of player A
            player_b: Name of player B
            actual_score: Actual score (1.0 for A win, 0.5 for draw, 0.0 for A loss)

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        # Update ratings
        new_rating_a = rating_a + self.k * (actual_score - expected_a)
        new_rating_b = rating_b + self.k * ((1.0 - actual_score) - expected_b)

        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b

        return new_rating_a, new_rating_b

    def process_game(
        self,
        player_a: str,
        player_b: str,
        winner: str | None
    ) -> Tuple[float, float]:
        """
        Process a game result and update ratings.

        Args:
            player_a: Name of player A
            player_b: Name of player B
            winner: Name of winner (None for draw)

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        if winner is None:
            actual_score = 0.5  # Draw
        elif winner == player_a:
            actual_score = 1.0  # A wins
        elif winner == player_b:
            actual_score = 0.0  # B wins
        else:
            raise ValueError(f"Winner must be {player_a}, {player_b}, or None")

        return self.update_ratings(player_a, player_b, actual_score)

    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Get all players ranked by rating.

        Returns:
            List of (player_name, rating) tuples, sorted by rating (descending)
        """
        rankings = [(player, rating) for player, rating in self.ratings.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


def compute_ratings_from_tournament(
    results: List[Dict],
    player1_key: str = 'player1',
    player2_key: str = 'player2',
    winner_key: str = 'winner',
    k: int = 32,
    initial_rating: int = 1500
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Compute Elo ratings from a tournament's results.

    Args:
        results: List of game result dictionaries
        player1_key: Key for player 1 name in result dict
        player2_key: Key for player 2 name in result dict
        winner_key: Key for winner name in result dict (None for draw)
        k: K-factor for rating system
        initial_rating: Initial rating

    Returns:
        Tuple of (final_ratings_dict, rating_history_list)
            rating_history_list: List of rating snapshots after each game
    """
    elo = EloRatingSystem(k=k, initial_rating=initial_rating)
    history = []

    for game_result in results:
        player1 = game_result[player1_key]
        player2 = game_result[player2_key]
        winner = game_result.get(winner_key)

        elo.process_game(player1, player2, winner)

        # Save snapshot of current ratings
        history.append(dict(elo.ratings))

    return dict(elo.ratings), history


def track_rating_evolution(
    game_sequence: List[Tuple[str, str, str | None]],
    k: int = 32,
    initial_rating: int = 1500
) -> Dict[str, List[float]]:
    """
    Track rating evolution over a sequence of games.

    Args:
        game_sequence: List of (player1, player2, winner) tuples
        k: K-factor
        initial_rating: Initial rating

    Returns:
        Dictionary mapping player names to lists of ratings over time
    """
    elo = EloRatingSystem(k=k, initial_rating=initial_rating)

    # Initialize tracking dict
    evolution: Dict[str, List[float]] = {}

    for player1, player2, winner in game_sequence:
        # Ensure players are tracked
        if player1 not in evolution:
            evolution[player1] = [initial_rating]
        if player2 not in evolution:
            evolution[player2] = [initial_rating]

        # Process game
        new_rating1, new_rating2 = elo.process_game(player1, player2, winner)

        # Record new ratings
        evolution[player1].append(new_rating1)
        evolution[player2].append(new_rating2)

        # For players not in this game, maintain their current rating
        for player in evolution:
            if player != player1 and player != player2:
                evolution[player].append(evolution[player][-1])

    return evolution


def rating_difference_to_win_probability(rating_diff: float) -> float:
    """
    Convert rating difference to win probability.

    Args:
        rating_diff: Rating difference (player1 - player2)

    Returns:
        Win probability for player1
    """
    return 1.0 / (1.0 + math.pow(10, -rating_diff / 400.0))


def win_probability_to_rating_difference(win_prob: float) -> float:
    """
    Convert win probability to rating difference.

    Args:
        win_prob: Win probability (must be between 0 and 1, exclusive)

    Returns:
        Rating difference that would produce this win probability
    """
    if win_prob <= 0 or win_prob >= 1:
        raise ValueError("Win probability must be between 0 and 1 (exclusive)")

    return -400.0 * math.log10((1.0 / win_prob) - 1.0)

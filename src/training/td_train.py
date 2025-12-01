from typing import Optional
import numpy as np


class TDTrain:
    def __init__(
        self,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        initial_weights: Optional[np.ndarray] = None
    ):
        """
        Args:
            learning_rate: alpha, step size for gradient descent
            discount_factor: gamma, how much to value future rewards (used in td error)
            initial_weights: Starting weights [path, progress, wall], 
                        defaults to [12.0, 1.5, 2.0]
        """

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.weights = initial_weights if initial_weights is not None else np.array([12.0, 1.5, 2.0])
    
    def compute_value(self, features: np.ndarray) -> float:
        return features @ self.weights

    def compute_td_error(
        self, 
        current_features: np.ndarray,
        next_features: np.ndarray,
        reward: float
    ) -> float:
        
        # TD error: reward + gamma * V(phi(s')) - V(phi(s))
        
        # We need to calculate the value at the current state and at the next (successor) state
        curr_value = self.compute_value(current_features)
        next_value = self.compute_value(next_features)

        return reward + self.discount_factor * next_value - curr_value
    
    def update_weights(
        self, 
        current_features: np.ndarray,
        td_error: float
    ) -> None:

        # w = w + lr * td_error * features
        self.weights = self.weights + self.learning_rate * td_error * current_features
    
    def get_weights(self) -> np.ndarray:
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights.copy()

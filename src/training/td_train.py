from typing import Optional
import numpy as np


class TDTrain:
    def __init__(
        self,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        initial_weights: Optional[np.ndarray] = None,
        num_features: int = 3,
    ):
        """
        Args:
            learning_rate: alpha, step size for gradient descent
            discount_factor: gamma, how much to value future rewards (used in td error)
            initial_weights: Starting weights [path, progress, wall], 
                        defaults to random initialization from uniform[-0.01, 0.01]
        """

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        if initial_weights is not None:
            self.weights = initial_weights
        else:
            # random initial weights sampled from uniform distribution between -0.01, 0.01
            rand = np.random.default_rng()
            self.weights = rand.uniform(-0.01, 0.01, size=num_features)
    
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

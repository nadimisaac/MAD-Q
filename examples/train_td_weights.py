#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.training.td_train import TDTrain
from src.agents.minimax_agent import MinimaxAgent
from src.game import State


def play_training_game(config, depth, trainer):
    state = State(config)
    
    while not state.game_over:
        player = state.current_player
        w = trainer.get_weights()
        
        agent = MinimaxAgent(
            player_number=player,
            depth=depth,
            path_weight=w[0],
            progress_weight=w[1],
            wall_weight=w[2],
        )
        
        # Features before move, from this player's perspective

        curr_features = agent.extract_td_features(state)
        action_type, position = agent.select_action(state)
        next_state = state.make_move(action_type, position)
        
        if next_state.game_over:
            if next_state.winner == player:
                reward = 1.0
            elif next_state.winner is None:
                reward = 0.0
            else:
                reward = -1.0
        else:
            reward = 0.0
        
        # Features after move, from this player's perspective
        if next_state.game_over:
            next_features = np.zeros_like(curr_features)
        else:
            next_features = agent.extract_td_features(next_state)
        
        # TD update for this player's move
        td_error = trainer.compute_td_error(curr_features, next_features, reward)
        trainer.update_weights(curr_features, td_error)        
        state = next_state
    
    return state.winner

def train(
    num_episodes: int = 100,
    learning_rate: float = 0.01,
    discount_factor: float = 0.99,
    depth: int = 2,
    config_name: str = "standard",
    verbose: bool = True
):
    """
    Main TD Training Loop
    
    Args:
        num_episodes: # of games played during training
        learning_rate: alpha used in TD updates
        discount_factor: gamma used for reward discount factor
        depth: Minimax search depth for agents
        config_name: Game configuration to use
        verbose: Print progress
    """
    trainer = TDTrain(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
    )
    
    # Load game config
    from src.game.config import load_game_config
    config = load_game_config(config_name)
    
    # Track statistics
    win_counts = {1: 0, 2: 0, None: 0}
    
    if verbose:
        print("=" * 60)
        print("TD(0) TRAINING - Online Self-Play")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"Learning rate (α): {learning_rate}")
        print(f"Discount factor (γ): {discount_factor}")
        print(f"Search depth: {depth}")
        print(f"Initial weights: {trainer.get_weights()}")
        print("=" * 60)
        print()
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        winner = play_training_game(config, depth, trainer)
        win_counts[winner] += 1
        
        if verbose and episode % 10 == 0:
            weights = trainer.get_weights()
            print(f"Episode {episode:3d} | "
                  f"Weights: [{weights[0]:6.2f}, {weights[1]:6.2f}, {weights[2]:6.2f}] | "
                  f"P1: {win_counts[1]:2d} P2: {win_counts[2]:2d} Draw: {win_counts[None]:2d}")
    
    # Final results
    final_weights = trainer.get_weights()
    if verbose:
        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final weights: [{final_weights[0]:.3f}, {final_weights[1]:.3f}, {final_weights[2]:.3f}]")
        print(f"Win distribution - P1: {win_counts[1]}, P2: {win_counts[2]}, Draws: {win_counts[None]}")
        print("=" * 60)
    
    return final_weights


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Minimax evaluation weights using TD(0)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training games")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate (alpha)")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--depth", type=int, default=2, help="Minimax search depth")
    parser.add_argument("--config", type=str, default="standard", choices=["standard", "small", "tiny"])
    
    args = parser.parse_args()
    
    trained_weights = train(
        num_episodes=args.episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        depth=args.depth,
        config_name=args.config,
        verbose=True
    )
    
    print(f"\nTrained weights: {trained_weights}")
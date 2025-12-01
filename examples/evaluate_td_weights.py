#!/usr/bin/env python3
"""Evaluate TD-trained weights against baseline opponents."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.agents.baseline_agent import BaselineAgent
from src.game import State
from src.game.config import load_game_config


def play_match(agent1, agent2, config, num_games=20):
    """
    Play multiple games and return win statistics.
    
    Args:
        agent1: First agent (plays as P1 half the time, P2 half the time)
        agent2: Second agent
        config: Game configuration
        num_games: Number of games to play
        
    Returns:
        dict: Statistics {agent1_wins, agent2_wins, draws}
    """
    stats = {1: 0, 2: 0, None: 0}
    
    for game_num in range(num_games):
        # Alternate who plays as P1/P2
        if game_num % 2 == 0:
            agents = {1: agent1, 2: agent2}
            agent1_player = 1
        else:
            agents = {1: agent2, 2: agent1}
            agent1_player = 2
        
        state = State(config)
        
        while not state.game_over:
            current_agent = agents[state.current_player]
            action_type, position = current_agent.select_action(state)
            state = state.make_move(action_type, position)
        
        # Record from agent1's perspective
        if state.winner == agent1_player:
            stats[1] += 1
        elif state.winner == 3 - agent1_player:
            stats[2] += 1
        else:
            stats[None] += 1
    
    return stats


def evaluate_weights(
    trained_weights,
    default_weights=np.array([12.0, 1.5, 2.0]),
    config_name="small",
    depth=2,
    num_games=20
):
    """
    Evaluate trained weights vs default weights against opponents.
    
    Args:
        trained_weights: Weights learned from TD training
        default_weights: Original hand-crafted weights
        config_name: Game configuration
        depth: Minimax search depth
        num_games: Games to play per matchup
    """
    config = load_game_config(config_name)
    
    print("=" * 70)
    print("EVALUATING TD-TRAINED WEIGHTS")
    print("=" * 70)
    print(f"Trained weights: {trained_weights}")
    print(f"Default weights: {default_weights}")
    print(f"Configuration: {config_name}, Depth: {depth}, Games per matchup: {num_games}")
    print("=" * 70)
    print()
    
    # Create agents with different weight configurations
    trained_agent = MinimaxAgent(
        player_number=1,
        depth=depth,
        path_weight=trained_weights[0],
        progress_weight=trained_weights[1],
        wall_weight=trained_weights[2]
    )
    
    default_agent = MinimaxAgent(
        player_number=1,
        depth=depth,
        path_weight=default_weights[0],
        progress_weight=default_weights[1],
        wall_weight=default_weights[2]
    )
    
    # Test 1: Trained vs Random
    print("🎲 TEST 1: Trained Weights vs Random Agent")
    print("-" * 70)
    random_agent = RandomAgent(player_number=2)
    stats = play_match(trained_agent, random_agent, config, num_games)
    print(f"Trained wins: {stats[1]}/{num_games} ({100*stats[1]/num_games:.1f}%)")
    print(f"Random wins:  {stats[2]}/{num_games} ({100*stats[2]/num_games:.1f}%)")
    print(f"Draws:        {stats[None]}/{num_games} ({100*stats[None]/num_games:.1f}%)")
    print()
    
    # Test 2: Default vs Random
    print("🎲 TEST 2: Default Weights vs Random Agent")
    print("-" * 70)
    stats = play_match(default_agent, random_agent, config, num_games)
    print(f"Default wins: {stats[1]}/{num_games} ({100*stats[1]/num_games:.1f}%)")
    print(f"Random wins:  {stats[2]}/{num_games} ({100*stats[2]/num_games:.1f}%)")
    print(f"Draws:        {stats[None]}/{num_games} ({100*stats[None]/num_games:.1f}%)")
    print()
    
    # Test 3: Trained vs Baseline
    print("🤖 TEST 3: Trained Weights vs Baseline Agent")
    print("-" * 70)
    baseline_agent = BaselineAgent(player_number=2)
    stats = play_match(trained_agent, baseline_agent, config, num_games)
    print(f"Trained wins:  {stats[1]}/{num_games} ({100*stats[1]/num_games:.1f}%)")
    print(f"Baseline wins: {stats[2]}/{num_games} ({100*stats[2]/num_games:.1f}%)")
    print(f"Draws:         {stats[None]}/{num_games} ({100*stats[None]/num_games:.1f}%)")
    print()
    
    # Test 4: Default vs Baseline
    print("🤖 TEST 4: Default Weights vs Baseline Agent")
    print("-" * 70)
    stats = play_match(default_agent, baseline_agent, config, num_games)
    print(f"Default wins:  {stats[1]}/{num_games} ({100*stats[1]/num_games:.1f}%)")
    print(f"Baseline wins: {stats[2]}/{num_games} ({100*stats[2]/num_games:.1f}%)")
    print(f"Draws:         {stats[None]}/{num_games} ({100*stats[None]/num_games:.1f}%)")
    print()
    
    # Test 5: Head-to-head
    print("⚔️  TEST 5: Trained Weights vs Default Weights (Head-to-Head)")
    print("-" * 70)
    stats = play_match(trained_agent, default_agent, config, num_games)
    print(f"Trained wins: {stats[1]}/{num_games} ({100*stats[1]/num_games:.1f}%)")
    print(f"Default wins: {stats[2]}/{num_games} ({100*stats[2]/num_games:.1f}%)")
    print(f"Draws:        {stats[None]}/{num_games} ({100*stats[None]/num_games:.1f}%)")
    print()
    
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate TD-trained weights")
    parser.add_argument("--trained", type=float, nargs=3, required=True,
                       help="Trained weights [path, progress, wall]")
    parser.add_argument("--default", type=float, nargs=3, default=[12.0, 1.5, 2.0],
                       help="Default weights (default: [12.0, 1.5, 2.0])")
    parser.add_argument("--config", type=str, default="small", 
                       choices=["standard", "small", "tiny"])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--games", type=int, default=20,
                       help="Number of games per matchup")
    
    args = parser.parse_args()
    
    trained = np.array(args.trained)
    default = np.array(args.default)
    
    evaluate_weights(trained, default, args.config, args.depth, args.games)


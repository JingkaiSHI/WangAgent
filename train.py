"""
Main training script for Go reinforcement learning agent
"""
import os
import time
from my_player3 import QLearningAgent, X_TYPE
from training import progressive_training, DEFAULT_TRAINING_CONFIG

def main():
    # Record start time for overall timing
    start_time = time.time()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create your Q-learning agent
    agent = QLearningAgent(
        epsilon=DEFAULT_TRAINING_CONFIG["init_epsilon"],
        epsilon_min=DEFAULT_TRAINING_CONFIG["min_epsilon"],
        epsilon_decay=DEFAULT_TRAINING_CONFIG["epsilon_decay"],
        alpha=DEFAULT_TRAINING_CONFIG["alpha"],
        gamma=DEFAULT_TRAINING_CONFIG["gamma"],
        piece_type=X_TYPE
    )
    
    # Use progressive training
    win_rate = progressive_training(agent)
    
    # Report final timing
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Final win rate: {win_rate:.4f}")
    
    return win_rate

if __name__ == "__main__":
    main()

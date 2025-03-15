import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot_training_performance(win_history, episode_rewards=None, epsilon_history=None, q_stats=None):
    """
    Visualize training performance metrics
    
    Parameters:
    - win_history: List of 0s and 1s indicating losses and wins
    - episode_rewards: List of rewards per episode
    - epsilon_history: List of epsilon values over time
    - q_stats: Dict with q_value statistics over time
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Win rate over time (moving average)
    plt.subplot(2, 2, 1)
    window_size = min(100, len(win_history))
    if window_size > 0:
        moving_avg = np.convolve(win_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(len(moving_avg)), moving_avg)
        plt.title(f'Win Rate (Moving Avg over {window_size} games)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.grid(True)
    
    # Plot 2: Win rate by training phase
    plt.subplot(2, 2, 2)
    if len(win_history) > 0:
        # Split into phases if we have enough data
        phase_size = len(win_history) // 4 if len(win_history) >= 400 else len(win_history) // 2
        if phase_size > 0:
            phases = []
            labels = []
            for i in range(0, len(win_history), phase_size):
                if i + phase_size <= len(win_history):
                    phase_win_rate = sum(win_history[i:i+phase_size]) / phase_size
                    phases.append(phase_win_rate)
                    labels.append(f"Phase {i//phase_size+1}")
            
            plt.bar(labels, phases)
            plt.title('Win Rate by Training Phase')
            plt.ylabel('Win Rate')
            plt.grid(axis='y')
    
    # Plot 3: Episode rewards if available
    plt.subplot(2, 2, 3)
    if episode_rewards and len(episode_rewards) > 0:
        window_size = min(100, len(episode_rewards))
        smoothed_rewards = np.convolve(episode_rewards, 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards)
        plt.title(f'Average Reward (Moving Avg over {window_size} episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
    
    # Plot 4: Q-value distribution if available
    plt.subplot(2, 2, 4)
    if q_stats and len(q_stats.get('max_q', [])) > 0:
        plt.plot(q_stats['max_q'], label='Max Q')
        plt.plot(q_stats['min_q'], label='Min Q')
        plt.plot(q_stats['avg_q'], label='Avg Q')
        plt.title('Q-Value Statistics Over Time')
        plt.xlabel('Checkpoint')
        plt.ylabel('Q-Value')
        plt.legend()
        plt.grid(True)
    elif epsilon_history and len(epsilon_history) > 0:
        plt.plot(epsilon_history)
        plt.title('Exploration Rate (Epsilon) Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_performance.png')
    plt.show()

def analyze_agent(agent, results_dir='results'):
    """Analyze agent's function approximator weights"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Use weights instead of q_table
    weights = agent.q_function.weights
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights)
    plt.title('Feature Weights in Function Approximator')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight Value') 
    plt.grid(True)
    plt.savefig(f'{results_dir}/feature_weights.png')
    plt.close()
    
    # Save summary statistics
    stats = {
        'min_weight': min(weights),
        'max_weight': max(weights), 
        'avg_weight': sum(weights) / len(weights),
        'feature_count': len(weights)
    }
    
    with open(f'{results_dir}/agent_stats.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    return stats
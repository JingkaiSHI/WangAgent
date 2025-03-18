from training.evaluation import assess_agent
from training.agent_io import save_agent, load_agent
from random_player import RandomPlayer
from my_trainer import trainer
from my_player3 import O_TYPE
from visualize import plot_training_performance, analyze_agent
from training.opponents import create_greedy_opponent, create_self_play_opponent, create_mixed_opponent

def progressive_training(agent):
    """Training pipeline with gradually increasing opponent difficulty"""
    # Keep track of the best performance
    best_win_rate = 0.0
    best_model_file = "best_agent.pkl"

    # Lists to store metrics across all training phases
    combined_win_history = []
    combined_reward_history = []
    combined_epsilon_history = []
    q_stats_history = {'min_q': [], 'max_q': [], 'avg_q': []}
    
    # Initial assessment
    print("Initial assessment:")
    initial_win_rate = assess_agent(agent)
    best_win_rate = initial_win_rate
    save_agent(agent, best_model_file)

    # Phase 1: LONGER random opponent training to build solid foundation
    print("Phase 1: Extended random opponent training...")
    random_opponent = RandomPlayer()
    random_opponent.player = O_TYPE
    trainer1 = trainer(agent, random_opponent, num_episodes=5000)  # Increased from 3000
    trainer1.train()

    # Add these lines after trainer1.train()
    combined_win_history.extend(trainer1.win_history)
    combined_reward_history.extend(trainer1.episode_rewards)
    combined_epsilon_history.extend(trainer1.epsilon_history)

    if hasattr(trainer1, 'q_stats_history'):
        for key in q_stats_history:
            if key in trainer1.q_stats_history:
                q_stats_history[key].extend(trainer1.q_stats_history[key])
    
    # Assessment and checkpoint
    win_rate = assess_agent(agent)
    if win_rate > best_win_rate + 0.02:
        best_win_rate = win_rate
        save_agent(agent, best_model_file)
        print(f"New best model saved with win rate: {best_win_rate:.2f}")
    
    # Phase 2: Purely GREEDY opponent that prioritizes captures
    print("Phase 2: Training against greedy tactical opponent...")
    greedy_opponent = create_greedy_opponent()  # New function - defined below
    greedy_opponent.player = O_TYPE
    trainer2 = trainer(agent, greedy_opponent, num_episodes=2500)
    trainer2.train()

    # Add these lines after trainer2.train()
    combined_win_history.extend(trainer2.win_history)
    combined_reward_history.extend(trainer2.episode_rewards)
    combined_epsilon_history.extend(trainer2.epsilon_history)

    if hasattr(trainer2, 'q_stats_history'):
        for key in q_stats_history:
            if key in trainer2.q_stats_history:
                q_stats_history[key].extend(trainer2.q_stats_history[key])
    
    # Assessment and checkpoint
    win_rate = assess_agent(agent)
    if win_rate > best_win_rate + 0.02:
        best_win_rate = win_rate
        save_agent(agent, best_model_file)
        print(f"New best model saved with win rate: {best_win_rate:.2f}")
    
    # Phase 3: Self-play with minimal randomization
    print("Phase 3: Strategic self-play training...")
    self_play_opponent = create_self_play_opponent(agent)
    self_play_opponent.epsilon = 0.1  # Lower exploration - more strategic play
    trainer3 = trainer(agent, self_play_opponent, num_episodes=1500)
    trainer3.train()

    # Add these lines after trainer3.train()
    combined_win_history.extend(trainer3.win_history)
    combined_reward_history.extend(trainer3.episode_rewards)
    combined_epsilon_history.extend(trainer3.epsilon_history)

    if hasattr(trainer3, 'q_stats_history'):
        for key in q_stats_history:
            if key in trainer3.q_stats_history:
                q_stats_history[key].extend(trainer3.q_stats_history[key])
    
    # Assessment and checkpoint
    win_rate = assess_agent(agent)
    if win_rate > best_win_rate + 0.02:
        best_win_rate = win_rate
        save_agent(agent, best_model_file)
        print(f"New best model saved with win rate: {best_win_rate:.2f}")
    
    # Phase 4: Mixed training with balanced opponent distribution
    print("Phase 4: Mixed opponent generalization training...")
    mixed_opponent = create_mixed_opponent(agent, 
                                          random_weight=0.4,   # 40% random
                                          greedy_weight=0.3,   # 30% greedy
                                          pattern_weight=0.2,  # 20% pattern
                                          self_play_weight=0.1) # 10% self-play
    trainer4 = trainer(agent, mixed_opponent, num_episodes=4000)
    trainer4.train()

    # Add these lines after trainer4.train()
    combined_win_history.extend(trainer4.win_history)
    combined_reward_history.extend(trainer4.episode_rewards)
    combined_epsilon_history.extend(trainer4.epsilon_history)

    if hasattr(trainer4, 'q_stats_history'):
        for key in q_stats_history:
            if key in trainer4.q_stats_history:
                q_stats_history[key].extend(trainer4.q_stats_history[key])
    
    # Load best model and generate visualizations
    print("Generating performance visualizations...")
    agent = load_agent(best_model_file, agent)
    # Analyze final agent
    analyze_agent(agent)
    
    # Plot overall training performance
    plot_training_performance(
        combined_win_history,
        episode_rewards=combined_reward_history,
        epsilon_history=combined_epsilon_history,
        q_stats=q_stats_history
    )
    
    # Final assessment
    final_win_rate = assess_agent(agent, num_test_games=200)
    return final_win_rate
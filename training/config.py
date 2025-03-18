"""Training configuration constants"""

DEFAULT_TRAINING_CONFIG = {
    # Phase durations
    "random_phase_episodes": 5000,
    "greedy_phase_episodes": 2500,
    "self_play_episodes": 1500,
    "mixed_opponent_episodes": 4000,
    
    # Opponent weights for mixed training
    "mixed_random_weight": 0.4,
    "mixed_greedy_weight": 0.3,
    "mixed_pattern_weight": 0.2,
    "mixed_self_play_weight": 0.1,
    
    # Assessment settings
    "assessment_games": 100,
    "final_assessment_games": 200,
    
    # Agent settings
    "init_epsilon": 0.8,
    "min_epsilon": 0.01,
    "epsilon_decay": 0.995,
    "alpha": 0.02,
    "gamma": 0.8,
    
    # File locations
    "best_model_file": "best_agent.pkl",
    "results_dir": "results"
}
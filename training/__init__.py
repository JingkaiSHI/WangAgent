"""
Go Agent Training Package - Tools for training Go reinforcement learning agents
"""

from training.training_pipeline import progressive_training
from training.evaluation import assess_agent, play_game
from training.agent_io import save_agent, load_agent
from training.config import DEFAULT_TRAINING_CONFIG

__all__ = [
    'progressive_training', 
    'assess_agent', 
    'play_game', 
    'save_agent', 
    'load_agent',
    'DEFAULT_TRAINING_CONFIG'
]
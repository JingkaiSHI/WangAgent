import pickle
import os

def save_agent(agent, filename):
    """Save agent to file with weights and training parameters"""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump({
            'weights': agent.q_function.weights,
            'epsilon': agent.epsilon,
            'alpha': agent.alpha,
        }, f)
    print(f"Agent saved to {filename}")

def load_agent(filename, base_agent):
    """Load agent from file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        base_agent.q_function.weights = data['weights']
        base_agent.epsilon = data.get('epsilon', 0.01)
        base_agent.alpha = data.get('alpha', 0.01)
    return base_agent
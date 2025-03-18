import random
import numpy as np
from go_helper import parse_input, write_move, get_all_legal_moves
from feature_extract_module import extract_action_features, extract_state_features
import pickle

X_TYPE = 1
O_TYPE = 2


class LinearQFunction:
    def __init__(self, feature_count=20):
        # Initialize weights with small random values for better convergence
        self.weights = np.random.uniform(-0.1, 0.1, feature_count)
        self.feature_count = feature_count
        
    def predict(self, state_features, action_features):
        """Calculate Q-value from combined features"""
        # Check dimensions match expectations  
        if len(state_features) + len(action_features) != self.feature_count:
            raise ValueError(f"Feature dimension mismatch: got {len(state_features)} + {len(action_features)}, " 
                           f"expected {self.feature_count}")
            
        combined_features = np.concatenate([state_features, action_features])
        return np.dot(combined_features, self.weights)
    
    def update(self, state_features, action_features, target, alpha):
        """Update weights using TD error"""
        # Make prediction
        prediction = self.predict(state_features, action_features)
        
        # Calculate error
        error = target - prediction
        
        # Clip error for stability
        error = np.clip(error, -1.0, 1.0)
        
        # Update weights (TD update)
        combined_features = np.concatenate([state_features, action_features])
        self.weights += alpha * error * combined_features
        
        # Clip weights to reasonable range
        self.weights = np.clip(self.weights, -5.0, 5.0)
        
        return error


# Maybe we need 2 variation of this function to make things faster


################################### Definition of Agent ##################################

# REPLACE the entire QLearningAgent class (starting at line ~590) with:

class QLearningAgent:
    def __init__(self, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.995, 
                 alpha=0.05, gamma=0.9, piece_type=X_TYPE):
        # Learning parameters - lower initial values for stability
        self.epsilon = 0
        self.epsilon_min = 0
        self.epsilon_decay = 0
        self.alpha = alpha
        self.gamma = gamma
        self.piece_type = piece_type
        
        # Feature count: 14 state + 6 action = 20 total features
        self.q_function = LinearQFunction(20)

        try:
            with open("best_agent.pkl", 'rb') as f:
                data = pickle.load(f)
                self.q_function.weights = data['weights']
                print("Loaded Q-function weights from best_agent.pkl")
        except FileNotFoundError:
            print("No pre-trained weights found, using default weights.")
        except Exception as e:
            print(f"Error loading pre-trained weights: {str(e)}")
        
        # Game state tracking
        self.prev_board = None
        self.curr_board = None
        self.last_action = None
        self.last_state_features = None
        self.last_action_features = None
        self.reward = 0.0  # Always initialize with a value
        self.game_end = False
        
        # Experience buffer - smaller size, simpler implementation
        self.replay_buffer = []
        self.replay_buffer_size = 1000
        self.min_replay_size = 64
        self.batch_size = 16
        
        # Track errors for debugging
        self.training_errors = []
    
    def initiate_game(self):
        """Reset agent state for new game"""
        self.prev_board = None
        self.curr_board = None
        self.last_action = None
        self.last_state_features = None
        self.last_action_features = None
        self.reward = 0.0  # Initialize reward with zero 
        self.game_end = False
        
    def load_cur_state(self, dir="input.txt"):
        """Load current state from input file"""
        piece_type, prev_board, curr_board = parse_input(dir)
        self.curr_board = curr_board
        self.prev_board = prev_board
        self.piece_type = piece_type
    
    def take_action(self):
        """Select an action using ε-greedy policy"""
        legal_moves = get_all_legal_moves(self.curr_board, self.prev_board, self.piece_type)
        
        # Extract current state features
        state_features = extract_state_features(self.curr_board, self.piece_type)
        self.last_state_features = state_features
        
        # Handle PASS-only case
        if legal_moves == ["PASS"]:
            self.last_action_features = extract_action_features(self.curr_board, "PASS", self.piece_type)
            self.last_action = "PASS"
            write_move("PASS")
            return
        
        # Exploration: select random non-PASS move
        if random.random() < self.epsilon:
            non_pass_moves = [m for m in legal_moves if m != "PASS"]
            if non_pass_moves:
                action = random.choice(non_pass_moves)
            else:
                action = "PASS"
        else:
            # Exploitation: select best move according to Q-function
            best_action = None
            best_q_value = float('-inf')
            
            action_values = []
            for move in legal_moves:
                action_features = extract_action_features(self.curr_board, move, self.piece_type)
                q_value = self.q_function.predict(state_features, action_features)
                action_values.append((move, q_value))
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = move
            
            action = best_action if best_action is not None else "PASS"
            
            # Debug output occasionally
            if random.random() < 0.001:
                print(f"\nBoard Position:")
                for row in self.curr_board:
                    print(row)
                print(f"Selected move: {action}, Q-value: {best_q_value:.2f}")
                top_actions = sorted(action_values, key=lambda x: x[1], reverse=True)[:3]
                print(f"Top 3 actions: {top_actions}")
        
        # Store features and write move
        self.last_action_features = extract_action_features(self.curr_board, action, self.piece_type)
        self.last_action = action
        write_move(action)
        
    
# doesn't need to worry about other util functions, only worry about how things are calculated
if __name__ == "__main__":
    agent = QLearningAgent()
    agent.load_cur_state()
    agent.take_action()

import math
import random
import numpy as np
from go_helper import parse_input, write_move, get_all_legal_moves
from feature_extract_module import extract_action_features, extract_state_features

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
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.piece_type = piece_type
        
        # Feature count: 14 state + 6 action = 20 total features
        self.q_function = LinearQFunction(20)
        
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
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def decay_alpha(self):
        """Decay learning rate more gently"""
        self.alpha = max(0.01, self.alpha * 0.9995)  # Slower decay
        
    def load_cur_state(self, dir="input.txt"):
        """Load current state from input file"""
        _, prev_board, curr_board = parse_input(dir)
        self.curr_board = curr_board
        self.prev_board = prev_board
    
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
        
    def observe_world(self):
        """Calculate reward from current board state"""
        _, _, self.curr_board = parse_input("input.txt")
        opponent = 3 - self.piece_type
        
        # Material advantage (normalized)
        my_stones = sum(row.count(self.piece_type) for row in self.curr_board)
        opp_stones = sum(row.count(opponent) for row in self.curr_board)
        material_advantage = (my_stones - opp_stones) / 5.0  # Scale down
        
        # Simple territory estimation
        territory = 0
        for i in range(5):
            for j in range(5):
                if self.curr_board[i][j] == 0:  # Empty space
                    my_adj = 0
                    opp_adj = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 5 and 0 <= nj < 5:
                            if self.curr_board[ni][nj] == self.piece_type:
                                my_adj += 1
                            elif self.curr_board[ni][nj] == opponent:
                                opp_adj += 1
                    
                    if my_adj > 0 and opp_adj == 0:
                        territory += 0.2  # Scale down
                    elif opp_adj > 0 and my_adj == 0:
                        territory -= 0.2  # Scale down
        
        # Small penalty for passing (discourage without overwhelming other factors)
        pass_penalty = -0.5 if self.last_action == "PASS" else 0.0
        
        # Combine components into final reward
        self.reward = material_advantage + territory + pass_penalty
        
    def update_q_value(self):
        """Update Q-function with TD learning"""
        if self.last_state_features is None or self.last_action_features is None:
            return
            
        # Safety check
        if self.reward is None:
            self.reward = 0.0
        
        # If game has ended, use terminal reward
        if self.game_end:
            player_stones = sum(row.count(self.piece_type) for row in self.curr_board)
            opponent_stones = sum(row.count(3 - self.piece_type) for row in self.curr_board)
            
            # Apply komi for white
            if 3 - self.piece_type == O_TYPE:
                opponent_stones += 2.5
            
            # Terminal reward: set to moderate value to avoid extreme weights
            if player_stones > opponent_stones:
                target = 2.0  # Win
            elif player_stones < opponent_stones:
                target = -2.0  # Loss
            else:
                target = 0.0  # Draw
        else:
            # Non-terminal: use bootstrapped TD target
            next_state_features = extract_state_features(self.curr_board, self.piece_type)
            legal_moves = get_all_legal_moves(self.curr_board, self.prev_board, self.piece_type)
            
            # Calculate max next Q-value
            if not legal_moves or legal_moves == ["PASS"]:
                max_next_q = 0.0
            else:
                next_q_values = []
                for move in legal_moves:
                    action_features = extract_action_features(self.curr_board, move, self.piece_type)
                    next_q_values.append(self.q_function.predict(next_state_features, action_features))
                max_next_q = max(next_q_values) if next_q_values else 0.0
            
            # TD target
            target = self.reward + self.gamma * max_next_q
        
        # Update Q-function weights
        error = self.q_function.update(
            self.last_state_features,
            self.last_action_features,
            target,
            self.alpha
        )
        
        self.training_errors.append(error)
        
        # Store experience for replay
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
            
        self.replay_buffer.append((
            self.last_state_features,
            self.last_action_features,
            self.reward,
            extract_state_features(self.curr_board, self.piece_type) if not self.game_end else None,
            self.game_end
        ))
        
        # Perform replay if buffer is large enough
        if len(self.replay_buffer) >= self.min_replay_size:
            self.replay_experience()
            
    def replay_experience(self):
        """Simple experience replay - revisit past experiences"""
        if len(self.replay_buffer) < self.min_replay_size:
            return
            
        # Sample experiences with bias toward recent ones
        batch_indices = []
        
        # Half recent, half random
        recent_count = min(self.batch_size // 2, len(self.replay_buffer) // 10)
        random_count = self.batch_size - recent_count
        
        # Get recent experiences
        recent_indices = list(range(max(0, len(self.replay_buffer) - recent_count), len(self.replay_buffer)))
        
        # Get random experiences from the rest of buffer
        if len(self.replay_buffer) > recent_count:
            eligible_indices = list(range(len(self.replay_buffer) - recent_count))
            random_indices = random.sample(eligible_indices, min(random_count, len(eligible_indices)))
        else:
            random_indices = []
            
        batch_indices = recent_indices + random_indices
        
        # Update on all sampled experiences
        for idx in batch_indices:
            state_features, action_features, reward, next_state_features, game_end = self.replay_buffer[idx]
            
            if game_end:
                target = reward
            else:
                # Simple approximation instead of recalculating next Q-values
                target = reward + self.gamma * 0.5  # Conservative estimate
                
            # Use lower learning rate for replay
            self.q_function.update(state_features, action_features, target, self.alpha * 0.3)

# doesn't need to worry about other util functions, only worry about how things are calculated
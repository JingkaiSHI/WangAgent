import math
import random
import numpy as np
from go_helper import parse_input, write_move, get_all_legal_moves, count_liberties
from feature_extract_module import extract_action_features, extract_state_features
from host import GO
from read import readInput

import sys
import os
from MCTS_module.mcts import get_best_move as mcts_get_best_move

X_TYPE = 1
O_TYPE = 2


class LinearQFunction:
    def __init__(self, feature_count=28):
        # Initialize weights with small random values for better convergence
        self.feature_count = feature_count
        if feature_count == 28:
            self.weights = np.array([
                # Original state features (14)
                1.0,    # Player_Stones - important but not dominant
                -1.0,   # Opponent_Stones - negative because opponent stones are bad
                0.5,    # Player_Liberties - moderately important
                -0.5,   # Opponent_Liberties - negative but less critical
                0.8,    # Player_Groups - fewer groups is generally better
                -0.8,   # Opponent_Groups
                0.4,    # Center_Control - strategic position matters
                0.3,    # Edge_Control
                0.2,    # Corner_Control
                0.6,    # Player_Territory
                -0.6,   # Opponent_Territory
                0.1,    # Ko_Potential
                -0.1,   # Last_Move_Distance
                0.7,    # Board_Fill_Ratio
                
                # NEW state features (6)
                0.6,    # My_Pre_Atari - recognize stones with 2 liberties
                -0.6,   # Opp_Pre_Atari - enemy stones with 2 liberties (targets)
                -0.5,   # My_Cutting_Points - vulnerable locations in our groups
                0.5,    # Opp_Cutting_Points - opportunities to split opponent 
                0.7,    # My_Eye_Potential - ability to form secure territory
                -0.7,   # Opp_Eye_Potential - prevent opponent forming territory
                
                # Original action features (6)
                0.8,    # Captures - tactically important
                0.5,    # Liberty - more liberties after move is good
                0.3,    # Connects - connecting stones is generally good
                0.2,    # Is_Center - slight preference for central moves
                -0.2,   # Is_Edge - slight bias against edge (except specific situations)
                -0.3,   # Distance_From_Center - closer to center generally better
                
                # NEW action features (2)
                -0.9,   # Self_Atari_Move - strongly avoid moves that put us in atari
                0.8     # Protects_Vulnerable - moves that protect stones in pre-atari
            ])
        else:
            self.weights = np.random.uniform(-0.1, 0.1, feature_count)
        
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
        error = target - prediction
        
        combined_features = self._combine_features(state_features, action_features)
        
        regularization_factor = 0.0001  # L2 regularization to prevent overfitting
        for i in range(len(self.weights)):
            gradient = error * combined_features[i]
            regularization = -regularization_factor * self.weights[i]
            self.weights[i] += alpha * (gradient + regularization)
            self.weights[i] = max(min(self.weights[i], 5.0), -5.0)  # Clamping to prevent extreme values
        
        return abs(error)
    
    def _combine_features(self, state_features, action_features):
        """Combine state and action features into a single array"""
        return np.concatenate([state_features, action_features])


# Maybe we need 2 variation of this function to make things faster


################################### Definition of Agent ##################################

# REPLACE the entire QLearningAgent class (starting at line ~590) with:

class QLearningAgent:
    def __init__(self, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.995, 
                 alpha=0.05, gamma=0.9, piece_type=X_TYPE, use_mcts=False):
        # Learning parameters - lower initial values for stability
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.piece_type = piece_type
        
        # Feature count: 14 state + 6 action = 20 total features
        # new feature: 6 state new features + 2 new action features: 28 features overall
        self.q_function = LinearQFunction(28)
        
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
        
        # MCTS parameters
        self.use_mcts = use_mcts
        self.mcts_iteration_limit = 3
        self.mcts_time_limit = 0.9
        
        self.mcts_for_early_game = True
        self.mcts_for_critical = True
        self.early_game_threshold = 6
        
        # Track errors for debugging
        self.training_errors = []


    def evaluate_move_with_lookahead(self, board, move, piece_type, depth=2):
        """Evaluate a move with N-step lookahead using host system"""
        if move == "PASS" or depth <= 0:
            return 0.0
            
        from host import GO
        # Create test board
        test_go = GO(5)
        test_go.board = [row[:] for row in board]
        test_go.previous_board = self.prev_board if self.prev_board else None
        
        i, j = move
        # Check validity
        if not test_go.valid_place_check(i, j, piece_type):
            return -1.0
            
        # Place stone and get captures
        test_go.place_chess(i, j, piece_type)
        opponent = 3 - piece_type
        died_pieces = test_go.remove_died_pieces(opponent)
        captures = len(died_pieces)
        
        # Base score from captures
        score = captures
        
        # Add territory influence
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < 5 and 0 <= nj < 5:
                if test_go.board[ni][nj] == piece_type:
                    score += 0.1  # Connection bonus
        
        # Look ahead if needed
        if depth > 1:
            best_opponent_score = -float('inf')
            for oi in range(5):
                for oj in range(5):
                    if test_go.board[oi][oj] == 0:  # Empty intersection
                        opp_test_go = GO(5)
                        opp_test_go.board = [row[:] for row in test_go.board]
                        opp_test_go.previous_board = [row[:] for row in board]
                        
                        # Check if move is valid for opponent
                        if opp_test_go.valid_place_check(oi, oj, opponent):
                            # Evaluate opponent's move
                            opponent_score = -self.evaluate_move_with_lookahead(
                                opp_test_go.board, (oi, oj), opponent, depth - 1)
                            
                            if opponent_score > best_opponent_score:
                                best_opponent_score = opponent_score
            
            # If opponent has a good response, reduce our score
            if best_opponent_score != -float('inf'):
                score -= best_opponent_score * 0.5  # Discount future rewards
        
        return score
        
    
    def initiate_game(self):
        """Reset agent state for new game"""
        self.prev_board = None
        self.curr_board = None
        self.last_action = None
        self.last_state_features = None
        self.last_action_features = None
        self.reward = 0.0  # Initialize reward with zero 
        self.game_end = False
        

    def get_opening_move(self):
        """Get strong opening moves for early game"""
        # Count stones to check if we're in opening phase
        stone_count = sum(row.count(1) + row.count(2) for row in self.curr_board)
        
        # Only use prepared openings in the very early game
        if stone_count > 4:
            return None
            
        if self.piece_type == 1:  # Black's openings
            # Check if board is empty (first move)
            if stone_count == 0:
                # Play in or near center - strongest first move
                return (2, 2)  # Center
                
            # If white played first move away from center
            white_played = None
            for i in range(5):
                for j in range(5):
                    if self.curr_board[i][j] == 2:  # White stone
                        white_played = (i, j)
                        break
                if white_played:
                    break
                    
            if white_played:
                # If white played far from center, take center
                dist_from_center = abs(white_played[0] - 2) + abs(white_played[1] - 2)
                if dist_from_center >= 2:
                    if self.curr_board[2][2] == 0:
                        return (2, 2)
                        
        elif self.piece_type == 2:  # White's openings
            # As white, often good to approach black's stones
            for i in range(5):
                for j in range(5):
                    if self.curr_board[i][j] == 1:  # Black stone
                        # Play adjacent if legal
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 5 and 0 <= nj < 5 and self.curr_board[ni][nj] == 0:
                                # Prioritize moves toward center
                                if abs(ni - 2) + abs(nj - 2) <= 1:
                                    return (ni, nj)
        
        return None

    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def decay_alpha(self):
        """Decay learning rate more gently"""
        self.alpha = max(0.01, self.alpha * 0.9995)  # Slower decay
        
    def load_cur_state(self, dir="input.txt"):
        """Load current state from input file"""
        piece_type, prev_board, curr_board = parse_input(dir)
        self.piece_type = piece_type
        self.prev_board = prev_board
        self.curr_board = curr_board
    

    def take_action(self, go):
        """Select an action using either MCTS or ε-greedy Q-learning based on game state."""
        self.curr_board = go.board
        self.prev_board = go.previous_board
        
        # get all legal moves
        legal_moves = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, self.piece_type):
                    legal_moves.append((i, j))
        legal_moves.append("PASS")
        
        # Extract current state features
        state_features = extract_state_features(self.curr_board, self.piece_type)
        self.last_state_features = state_features
        
        # Handle PASS-only case
        if legal_moves == ["PASS"]:
            self.last_action_features = extract_action_features(self.curr_board, "PASS", self.piece_type)
            self.last_action = "PASS"
            write_move("PASS")
            return
        
        # not pass only, just remove pass, we don't need it any way
        legal_moves.remove("PASS")
        # 5. Adapt evaluate_move_with_lookahead for host system
        def host_evaluate_move(move):
            if move == "PASS":
                return 0.0
                
            i, j = move
            # Create a test go instance
            test_go = copy_go_board(go)
            
            # Try the move
            if not test_go.place_chess(i, j, self.piece_type):
                return -1.0  # Invalid move
                
            # Check captures (get number of stones removed)
            captures = len(test_go.remove_died_pieces(3 - self.piece_type))
                
            # Base score from captures
            return captures * 0.2  # Scale down tactical advantage
        
        # 6. Integration with urgent threats and exploration/exploitation
        urgent_defense, urgent_attack = detect_urgent_threats(self.curr_board, self.piece_type)
        valid_urgent_defense = [move for move in urgent_defense if move in legal_moves]
        valid_urgent_attack = [move for move in urgent_attack if move in legal_moves]
        
        if valid_urgent_defense and random.random() < 0.9:
            action = random.choice(valid_urgent_defense)
        elif valid_urgent_attack and random.random() < 0.8:
            action = random.choice(valid_urgent_attack)
        elif random.random() < self.epsilon:
            explore_type = random.random()
        
            if explore_type < 0.4:  # 40% - Look for tactical moves (captures, cuts)
                capture_moves = []
                cut_moves = []
                connect_moves = []
                
                for move in legal_moves:
                    if move == "PASS":
                        continue
                        
                    i, j = move
                    if self.curr_board[i][j] != 0:
                        continue
                    
                    # Check for captures
                    temp_board = [row.copy() for row in self.curr_board]
                    temp_board[i][j] = self.piece_type
                    opponent = 3 - self.piece_type
                    
                    # Check for capture
                    is_capture = False
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 5 and 0 <= nj < 5 and self.curr_board[ni][nj] == opponent:
                            libs_before = count_liberties(self.curr_board, ni, nj, opponent)
                            libs_after = count_liberties(temp_board, ni, nj, opponent)
                            if libs_after == 0 and libs_before > 0:
                                is_capture = True
                                capture_moves.append(move)
                                break
                    
                    if is_capture:
                        continue
                    
                    # Check for cutting opponent stones
                    for d1 in [(-1,0), (0,-1)]:
                        n1i, n1j = i + d1[0], j + d1[1]
                        n2i, n2j = i - d1[0], j - d1[1]
                        
                        if (0 <= n1i < 5 and 0 <= n1j < 5 and 
                            0 <= n2i < 5 and 0 <= n2j < 5 and
                            self.curr_board[n1i][n1j] == opponent and 
                            self.curr_board[n2i][n2j] == opponent):
                            cut_moves.append(move)
                            break
                    
                    # Check for connection
                    own_neighbors = 0
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 5 and 0 <= nj < 5 and self.curr_board[ni][nj] == self.piece_type:
                            own_neighbors += 1
                    
                    if own_neighbors >= 0:
                        connect_moves.append(move)
                
                if capture_moves:
                    action = random.choice(capture_moves)
                elif cut_moves:
                    action = random.choice(cut_moves)
                elif connect_moves:
                    action = random.choice(connect_moves)
                else:
                    action = random.choice(legal_moves)
            elif explore_type < 0.7:
                
                center_moves = []
                for move in legal_moves:
                    if move == "PASS":
                        continue
                    
                    i, j = move
                    if self.curr_board[i][j] != 0:
                        continue
                
                    dist_from_center = abs(i - 2) + abs(j - 2)
                    if dist_from_center <= 2:
                        center_moves.append(move)
                if center_moves:
                    action = random.choice(center_moves)
                else:
                    action = random.choice(legal_moves)
            else:
                # Random exploration
                action = random.choice(legal_moves)
            
        else:
            # Exploitation: select best move according to Q-function
            best_action = None
            best_q_value = float('-inf')
            
            opening_move = self.get_opening_move()
            for move in legal_moves:
                action_features = extract_action_features(self.curr_board, move, self.piece_type)
                q_value = self.q_function.predict(state_features, action_features)
                
                if move != "PASS":
                    tactical_bonus = host_evaluate_move(move)
                    q_value += tactical_bonus * 0.2
                    
                    if move == opening_move:
                        stone_count = sum(row.count(1) + row.count(2) for row in self.curr_board)
                        opening_bonus = max(0.0, (4.0 - stone_count)) * 0.3
                        q_value += opening_bonus
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = move
            action = best_action if best_action else random.choice(legal_moves)
        
            # 7. Final validation using host system to prevent any invalid moves
        if action != "PASS":
            test_go = copy_go_board(go)
            if not test_go.valid_place_check(action[0], action[1], self.piece_type, test_check=True):
                # Invalid move detected - fall back to PASS
                print(f"WARNING: Invalid move {action} detected at final check!")
                action = "PASS"
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
            else:
                player_stones += 2.5
            
            # Terminal reward: set to moderate value to avoid extreme weights
            if player_stones > opponent_stones:
                target = 1.0  # Win
            elif player_stones < opponent_stones:
                target = -1.0  # Loss
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
# other helpers:

# Add these missing functions at the top of your file, after imports:

# Helper functions to replace removed go_helper imports


def detect_urgent_threats(board, piece_type):
    """Detect urgent threats requiring immediate response"""
    opponent = 3 - piece_type
    urgent_defense_moves = []
    urgent_attack_moves = []
    
    # Create GO object for liberty finding
    from host import GO
    test_go = GO(5)
    test_go.board = [row[:] for row in board]
    
    # Detect stones in atari (capture threats)
    for i in range(5):
        for j in range(5):
            if board[i][j] == piece_type:
                # Check if this group has only one liberty
                if test_go.find_liberty(i, j):  # Make sure it has at least one liberty
                    group = test_go.ally_dfs(i, j)
                    liberty_positions = set()
                    
                    # Find all liberty positions for this group
                    for x, y in group:
                        neighbors = test_go.detect_neighbor(x, y)
                        for nx, ny in neighbors:
                            if board[nx][ny] == 0:  # Empty = liberty
                                liberty_positions.add((nx, ny))
                    
                    if len(liberty_positions) == 1:  # Only one liberty = in atari
                        liberty = list(liberty_positions)[0]
                        urgent_defense_moves.append(liberty)
            
            # Look for opponent stones we can capture
            if board[i][j] == opponent:
                # Check if this group has only one liberty
                if test_go.find_liberty(i, j):  # Make sure it has at least one liberty
                    group = test_go.ally_dfs(i, j)
                    liberty_positions = set()
                    
                    # Find all liberty positions for this group
                    for x, y in group:
                        neighbors = test_go.detect_neighbor(x, y)
                        for nx, ny in neighbors:
                            if board[nx][ny] == 0:  # Empty = liberty
                                liberty_positions.add((nx, ny))
                    
                    if len(liberty_positions) == 1:  # Only one liberty = in atari
                        liberty = list(liberty_positions)[0]
                        urgent_attack_moves.append(liberty)
    
    return urgent_defense_moves, urgent_attack_moves

def copy_go_board(go):
    """Create a deep copy of a GO instance"""
    from host import GO
    new_go = GO(5)
    new_go.board = [row[:] for row in go.board]
    new_go.previous_board = [row[:] for row in go.previous_board] if go.previous_board else None
    new_go.died_pieces = go.died_pieces.copy() if hasattr(go, 'died_pieces') else []
    return new_go
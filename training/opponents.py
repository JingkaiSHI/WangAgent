import random
import types
import copy
from random_player import RandomPlayer
from my_player3 import X_TYPE, O_TYPE
from go_helper import get_group, count_liberties, get_all_legal_moves, parse_input, write_move

def create_pattern_opponent():
    """Create a smarter rule-based opponent"""
    from random_player import RandomPlayer
    
    # Create a new player based on the random player
    player = RandomPlayer()
    
    # Override the select_move method
    def smarter_select_move(self):
        """Select moves using basic Go patterns"""
        piece_type, prev_board, board = parse_input()
        self.piece_type = piece_type
        
        # Get legal moves but add our own safety check
        potential_moves = get_all_legal_moves(board, prev_board, piece_type)
        
        # Additional safety filter to remove occupied positions
        filtered_moves = []
        for move in potential_moves:
            if move == "PASS":
                filtered_moves.append(move)
                continue
                
            i, j = move
            if board[i][j] == 0:  # Only add if position is empty
                filtered_moves.append(move)
            else:
                # Skip this position - it's already occupied
                continue
        
        # Use our filtered list
        moves = filtered_moves
        
        if not moves:
            write_move("PASS")
            return
        
        # Score moves based on simple heuristics
        scored_moves = []
        for move in moves:
            if move == "PASS":
                scored_moves.append((move, -5))  # Discourage passing
                continue
                
            i, j = move
            score = 0
            
            # Prefer center
            center_dist = abs(i-2) + abs(j-2)  # Manhattan distance from center
            score += (4 - center_dist) * 0.5
            
            # Check for capture opportunities
            temp_board = [row.copy() for row in board]
            temp_board[i][j] = piece_type
            opponent = 3 - piece_type
            
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == opponent:
                    if count_liberties(temp_board, ni, nj, opponent) == 0:
                        score += 3  # Capture bonus
            
            scored_moves.append((move, score))
        
        # Select move with highest score
        if scored_moves:
            best_move = max(scored_moves, key=lambda x: x[1])[0]
            # Final safety check
            if best_move != "PASS":
                i, j = best_move
                if board[i][j] != 0:
                    # Fallback to PASS if somehow we still got an invalid move
                    print(f"Caught invalid move at ({i},{j}) - already occupied")
                    write_move("PASS")
                    return
            write_move(best_move)
        else:
            write_move("PASS")
    
    # Replace the method
    player.select_move = types.MethodType(smarter_select_move, player)
    return player

def create_self_play_opponent(agent):
    """Create a copy of the agent that can be used as an opponent with more randomization"""
    self_play_agent = copy.deepcopy(agent)
    self_play_agent.piece_type = O_TYPE
    self_play_agent.epsilon = 0.2  # Higher exploration to avoid local optima
    
    # Create adapter method that maps select_move -> take_action with strategic noise
    def select_move(self):
        """Adapter method with added randomization for diversity"""
        # Sometimes (10% of time) make a completely random legal move
        if random.random() < 0.1:  # Add true randomness to break out of self-play patterns
            piece_type, prev_board, board = parse_input()
            moves = get_all_legal_moves(board, prev_board, piece_type)
            non_pass = [m for m in moves if m != "PASS"]
            if non_pass:  # Prefer non-pass moves
                move = random.choice(non_pass)
            else:
                move = "PASS"
            write_move(move)
            return
            
        # Otherwise use agent logic
        self.take_action()
    
    # Attach the adapter method
    self_play_agent.select_move = types.MethodType(select_move, self_play_agent)
    
    return self_play_agent

def create_mixed_opponent(agent, random_weight=0.5, greedy_weight=0.15, pattern_weight=0.25, self_play_weight=0.1):
    """Creates an opponent that randomly switches between different strategies"""
    # Create the different opponent types
    random_opp = RandomPlayer()
    random_opp.player = O_TYPE
    
    pattern_opp = create_pattern_opponent()
    pattern_opp.player = O_TYPE
    
    # Create a noisy version of the agent for self-play that's much more random
    self_play_opp = create_self_play_opponent(agent)
    
    # Add a purely aggressive opponent
    aggressive_opp = create_pattern_opponent()  # Start with pattern opponent
    aggressive_opp.player = O_TYPE
    aggressive_opp.aggressiveness = 2.0  # Mark as aggressive
    
    
    def aggressive_select_move(self):
        """Modified move selection that prioritizes captures and attacks"""
        piece_type, prev_board, board = parse_input()
        self.piece_type = piece_type
        
        moves = get_all_legal_moves(board, prev_board, piece_type)
        if not moves or moves == ["PASS"]:
            write_move("PASS")
            return
            
        # Score moves with higher aggression
        scored_moves = []
        for move in moves:
            if move == "PASS":
                scored_moves.append((move, -10))  # Heavily discourage passing
                continue
                
            i, j = move
            score = 0
            temp_board = [row.copy() for row in board]
            temp_board[i][j] = piece_type
            opponent = 3 - piece_type
            
            # Heavily prioritize captures
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == opponent:
                    if count_liberties(temp_board, ni, nj, opponent) == 0:
                        score += 10  # Much bigger capture bonus
            
            # Prioritize attacking opponent stones
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == opponent:
                    if count_liberties(board, ni, nj, opponent) == 1:
                        score += 5  # Attack stones with 1 liberty
            
            scored_moves.append((move, score))
        
        # Select move with highest score or random if tied
        best_score = max(scored_moves, key=lambda x: x[1])[1]
        best_moves = [move for move, score in scored_moves if score == best_score]
        write_move(random.choice(best_moves))
    
    aggressive_opp.select_move = types.MethodType(aggressive_select_move, aggressive_opp)
    
    # Create container object with BETTER BALANCE OF OPPONENTS
    class MixedOpponent:
        def __init__(self):
            self.opponents = [
                ("random", random_opp, random_weight),     # 50% random play - crucial for generalization
                ("pattern", pattern_opp, pattern_weight),   # 25% pattern-based play
                ("aggressive", aggressive_opp, greedy_weight),  # 15% aggressive play
                ("self", self_play_opp, self_play_weight)     # Only 10% self-play
            ]
            self.current = self.opponents[0][1]
            self.piece_type = O_TYPE
            self.player = O_TYPE
        
        def select_move(self):
            # Choose an opponent type based on weights
            weights = [weight for _, _, weight in self.opponents]
            choice_idx = random.choices(range(len(self.opponents)), weights=weights)[0]
            opponent_name, opponent, _ = self.opponents[choice_idx]
            self.current = opponent
            
            # For debugging
            # print(f"Using opponent: {opponent_name}")
            
            # Use the selected opponent to make a move
            opponent.load_cur_state("input.txt")
            opponent.select_move()
        
        def load_cur_state(self, board_dir):
            self.current.load_cur_state(board_dir)
    
    return MixedOpponent()

def create_greedy_opponent():
    """Create a greedy opponent that focuses on captures and material gain"""
    player = RandomPlayer()
    
    def greedy_select_move(self):
        """Select moves that maximize immediate captures and liberty control"""
        piece_type, prev_board, board = parse_input()
        self.piece_type = piece_type
        opponent = 3 - piece_type
        
        moves = get_all_legal_moves(board, prev_board, piece_type)
        if not moves or moves == ["PASS"]:
            write_move("PASS")
            return
            
        # Score moves based on immediate tactical considerations
        scored_moves = []
        for move in moves:
            if move == "PASS":
                scored_moves.append((move, -20))  # Strongly discourage passing
                continue
                
            i, j = move
            score = 0
            temp_board = [row.copy() for row in board]
            temp_board[i][j] = piece_type
            
            # 1. HIGHEST PRIORITY: Capture opponent stones
            capture_count = 0
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == opponent:
                    if count_liberties(temp_board, ni, nj, opponent) == 0:
                        group = get_group(board, ni, nj, opponent)
                        capture_count += len(group)
            
            # Massive bonus for captures - 15 points per stone
            score += capture_count * 15
            
            # 2. SECOND PRIORITY: Defend own stones in atari (1 liberty)
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == piece_type:
                    if count_liberties(board, ni, nj, piece_type) == 1:
                        if count_liberties(temp_board, ni, nj, piece_type) > 1:
                            score += 10  # Big bonus for saving own stones
            
            # 3. THIRD PRIORITY: Put opponent stones in atari
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == opponent:
                    if count_liberties(board, ni, nj, opponent) > 1:
                        if count_liberties(temp_board, ni, nj, opponent) == 1:
                            score += 5  # Bonus for putting opponent in atari
            
            # 4. FOURTH PRIORITY: Maximize own liberties
            my_liberty_count = 0
            group = get_group(temp_board, i, j, piece_type)
            my_liberty_count = count_liberties(temp_board, i, j, piece_type)
            score += my_liberty_count * 0.5
            
            scored_moves.append((move, score))
        
        # Choose the highest-scored move
        best_score = max(scored_moves, key=lambda x: x[1])[1]
        best_moves = [move for move, score in scored_moves if score == best_score]
        
        # If there are multiple equally good moves, choose randomly
        chosen_move = random.choice(best_moves)
        write_move(chosen_move)
    
    player.select_move = types.MethodType(greedy_select_move, player)
    return player
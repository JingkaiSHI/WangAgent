import os
from my_player3 import QLearningAgent, X_TYPE, O_TYPE, parse_input, write_move, count_liberties, get_all_legal_moves, get_group 
from random_player import RandomPlayer
from my_trainer import trainer
import types
import copy
import random
import pickle
from visualize import plot_training_performance, analyze_agent


# Add this function
def save_agent(agent, filename):
    """Save agent to file"""
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
        base_agent.epsilon = data['epsilon']
        base_agent.alpha = data['alpha']
    return base_agent

# Simple territory evaluation function:
def evaluate_territory(board, player):
    return sum(row.count(player) for row in board)

def play_game(agent, opponent, board_dir="input.txt", action_dir="output.txt", step_limit=24, komi=2.5):
    """
    Simulate one full game between the agent and the opponent.
    Uses the trainer's update_input and should_end routines.
    Returns 1 if the agent wins, 0 if it loses.
    """
    # Create a temporary trainer instance for a single game.
    game_trainer = trainer(agent, opponent, board_dir, action_dir, num_episodes=1, step_limit=step_limit, komi=komi)
    # Reset the board to the initial state.
    game_trainer.reset_input()  # Assumes reset_input() is implemented as shown earlier.
    game_trainer.cur_step = step_limit
    game_trainer.game_end = False

    while not game_trainer.game_end:
        if agent.piece_type == X_TYPE:
            # Agent goes first.
            game_trainer.cur_piece = X_TYPE
            agent.load_cur_state(board_dir)
            agent.take_action()
            game_trainer.update_input()
            game_trainer.cur_step -= 1

            game_trainer.cur_piece = O_TYPE
            opponent.load_cur_state(board_dir)
            opponent.select_move()
            game_trainer.update_input()
            game_trainer.cur_step -= 1
        else:
            # Opponent goes first.
            game_trainer.cur_piece = O_TYPE
            opponent.load_cur_state(board_dir)
            opponent.select_move()
            game_trainer.update_input()
            game_trainer.cur_step -= 1

            game_trainer.cur_piece = X_TYPE
            agent.load_cur_state(board_dir)
            agent.take_action()
            game_trainer.update_input()
            game_trainer.cur_step -= 1
        
        # Check for end-of-game conditions.
        if not game_trainer.game_end:
            game_trainer.game_end = game_trainer.should_end()

    # Once game ends, evaluate final board state.
    piece_type, prev_board, curr_board = parse_input(board_dir)
    score_X = evaluate_territory(curr_board, X_TYPE)
    score_O = evaluate_territory(curr_board, O_TYPE)
    # Add komi bonus to White.
    score_O += (komi if O_TYPE == O_TYPE else 0)

    # Determine winner from the agent's perspective.
    if agent.piece_type == X_TYPE:
        return 1 if score_X > score_O else 0
    else:
        return 1 if score_O > score_X else 0

def assess_agent(agent, num_test_games=100):
    """Assess agent performance against a random opponent"""
    opponent = RandomPlayer()
    opponent.player = O_TYPE
    wins = 0
    
    for _ in range(num_test_games):
        outcome = play_game(agent, opponent)
        wins += outcome
    
    win_rate = wins / num_test_games
    print(f"Assessment: Agent win rate over {num_test_games} test games: {win_rate:.2f}")
    return win_rate

def train_and_assess():
    # Create your Q-learning agent.
    agent = QLearningAgent(
        epsilon=0.8, 
        epsilon_min=0.01,
        epsilon_decay=0.995,
        alpha=0.5, 
        gamma=0.8, 
        piece_type=X_TYPE
    )
    agent.agent_piece = X_TYPE  # Use this field for reward evaluation.
    
    # Create a RandomPlayer as the opponent.
    opponent = RandomPlayer()
    opponent.player = O_TYPE  # Set opponent's piece type.
    
    # Create a trainer instance for training.
    train_trainer = trainer(agent, opponent, board_dir="input.txt", action_dir="output.txt",
                            num_episodes=10000, step_limit=24, komi=2.5)
    # Train the agent.
    print("Starting training...")
    train_trainer.train()
    print("Training completed.")

    # Now, assess performance over a number of test games.
    test_games = 100
    wins = 0
    for _ in range(test_games):
        outcome = play_game(agent, opponent)
        wins += outcome
    win_rate = wins / test_games
    print("Assessment: Agent win rate over {} test games: {:.2f}".format(test_games, win_rate))


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
    
    # Add phase markers to visualize transitions between training phases
    phase_lengths = [5000, 2500, 1500, 4000]
    phase_names = ["Random", "Greedy", "Self-play", "Mixed"]
    plot_training_phases(phase_lengths, phase_names)
    
    # Final assessment
    final_win_rate = assess_agent(agent, num_test_games=200)
    return final_win_rate

def plot_training_phases(phase_lengths, phase_names):
    """Plot vertical lines marking different training phases"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculate cumulative episodes for phase transitions
    phase_transitions = np.cumsum(phase_lengths)
    
    plt.figure(figsize=(12, 6))
    
    # Plot vertical lines at phase transitions
    for i, pos in enumerate(phase_transitions):
        if i < len(phase_names) - 1:  # Don't plot after the last phase
            plt.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
            plt.text(pos + 100, 0.9, f"Start {phase_names[i+1]}", 
                    rotation=90, verticalalignment='top')
    
    # Mark phase regions
    prev = 0
    for i, pos in enumerate(phase_transitions):
        plt.axvspan(prev, pos, alpha=0.1, color=f'C{i}')
        plt.text((prev + pos) / 2, 0.95, phase_names[i], 
                horizontalalignment='center')
        prev = pos
    
    plt.ylim(0, 1)
    plt.xlim(0, phase_transitions[-1] + 500)
    plt.title('Training Phases')
    plt.xlabel('Episodes')
    plt.ylabel('Phase Regions')
    plt.savefig('results/training_phases.png')
    plt.close()


if __name__ == "__main__":
    # Create your Q-learning agent
    agent = QLearningAgent(
        epsilon=0.8, 
        epsilon_min=0.01,
        epsilon_decay=0.995,
        alpha=0.02, 
        gamma=0.8, 
        piece_type=X_TYPE
    )
    agent.agent_piece = X_TYPE  # Use this field for reward evaluation
    
    # Use progressive training instead of basic training
    progressive_training(agent)

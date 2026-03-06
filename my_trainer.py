from my_player3 import X_TYPE, O_TYPE
from go_helper import parse_input, count_liberties, get_group, read_output
import time
import pickle
import numpy as np
import os
from log_module import TrainingLogger
from read import readInput
from host import GO

class trainer:
    def __init__(self, player, opponent, board_dir="input.txt", action_dir="output.txt", num_episodes=1000, step_limit=24, komi=2.5, current_phase=1):
        self.num_episodes = num_episodes
        self.agent = player
        self.opponent = opponent
        self.board_dir = board_dir
        self.action_dir = action_dir
        self.step_limit = step_limit
        self.komi = komi
        # these 2 are used for determining if the game should end
        self.agent_pass = False
        self.opponent_pass = False
        self.game_end = False
        self.cur_step = self.step_limit
        # track which piece to place onto the board
        self.cur_piece = None
        self.invalid_move = False
        # additional informational parameters
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.win_history = []
        self.recent_outcomes = []
        self.num_episodes_completed = 0

        self.episode_rewards = []
        self.epsilon_history = []
        self.q_stats_history = {'max_q': [], 'min_q': [], 'avg_q': []}
        
        self.logger = TrainingLogger()
        self.log_frequency = 100
        self.current_phase = current_phase

    def save_training_data(self):
        training_data = {
           'win_history': self.win_history,
           'episode_rewards': self.episode_rewards,
           'epsilon_history': self.epsilon_history,
           'q_stats_history': self.q_stats_history
        }
        with open('training_data.pkl', 'wb') as f:
            pickle.dump(training_data, f)
        print("training data saved to training_data.pkl")

    def reset_input(self):
        """
        Resets the board file to the initial state.
        The format is:
          Line 1: piece type (1 for X, since Black always goes first)
          Lines 2-6: previous board (5 lines, each '00000')
          Lines 7-11: current board (5 lines, each '00000')
        """
        initial_piece = X_TYPE
        initial_board = ["00000"] * 5
        with open(self.board_dir, "w") as f:
            f.write(f"{initial_piece}\n")
            for row in initial_board:
                f.write(row + "\n")
            for row in initial_board:
                f.write(row + "\n")

    def train(self):
        """Train the agent for the specified number of episodes."""
        # Setup training
        start_time = time.time()
        total_episodes = self.num_episodes
        print(f"Training started for {total_episodes} episodes.")
        
        # Disable MCTS for training speed
        if hasattr(self.agent, 'use_mcts'):
            self.agent.use_mcts = False
        
        # Main training loop
        while self.num_episodes > 0:
            # Display progress bar
            self._display_progress(start_time, total_episodes)
            
            # Initialize episode
            episode_data = self._setup_episode()
            
            # Game loop
            while not self.game_end:
                self._play_turn(episode_data)
                
                # Check for game end
                if not self.game_end:
                    self.game_end = self.should_end()
                    if self.game_end:
                        self._process_game_end(episode_data)
            
            # End of episode processing
            self._log_episode_results(episode_data)
            self.num_episodes -= 1
        
        print("\nTraining complete!")

    def _display_progress(self, start_time, total_episodes):
        """Display progress bar for training."""
        if self.num_episodes % 10 == 0:
            elapsed_time = time.time() - start_time
            progress = (total_episodes - self.num_episodes) / total_episodes
            ets = elapsed_time / progress - elapsed_time if progress > 0 else 0
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\rTraining Progress: |{bar}| {progress:.2%} complete. Elapsed: {elapsed_time:.2f}s. ETA: {ets:.2f}s.", end='')

    def _setup_episode(self):
        """Setup a new episode and return tracking data."""
        # Initialize game state
        self.cur_step = self.step_limit
        self.game_end = False
        self.agent_pass = False
        self.opponent_pass = False
        self.invalid_move = False
        
        # Initialize agent
        self.agent.initiate_game()
        self.reset_input()
        
        # Log initial board state
        initial_board = [[0 for _ in range(5)] for _ in range(5)]
        self.logger.log_board(
            self.current_phase, 
            self.num_episodes_completed, 
            0, 
            self.agent.piece_type, 
            initial_board, 
            None, 
            0.0
        )
        
        # Return episode tracking data
        return {
            'q_values': [],
            'final_reward': 0.0
        }

    def _play_turn(self, episode_data):
        """Play a single turn (agent and opponent)."""
        if self.agent.piece_type == X_TYPE:
            # Agent (Black) plays first
            self._agent_move(episode_data)
            
            # If game not ended by agent's move, opponent plays
            if not self.game_end:
                self._opponent_move()
                
                # After opponent's move, agent observes and updates Q-values
                self.agent.observe_world()
                self.agent.update_q_value()
        else:
            # Opponent (Black) plays first
            self._opponent_move()
            
            # If game not ended by opponent's move, agent plays
            if not self.game_end:
                self._agent_move(episode_data)
                
                # Update Q-value after agent's move if needed
                if self.agent.last_action is not None:
                    self.agent.observe_world()
                    self.agent.update_q_value()

    def _agent_move(self, episode_data):
        """Handle agent's move and logging."""
        self.cur_piece = self.agent.piece_type
        
        # Load current state and make move
        piece_type, previous_board, board = readInput(5)
        go = GO(5)
        go.set_board(piece_type, previous_board, board)
        self.agent.load_cur_state(self.board_dir)
        self.agent.take_action(go)
        
        # Update board state
        self.update_input()
        self.cur_step -= 1
        
        # Log move and Q-value
        q_value = self.agent.q_function.predict(
            self.agent.last_state_features,
            self.agent.last_action_features
        )
        episode_data['q_values'].append(q_value)
        
        # Log board state
        self.logger.log_board(
            self.current_phase, 
            self.num_episodes_completed, 
            self.step_limit - self.cur_step, 
            self.agent.piece_type, 
            self.agent.curr_board, 
            self.agent.last_action, 
            q_value
        )
        
        # Periodically log weights
        if self.num_episodes_completed % self.log_frequency == 0:
            self.logger.log_weights(
                self.current_phase, 
                self.num_episodes_completed, 
                self.step_limit - self.cur_step, 
                self.agent.q_function.weights
            )

    def _opponent_move(self):
        """Handle opponent's move."""
        self.cur_piece = O_TYPE if self.agent.piece_type == X_TYPE else X_TYPE
        
        # Load current state and make move
        self.opponent.load_cur_state(self.board_dir)
        self.opponent.select_move()
        
        # Update board state
        self.update_input()
        self.cur_step -= 1

    def _process_game_end(self, episode_data):
        """Process end of game, calculate scores and rewards."""
        # Parse final board state
        _, _, curr_board = parse_input(self.board_dir)
        
        # Calculate scores
        score_X = sum(row.count(X_TYPE) for row in curr_board)
        score_O = sum(row.count(O_TYPE) for row in curr_board) + self.komi
        
        # Determine outcome and reward
        final_reward = 0.0
        outcome = ""
        
        if (self.agent.piece_type == X_TYPE and score_X > score_O) or \
        (self.agent.piece_type == O_TYPE and score_O > score_X):
            center_bonus = 0.0
            center_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
            for i, j in center_positions:
                if curr_board[i][j] == self.agent.piece_type:
                    center_bonus += 0.1
            final_reward = 2.0 + center_bonus
            self.wins += 1
            outcome = "WIN"
        elif score_X == score_O:
            final_reward = 0.0
            self.draws += 1
            outcome = "DRAW"
        else:
            final_reward = -2.0
            self.losses += 1
            outcome = "LOSS"
        
        # Store results
        self.recent_outcomes.append(outcome)
        self.win_history.append(1 if outcome == "WIN" else 0)
        episode_data['final_reward'] = final_reward
        
        # Update agent with final reward
        self.agent.game_end = True
        self.agent.reward = final_reward
        self.agent.update_q_value()
        self.agent.decay_epsilon()
        self.agent.decay_alpha()

    def _log_episode_results(self, episode_data):
        """Log results at the end of an episode."""
        # Update episode counter
        self.num_episodes_completed += 1
        
        # Store episode data
        self.episode_rewards.append(episode_data['final_reward'])
        self.epsilon_history.append(self.agent.epsilon)
        
        # Log episode summary
        avg_q_value = sum(episode_data['q_values']) / len(episode_data['q_values']) if episode_data['q_values'] else 0
        
        self.logger.log_episode(
            self.current_phase,
            self.num_episodes_completed,
            self.step_limit - self.cur_step,  # Steps taken
            "WIN" if episode_data['final_reward'] > 0 else ("DRAW" if episode_data['final_reward'] == 0 else "LOSS"),
            episode_data['final_reward'],
            avg_q_value
        )
        
        # Analyze results periodically
        self.analyze_q_table(self.num_episodes)
        
        # Track weight statistics periodically
        if self.num_episodes_completed % 500 == 0:
            weights = self.agent.q_function.weights
            if len(weights) > 0:
                self.q_stats_history['max_q'].append(float(max(weights)))
                self.q_stats_history['min_q'].append(float(min(weights)))
                self.q_stats_history['avg_q'].append(float(np.mean(weights)))

    def update_input(self):
        """
        1. read action from output
        2. if it is "PASS", do nothing
        3. otherwise, read current board from input.txt
        4. place the move onto board, edit game_end if placing a piece on somewhere occupied or invalid
        5. delete the dead pieces
        6. update this new board as the current board, previous "current" board as prev board, alter the piece type in input.txt
        All actions are done in-place
        """
        curr_action = read_output()
        if curr_action == "PASS":
            if (self.cur_piece == X_TYPE and self.agent.piece_type == X_TYPE) or (self.cur_piece == O_TYPE and self.agent.piece_type == O_TYPE):
                self.agent_pass = True
            else:
                self.opponent_pass = True
            return
        assert type(curr_action) == tuple
        i, j = curr_action
        _, _, curr_board = parse_input(self.board_dir)
        try:
            temp_board = [row.copy() for row in curr_board]
            if temp_board[i][j] != 0:
                self.invalid_move = True
                for row in temp_board:
                    print(row)
                print(f"invalid move selected: {curr_action}, game ends")
                self.game_end = True
                return
            temp_board[i][j] = self.cur_piece
            # logic of removing dead pieces after this move
            # do a 2-pass approach: first is to scan the board, give pieces their liberty count
            # second pass remove those stones that are dead
            groups_to_remove = []
            visited = set()
            board_size = len(temp_board)
            for x in range(board_size):
                for y in range(board_size):
                    if temp_board[x][y] != 0 and (x, y) not in visited:
                        player = temp_board[x][y]
                        group = get_group(temp_board, x, y, player)
                        visited.update(group)
                        if count_liberties(temp_board, x, y, player) == 0:
                            groups_to_remove.append(group)
            
            for group in groups_to_remove:
                for (x, y) in group:
                    temp_board[x][y] = 0
            # update the input where prev_board = curr_board, curr_board = temp_board, piece_type updated to other type
            new_prev_board = [row.copy() for row in curr_board]
            new_curr_board = temp_board
            new_piece_type = O_TYPE if self.cur_piece == X_TYPE else X_TYPE

            with open(self.board_dir, 'w') as f:
                f.write(f"{new_piece_type}\n")
                for row in new_prev_board:
                    f.write(''.join(str(cell) for cell in row) + "\n")
                for row in new_curr_board:
                    f.write(''.join(str(cell) for cell in row) + "\n")
        except IndexError:
            print("Invalid move selected, game ends")
            self.invalid_move = True
            self.game_end = True

    def should_end(self):
        """
        determine if the game should end:
        several criteria:
        1. self.cur_step = 0
        2. both agent_pass and opponent_pass are true
        """
        if self.cur_step == 0:
            return True
        if self.agent_pass and self.opponent_pass:
            return True
        return False
    
    def analyze_performance(self):
        window_size = 100
        if len(self.win_history) > window_size:
            self.win_history.pop(0)
        if len(self.recent_outcomes) > window_size:
            win_rate = self.recent_outcomes.count("WIN") / len(self.recent_outcomes)
            print(f"\n===== Performance (Last {len(self.recent_outcomes)} games) =====")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Wins: {self.recent_outcomes.count('WIN')}, Losses: {self.recent_outcomes.count('LOSS')}, Draws: {self.recent_outcomes.count('DRAW')}")
            print(f"Overall stats - Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}")
            print(f"Current exploration rate (epsilon): {self.agent.epsilon:.4f}")
            print("======================================\n")
    
    def analyze_q_table(self, episode_num):
        """
        Analyze the weights after each episode.
        """
        if episode_num % 100 == 0:
            # Use weights from function approximator instead of q-table
            weights = self.agent.q_function.weights
        
            if len(weights) > 0:
                print("\n===== Function Approximator Analysis =====")
                print(f"Total features: {len(weights)}")
                print(f"Max weight: {max(weights)}")
                print(f"Min weight: {min(weights)}")
                print(f"Avg weight: {sum(weights) / len(weights):.2f}")
                print(f"Non-zero weights: {sum(1 for w in weights if abs(w) > 0.01)}/{len(weights)}")
                print("=======================================\n")
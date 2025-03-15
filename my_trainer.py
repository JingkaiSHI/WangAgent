from my_player3 import X_TYPE, O_TYPE, QLearningAgent, write_move, parse_input, get_group, count_liberties
from random_player import RandomPlayer
import time
import pickle
import numpy as np

def read_output(output_dir="output.txt"):
    """
    returns the current move writen in output.txt
    can be either:
    a coordinate in tuple (row, col)
    or:
    PASS as str
    """
    try:
        with open(output_dir, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            assert len(lines) == 1
            only_line = lines[0]
            if only_line == "PASS":
                return "PASS"
            coordinates = only_line.split(',')
            assert len(coordinates) == 2
            return int(coordinates[0]), int(coordinates[1])
    except Exception as e:
        raise RuntimeError(f"Input Parsing Failed: {str(e)}")

class trainer:
    def __init__(self, player, opponent, board_dir="input.txt", action_dir="output.txt", num_episodes=1000, step_limit=24, komi=2.5):
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
        start_time = time.time()
        total_episodes = self.num_episodes
        print(f"Training started for {total_episodes} episodes.")
        while self.num_episodes > 0:
            if self.num_episodes % 10 == 0:
                elapsed_time = time.time() - start_time
                progress = (total_episodes - self.num_episodes) / total_episodes
                ets = elapsed_time / progress - elapsed_time if progress > 0 else 0
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"\rTraining Progress: |{bar}| {progress:.2%} complete. Elapsed time: {elapsed_time:.2f}s. Estimated time remaining: {ets:.2f}s.", end='')
            # setup new game
            # initialize new game
            self.cur_step = self.step_limit
            self.game_end = False
            self.agent_pass = False
            self.opponent_pass = False
            self.invalid_move = False
            self.agent.initiate_game()
            self.reset_input()
            while not self.game_end:
                if self.agent.piece_type == X_TYPE:
                    # agent goes first
                    # agent take action (change signature later!)
                    self.cur_piece = X_TYPE
                    # sets the current agent's inner state to properly align with the board
                    # Caveat: during grading/submission, make sure you call it from take_action, modify take_action to match the signature online
                    # nothing needs to be done now as it is just training locally.
                    self.agent.load_cur_state(self.board_dir)
                    # now with the properly loaded inner state information gathered, time to make a decision
                    self.agent.take_action()
                    # decision is written into output, time to update the input for the new board state (or fail to update due to invalid move selected)
                    self.update_input()
                    #print("agent has placed his input")
                    # decrement step count after updating the input
                    self.cur_step -= 1
                    # opponent take action
                    self.cur_piece = O_TYPE
                    # sets the current agent's inner state to properly align with the board
                    self.opponent.load_cur_state(self.board_dir)
                    self.opponent.select_move()
                    self.cur_step -= 1
                    self.update_input()
                    # agent observe reward, current state, update q value
                    # agent observe effect of its action and the world
                    self.agent.observe_world()
                    self.agent.update_q_value()
                else:
                    self.cur_piece = O_TYPE
                    self.opponent.load_cur_state(self.board_dir)
                    self.opponent.select_move()
                    self.cur_step -= 1
                    self.update_input()
                    if self.agent.last_action is not None:
                        self.agent.update_q_value()
                    self.cur_piece = X_TYPE
                    self.agent.load_cur_state(self.board_dir)
                    self.agent.take_action()
                    self.cur_step -= 1
                    self.update_input()
                # determine if the game should end
                if not self.game_end:
                    self.game_end = self.should_end()
                    if self.game_end:
                        _, _, curr_board = parse_input(self.board_dir)
                        score_X = sum(row.count(X_TYPE) for row in curr_board)
                        score_O = sum(row.count(O_TYPE) for row in curr_board) + self.komi
                        final_reward = 0.0
                        outcome = ""
                        if (self.agent.piece_type == X_TYPE and score_X > score_O) or \
                            (self.agent.piece_type == O_TYPE and score_O > score_X):
                            final_reward = 100.0
                            # print("The agent won!")
                            self.wins += 1
                            outcome = "WIN"
                        elif score_X == score_O:
                            final_reward = 0.0
                            self.draws += 1
                            outcome = "DRAW"
                            # print("The game is a draw!")
                        else:
                            final_reward = -100.0
                            self.losses += 1
                            outcome = "LOSS"
                            # print("The agent Lost!")
                        self.recent_outcomes.append(outcome)
                        self.win_history.append(1 if outcome == "WIN" else 0)

                        self.agent.game_end = True
                        self.agent.reward = final_reward
                        self.agent.update_q_value()
                        self.agent.decay_epsilon()
                        self.agent.decay_alpha()
                else:
                    if self.invalid_move:
                        print("warning! invalid step is taken, check step validity logic!")
            self.num_episodes_completed += 1
            # self.analyze_performance()
            self.analyze_q_table(self.num_episodes)
            self.num_episodes -= 1
            self.episode_rewards.append(final_reward)
            self.epsilon_history.append(self.agent.epsilon)

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
from my_player3 import X_TYPE, O_TYPE, QLearningAgent, write_move, parse_input, get_group, count_liberties
from random_player import RandomPlayer

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
        pass

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
        while self.num_episodes > 0:
            # setup new game
            # initialize new game
            self.cur_step = self.step_limit
            self.game_end = False
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
                else:
                    print("warning! invalid step is taken, check step validity logic!")
            self.num_episodes -= 1

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
        print("Read action:", curr_action)
        if curr_action == "PASS":
            return
        i, j = curr_action
        print("Parsed coordinates:", i, j)
        _, _, curr_board = parse_input(self.board_dir)
        try:
            temp_board = [row.copy() for row in curr_board]
            if temp_board[i][j] != 0:
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
    

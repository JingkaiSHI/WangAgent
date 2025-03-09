from my_player3 import X_TYPE, O_TYPE, QLearningAgent, write_move, parse_input
from random_player import RandomPlayer

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
        pass

    def train(self):
        while self.num_episodes > 0:
            # setup new game
            # initialize new game
            self.cur_step = self.step_limit
            self.game_end = False
            while not self.game_end:
                if self.agent.piece_type == X_TYPE:
                    # agent goes first
                    # agent take action (change signature later!)
                    self.agent.load_cur_state(self.board_dir)
                    self.agent.take_action()
                    self.update_input()
                    self.cur_step -= 1
                    # opponent take action
                    self.opponent.load_cur_state(self.board_dir)
                    self.opponent.select_move()
                    self.cur_step -= 1
                    self.update_input()
                    # agent observe reward, current state, update q value
                    self.agent.update_q_value()
                else:
                    self.opponent.load_cur_state(self.board_dir)
                    self.opponent.select_move()
                    self.cur_step -= 1
                    self.update_input()
                    if self.agent.last_action is not None:
                        self.agent.update_q_value()

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
        """
        pass

    def should_end(self):
        """
        determine if the game should end:
        several criteria:
        1. self.cur_step = 0
        2. both agent_pass and opponent_pass are true
        3. any player exceeds time to pick an action (prevented from implementation, not considered here)
        """
        return False
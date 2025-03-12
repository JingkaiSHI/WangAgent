import math
import random
X_TYPE = 1
O_TYPE = 2

def parse_input(input_file="input.txt"):
    """
    Parse the input file for board and state for current piece type and board state
    :param input_file: a file directory
    :return: (piece_type, prev_board_after_player_action, board_after_opponent_action)
    """
    try:
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            assert len(lines) == 11
            piece_type = int(lines[0])
            assert piece_type == X_TYPE or piece_type == O_TYPE
            prev_board = []
            for line in lines[1:6]:
                assert len(line) == 5
                prev_board.append([int(c) for c in line])
            cur_board = []
            for line in lines[6:11]:
                assert len(line) == 5
                cur_board.append([int(c) for c in line])
            return piece_type, prev_board, cur_board
    except FileNotFoundError:
        raise FileNotFoundError("input not found")
    except Exception as e:
        raise RuntimeError(f"Input Parsing Failed: {str(e)}")

def write_move(move):
    try:
        with open("output.txt", 'w') as f:
            if move == "PASS":
                f.write("PASS")
            elif isinstance(move, tuple) and len(move) == 2:
                i, j = move
                if 0 <= i < 5 and 0 <= j < 5:
                    f.write(f"{i},{j}")
                else:
                    raise ValueError("Invalid move coordinates!")
    except Exception as e:
        raise RuntimeError(f"Failed to write output: {str(e)}")
    
def board_to_key(curr_board, piece_type):
    """
    Convert board to a string key for the Q-table.
    :param piece_type: the type of piece we are having
    :param board: the board
    :return: encoded version of board, length is 26: first 25 chars for the state, last char for piece type
    """
    cur_result = ''.join(''.join(str(cell) for cell in row) for row in curr_board)
    result = cur_result + str(piece_type)
    return result


def key_to_board(state):
    """
    Convert a state key back to the previous board, current board, and piece type.
    The state key is constructed as:
      [curr_board (board_size^2 characters)] +
      [piece_type (1 character)]
    
    :param state: String encoding of the state.
    :return: Tuple (prev_board, curr_board, piece_type)
             where prev_board and curr_board are 2D lists representing the boards,
             and piece_type is an integer.
    """
    total_len = len(state)
    # Compute board_size^2: total length minus one for piece type, divided by 2.
    board_size_sq = (total_len - 1)
    board_size = int(math.sqrt(board_size_sq))
    
    # Extract the previous board part, current board part, and piece type.
    curr_board_str = state[0:board_size_sq]
    piece_type = int(state[-1])
    
    # Reconstruct the current board as a 2D list.
    curr_board = []
    for i in range(board_size):
        row = [int(curr_board_str[i * board_size + j]) for j in range(board_size)]
        curr_board.append(row)
    
    return curr_board, piece_type


# Maybe we need 2 variation of this function to make things faster
def get_group(board, i, j, player):
    """
    Returns all coordinates belonging to the connected group of stones for the given player starting at (i, j).
    """
    group = []
    stack = [(i, j)]
    visited = set()
    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        group.append((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(board) and 0 <= ny < len(board[0]):
                if board[nx][ny] == player and (nx, ny) not in visited:
                    stack.append((nx, ny))
    return group

def count_liberties(board, i, j, player):
    """
    Count unique liberties (empty adjacent cells) for the group at position (i, j) belonging to player.
    """
    if board[i][j] != player:
        return 0

    visited = set()
    liberties = set()
    stack = [(i, j)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(board) and 0 <= ny < len(board[0]):
                if board[nx][ny] == 0:
                    liberties.add((nx, ny))
                elif board[nx][ny] == player and (nx, ny) not in visited:
                    stack.append((nx, ny))
    return len(liberties)

def is_suicide(board, i, j, player):
    """
    Examine if placing a stone at (i, j) will cause suicide.
    Assumes the slot is empty.
    """
    # Create a temporary copy and place the stone.
    temp_board = [row.copy() for row in board]
    temp_board[i][j] = player

    # Simulate capture: For each adjacent opponent stone, if its group has no liberties, remove it.
    opponent = X_TYPE if player == O_TYPE else O_TYPE
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for dx, dy in directions:
        nx, ny = i + dx, j + dy
        if 0 <= nx < len(temp_board) and 0 <= ny < len(temp_board[0]):
            if temp_board[nx][ny] == opponent:
                if count_liberties(temp_board, nx, ny, opponent) == 0:
                    group = get_group(temp_board, nx, ny, opponent)
                    for (gx, gy) in group:
                        temp_board[gx][gy] = 0  # Remove captured opponent stones.

    # Check liberties for the group that now includes the new stone.
    return count_liberties(temp_board, i, j, player) == 0

def violates_ko(current_board, prev_board, move, player):
    """
    Examine if placing a stone at 'move' violates the KO rule by simulating the move.
    The simulation includes:
      - Placing the stone.
      - Removing any adjacent opponent groups with no liberties (i.e., captures).
    If the resulting board is identical to prev_board, then the move violates KO.
    
    :param current_board: 2D list representing the current board.
    :param prev_board: 2D list representing the board from one move ago.
    :param move: A tuple (i, j) representing the move, or "PASS".
    :param player: The current player's piece type.
    :return: True if the move violates KO, otherwise False.
    """
    if move == "PASS":
        return False
    
    i, j = move
    # Create a deep copy of the current board.
    new_board = [row.copy() for row in current_board]
    # Place the stone.
    new_board[i][j] = player

    # Determine the opponent's piece type.
    opponent = X_TYPE if player == O_TYPE else O_TYPE

    # For each adjacent cell, check if it belongs to an opponent group that should be captured.
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for dx, dy in directions:
        nx, ny = i + dx, j + dy
        if 0 <= nx < len(new_board) and 0 <= ny < len(new_board[0]):
            if new_board[nx][ny] == opponent:
                # If this opponent group has no liberties, capture it.
                if count_liberties(new_board, nx, ny, opponent) == 0:
                    group = get_group(new_board, nx, ny, opponent)
                    for (gx, gy) in group:
                        new_board[gx][gy] = 0

    # Compare the resulting board to the previous board.
    return new_board == prev_board


def get_all_legal_moves(curr_board, prev_board, player):
    """
    Returns all legal moves for the player given the current and previous board.
    """
    legal_moves = []
    board_size = len(curr_board)
    for i in range(board_size):
        for j in range(len(curr_board[0])):
            if curr_board[i][j] == 0:
                if (not is_suicide(curr_board, i, j, player)) and (not violates_ko(curr_board, prev_board, (i, j), player)):
                    legal_moves.append((i, j))
    legal_moves.append("PASS")
    return legal_moves

################################### Definition of Agent ##################################

class QLearningAgent():
    def __init__(self, epsilon=0.5, alpha=0.1, gamma=0.8, piece_type=X_TYPE):
        # action order for ith round:
        # before this round: agent action i - 1, opponent action i - 1
        #  agent action i, opponent action i
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.piece_type = piece_type
        self.q_table = {}
        self.prev_board = None # board directly after the agent's last action (this is what we are looking at when we are picking a move)
        self.curr_board = None # board directly after the opponent's last action (input of select_move)
        self.last_action = None # Used to determine if reward update is valid
        self.reward = None
        self.next_state = None

    def load_cur_state(self, dir="input.txt"):
        _, prev_board, curr_board = parse_input(dir)
        self.curr_board = curr_board
        self.prev_board = prev_board

    def take_action(self):
        """
        accessible information: 
        self.curr_board
        self.prev_board
        self.q_table
        self.epsilon
        we have the prev_board and curr_board updated, do the following:
        1. get all legal moves (simple function call)
        2. encode current state to index the q table
        3. update the q_table arbitrarilly if the state is not encountered at all
        4. select action based on current epsilon
            - random if explore (without "PASS" as an option)
            - best option if exlpoit
        5. write action to output.txt
        """
        action = None
        # implement action choice logic (missing)
        legal_moves = get_all_legal_moves(self.curr_board, self.prev_board, self.piece_type)
        legal_moves_no_pass = [move for move in legal_moves if move != "PASS"]
        if legal_moves == ["PASS"]:
            action = "PASS"
        else:
            state_key = board_to_key(self.curr_board, self.piece_type)
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
                for move in legal_moves:
                    self.q_table[state_key][move] = 0.0
            
            if random.random() < self.epsilon:
                action = random.choice(legal_moves_no_pass)
            else:
                q_vals = self.q_table[state_key]
                best_val = -float("inf")
                best_moves = []
                for move in legal_moves:
                    val = q_vals.get(move, 0.0)
                    if val > best_val:
                        best_val = val
                        best_moves = [move]
                    elif val == best_val:
                        best_moves.append(move)
                action = random.choice(best_moves) if best_moves else random.choice(legal_moves)
        write_move(action)
        self.last_action = action
    
    def observe_world(self):
        """
        Observe the world, update current reward and s' for the formula
        s' is the result from a with the opponent's movement
        s is the result right after we exerted a on the board
        compute reward r from s' and s with a

        returns:
        nothing, updates reward and next_state parameter in place
        condition of this function getting called:
        right after opponents makes some move, and we have made some move previously
        accessible information:
        - current state of the board: direct effect after last opponent move
        - previous state of the board: the state of the board before our last action, stored in self.prev_board
        - last action taken at self.last_action
        Heuristic to follow while calculating reward:
        - maximize liberty
        - maximize territory
        - connecting stones
        - making eyes: number of eyes * reward for forming an eye
        
        """
        #implement the world observation logic (missing)
        def evaluate_board(board, player):
            visited = set()
            territory = 0
            total_liberties = 0
            board_size = len(board)
            for i in range(board_size):
                for j in range(board_size):
                    if board[i][j] == player and (i, j) not in visited:
                        territory += 1
                        group = get_group(board, i, j, player)
                        visited.update(group)
                        liberties = count_liberties(board, i, j, player)
                        total_liberties += liberties
            return territory + 0.5 * total_liberties
        
        s_value = evaluate_board(self.prev_board, self.piece_type)
        s_prime_value = evaluate_board(self.curr_board, self.piece_type)
        r = s_prime_value - s_value
        self.reward = r
        s_prime_encoded = board_to_key(self.curr_board, self.piece_type)
        self.next_state = s_prime_encoded

    def update_q_value(self):
        """
        implement the formula
        Q(s, a) -> Q(s, a) + learning_rate * (reward + gamma * max(a'){Q(s', a')} - Q(s, a))
        in this case:
        s: prev_board in parse_input
        s':curr_board in parse_input
        r: pre-calculated value that is updated by observe_world
        max(a'){Q(s', a')}: index into all legal action for s'
        # Realization moment: Hangon... do we actually need the prev board as part of the encoding?
        """
        # implements the logic of updating Q value function (missing)
        if self.prev_board is None or self.last_action is None or self.reward is None:
            return
        s = self.prev_board
        a = self.last_action
        r = self.reward
        next_state = self.next_state
        curr_s = board_to_key(s, self.piece_type)

        if next_state not in self.q_table:
            legal_moves = get_all_legal_moves(self.curr_board, self.prev_board, self.piece_type)
            self.q_table[next_state] = {move: 0.0 for move in legal_moves}
        
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        current_q = self.q_table[curr_s].get(a, 0.0)
        new_q = current_q + self.alpha * (r + self.gamma * max_next_q - current_q)
        self.q_table[curr_s][a] = new_q

# doesn't need to worry about other util functions, only worry about how things are calculated
if __name__ == "__main__":
    print("---------- Testing get_all_legal_moves ----------")
    # Test 1: Empty board should yield 25 moves plus "PASS"
    empty_board = [[0]*5 for _ in range(5)]
    moves_empty = get_all_legal_moves(empty_board, empty_board, 1)
    print("Empty board legal moves count (expected 26):", len(moves_empty))
    
    # Test 2: Suicide situation
    suicide_board = [
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    moves_suicide = get_all_legal_moves(suicide_board, suicide_board, 1)
    print("Is (2,2) legal on suicide board? Expected True, Got:", (2,2) in moves_suicide)
    print("Is (1,1) legal on suicide board? Expected True, Got:", (1,1) in moves_suicide)
    
    # Test 3: KO rule test
    prev_ko_board = [
        [0, 2, 1, 0, 0],
        [2, 0, 2, 1, 0],
        [0, 2, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    curr_ko_board = [
        [0, 2, 1, 0, 0],
        [2, 1, 0, 1, 0],
        [0, 2, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    moves_ko = get_all_legal_moves(curr_ko_board, prev_ko_board, 2)
    print("For KO test, is (1,2) legal? Expected False, Got:", (1,2) in moves_ko)
    
    print("\n---------- Testing board_to_key and key_to_board ----------")
    # Create test boards.
    test_curr_board = [
        [0, 1, 2, 0, 0],
        [1, 2, 2, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 2, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]
    test_piece_type = 1
    key = board_to_key(test_curr_board, test_piece_type)
    print("State key:", key)
    decoded_curr, decoded_piece = key_to_board(key)
    print("Decoded current board:", decoded_curr)
    print("Decoded piece type:", decoded_piece)
    
    # Validate that the decoded boards and piece type match the originals.
    assert test_curr_board == decoded_curr, "Decoded current board does not match original."
    assert test_piece_type == decoded_piece, "Decoded piece type does not match original."
    print("Board encoding/decoding test passed.")
    
    print("\n---------- Testing get_group and count_liberties ----------")
    # Test a simple board for get_group.
    group_board = [
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    group = get_group(group_board, 0, 0, 1)
    print("Group starting at (0,0):", group)
    liberties = count_liberties(group_board, 0, 0, 1)
    print("Liberties of the group at (0,0):", liberties)
    
    print("\n---------- Testing QLearningAgent Initialization ----------")
    agent = QLearningAgent(epsilon=0.5, alpha=0.1, piece_type=X_TYPE)
    print("Agent initialized with epsilon =", agent.epsilon, "alpha =", agent.alpha, "piece_type =", agent.piece_type)
    print("Initial Q-table:", agent.q_table)
    
    print("\n---------- All tests executed successfully ----------")
import math
import random
import numpy as np
X_TYPE = 1
O_TYPE = 2

class LinearQFunction:
    def __init__(self, feature_count=12):
        self.weights = np.zeros(feature_count)

    def predict(self, state_features, action_features):
        combined_features = np.concatenate([state_features, action_features])
        if np.all(combined_features == 0):
            return 0.0
        return np.dot(combined_features, self.weights)
    
    def update(self, state_features, action_features, target, alpha):
        combined_features = np.concatenate([state_features, action_features])
        prediction = self.predict(state_features, action_features)
        error = target - prediction
        if np.isnan(error) or np.isinf(error):
            print(f"Warning: Invalid error value: {error}")
            return 0.0
        error = np.clip(error, -10.0, 10.0)  # Clipping to avoid large updates
        self.weights += alpha * error * combined_features
        self.weights = np.clip(self.weights, -100.0, 100.0)  # Clipping weights to avoid overflow
        return error
    
def extract_state_features(board, piece_type):
    """
    Extract numeric features from board state
    """
    opponent = 3 - piece_type

    # count pieces and their positions
    my_pieces = sum(row.count(piece_type) for row in board)
    opponent_pieces = sum(row.count(opponent) for row in board)

    # strategic area control
    center_mine = 0
    center_opponent = 0
    edge_mine = 0
    edge_opponent = 0

    # liberty count
    my_liberties = 0
    opp_liberties = 0

    # group count
    my_groups = 0
    visited = set()

    for i in range(5):
        for j in range(5):
            if board[i][j] == piece_type:
                if 1 <= i <= 3 and 1 <= j <= 3:
                    center_mine += 1
                else:
                    edge_mine += 1
                
                my_liberties += count_liberties(board, i, j, piece_type)

                if (i, j) not in visited:
                    group = get_group(board, i, j, piece_type)
                    visited.update(group)
                    my_groups += 1
            elif board[i][j] == opponent:
                if 1 <= i <= 3 and 1 <= j <= 3:
                    center_opponent += 1
                else:
                    edge_opponent += 1
                
                opp_liberties += count_liberties(board, i, j, opponent)

    territory = 0
    for i in range(5):
        for j in range(5):
            if board[i][j] == 0:
                my_adjacent = 0
                opp_adjacent = 0
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 5 and 0 <= nj < 5:
                        if board[ni][nj] == piece_type:
                            my_adjacent += 1
                        elif board[ni][nj] == opponent:
                            opp_adjacent += 1
                if my_adjacent > 0 and opp_adjacent == 0:
                    territory += 1
                elif opp_adjacent > 0 and my_adjacent == 0:
                    territory -= 1
    features = np.array([
        my_pieces / 25.0,
        opponent_pieces / 25.0,
        center_mine / 9.0,
        center_opponent / 9.0,
        edge_mine  / 16.0,
        edge_opponent / 16.0,
        float(board[2][2] == piece_type),  # Center control
        my_liberties / (my_pieces + 0.1),  # Average liberties per piece
        opp_liberties / (opponent_pieces + 0.1),  # Average liberties per opponent piece
        float(my_groups),
        float(territory) / 5.0
    ])

    return features

def extract_action_features(board, action, player):
    """
    extract features from an action
    """
    if action == "PASS":
        return np.zeros(5)
    
    i, j = action
    opponent = 3 - player

    temp_board = [row.copy() for row in board]
    temp_board[i][j] = player

    # feature 1: will capture
    captures = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < 5 and 0 <= ny < 5 and temp_board[nx][ny] == opponent:
            if count_liberties(temp_board, nx, ny, opponent) == 0:
                captures += 1
    
    # feature 2: center/edge position
    is_center = 1.0 if (1 <= i <= 3 and 1 <= j <= 3) else 0.0

    # feature 3: connects stones
    connects = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < 5 and 0 <= ny < 5 and temp_board[nx][ny] == player:
            connects += 1

    # feature 4: liberty after move
    liberty_after = count_liberties(temp_board, i, j, player)

    # feature 5: forms eye
    creates_eye = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < 5 and 0 <= ny < 5 and temp_board[nx][ny] == 0:
            eye_neighbors = []
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nnx, nny = nx + ddx, ny + ddy
                if 0 <= nnx < 5 and 0 <= nny < 5:
                    eye_neighbors.append(temp_board[nnx][nny])
            if all(n == player for n in eye_neighbors):
                creates_eye = 1.0
    
    action_feature = np.array([
        float(captures),
        is_center,
        min(connects, 4) / 4.0,
        min(liberty_after, 4) / 4.0,
        creates_eye
    ])
    return action_feature
    

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


def generate_symmetries(board_str):
    board = [list(board_str[i*5:i*5+5]) for i in range(5)]
    original = [row[:] for row in board]
    symmetries = []
    
    for _ in range(4):
        symmetries.append(''.join(''.join(row) for row in board))
        board = list(zip(*board[::-1]))
        
    mirrored = [row[::-1] for row in original]
    for _ in range(4):
        symmetries.append(''.join(''.join(row) for row in mirrored))
        mirrored = list(zip(*mirrored[::-1]))
    
    return symmetries

def evaluate_board(board, player):
            opponent = 3 - player  # X_TYPE if player is O_TYPE, vice versa
    
            # Count stones on board
            stones = sum(row.count(player) for row in board)
            opponent_stones = sum(row.count(opponent) for row in board)
    
            # Count liberties
            liberties = sum(count_liberties(board, i, j, player) for i in range(5) for j in range(5) if board[i][j] == player)
    
            # Count territory (empty spaces surrounded by player's stones)
            territory = 0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:  # Empty space
                        surrounded = True
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 5 and 0 <= nj < 5:
                                if board[ni][nj] == opponent:
                                    surrounded = False
                                    break
                            elif ni < 0 or ni >= 5 or nj < 0 or nj >= 5:
                                continue
                            else:
                                surrounded = False
                        if surrounded:
                            territory += 1
    
            # Better eye detection across the whole board
            eyes = 0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:  # Empty space
                        neighbors = []
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 5 and 0 <= nj < 5:
                                neighbors.append(board[ni][nj])
                
                        # Check if this forms an eye
                        if all(n == player for n in neighbors):
                            # Value eyes differently based on position
                            if (i, j) in [(1,1), (1,3), (3,1), (3,3)]:  # Center-ish positions
                                eyes += 2
                            elif 0 < i < 4 and 0 < j < 4:  # Other internal positions
                                eyes += 1.5
                            else:  # Edge positions
                                eyes += 1
                
            return stones + 2 * territory + 0.5 * liberties + 2.0 * eyes - 0.5 * opponent_stones


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

    # Determine the opponent's piece type.
    opponent = X_TYPE if player == O_TYPE else O_TYPE
    
    # First identify which opponent groups will be captured
    groups_to_capture = []
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    
    for dx, dy in directions:
        nx, ny = i + dx, j + dy
        if 0 <= nx < len(temp_board) and 0 <= ny < len(temp_board[0]):
            if temp_board[nx][ny] == opponent:
                if count_liberties(temp_board, nx, ny, opponent) == 0:
                    group = get_group(temp_board, nx, ny, opponent)
                    group_positions = set(group)
                    already_captured = False
                    for existing_group in groups_to_capture:
                        if set(existing_group) == group_positions:
                            already_captured = True
                            break
                    if not already_captured:
                        groups_to_capture.append(group)
    
    # Remove all captured opponent stones at once
    for group in groups_to_capture:
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
    #if random.random() < 0.1:
    #    print(f"Legal moves for player {player}: {legal_moves}") 
    return legal_moves

################################### Definition of Agent ##################################

class QLearningAgent():
    def __init__(self, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.995, alpha=0.1, gamma=0.8, piece_type=X_TYPE):
        # action order for ith round:
        # before this round: agent action i - 1, opponent action i - 1
        #  agent action i, opponent action i
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.piece_type = piece_type

        self.q_function = LinearQFunction(16)   # 11 + 5 = 16

        self.prev_board = None # board directly after the agent's last action (this is what we are looking at when we are picking a move)
        self.curr_board = None # board directly after the opponent's last action (input of select_move)
        self.last_action = None # Used to determine if reward update is valid
        self.last_state_features = None
        self.last_action_features = None
        self.reward = None
        self.game_end = False

        # expereince replay
        self.replay_buffer = []
        self.replay_buffer_size = 2000
        self.min_replay_size = 100
        self.batch_size = 32

        # new parameters for prioritized experience replay
        self.training_errors = []
    
    def initiate_game(self):
        self.prev_board = None
        self.curr_board = None
        self.last_action = None
        self.reward = None
        self.last_state = None
        self.next_state = None
        self.game_end = False

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load_cur_state(self, dir="input.txt"):
        _, prev_board, curr_board = parse_input(dir)
        self.curr_board = curr_board
        self.prev_board = prev_board

    def take_action(self):
        legal_moves = get_all_legal_moves(self.curr_board, self.prev_board, self.piece_type)

        # Extract current state's feature
        state_features = extract_state_features(self.curr_board, self.piece_type)
        self.last_state_features = state_features

        if legal_moves == ["PASS"]:
            action = "PASS"
            self.last_action_features = np.zeros(5)
        else:
            if random.random() < self.epsilon:
                non_pass = [move for move in legal_moves if move != "PASS"]
                if non_pass:
                    action = random.choice(non_pass)
                else:
                    action = "PASS"
            else:
                best_val = float('-inf')
                best_action = None

                action_values = []
                for move in legal_moves:
                    action_features = extract_action_features(self.curr_board, move, self.piece_type)
                    q_value = self.q_function.predict(state_features, action_features)

                    if move != "PASS" and action_features[0] > 0:
                        q_value += 0.5

                    action_values.append((move, q_value))

                    if q_value > best_val:
                        best_val = q_value
                        best_action = move
                
                if best_action is None:
                    non_pass = [move for move in legal_moves if move != "PASS"]
                    if non_pass:
                        action = random.choice(non_pass)
                    else:
                        action = "PASS"
                else:
                    action = best_action

                if random.random() < 0.001:
                    print(f"\nBoard Position:")
                    for row in self.curr_board:
                        print(row)
                    print(f"Selected move: {action}, Q-value: {best_val:.2f}")
                    top_actions = sorted(action_values, key=lambda x: x[1], reverse=True)[:3]
                    print(f"Top 3 actions: {top_actions}")
            self.last_action_features = extract_action_features(self.curr_board, action, self.piece_type)
        write_move(action)
        self.last_action = action
    
    def observe_world(self):
        """Simplified reward calculation focused on winning"""
        _, _, self.curr_board = parse_input("input.txt")

        opponent = 3 - self.piece_type
        my_stones = sum(row.count(self.piece_type) for row in self.curr_board)
        opponent_stones = sum(row.count(opponent) for row in self.curr_board)

        territory = 0
        for i in range(5):
            for j in range(5):
                if self.curr_board[i][j] == 0:
                    my_adjacent = 0
                    opp_adjacent = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 5 and 0 <= nj < 5:
                            if self.curr_board[ni][nj] == self.piece_type:
                                my_adjacent += 1
                            elif self.curr_board[ni][nj] == opponent:
                                opp_adjacent += 1
                    if my_adjacent > 0 and opp_adjacent == 0:
                        territory += 0.5
                    elif opp_adjacent > 0 and my_adjacent == 0:
                        territory -= 0.5
        
        # calculate material and territory advantage
        advantage = my_stones - opponent_stones + territory

        if self.last_action == "PASS":
            advantage -= 1.0
        self.reward = advantage

    def decay_alpha(self):
        self.alpha = max(0.01, self.alpha * 0.999)

    def update_q_value(self):
        """
        update function approximator with reward
        """
        if self.last_state_features is None or self.last_action_features is None:
            return
        target = 0.0
        next_state_features = None
    
        if self.game_end:
            player_stones = sum(row.count(self.piece_type) for row in self.curr_board)
            opponent_stones = sum(row.count(3 - self.piece_type) for row in self.curr_board)

            if 3 - self.piece_type == O_TYPE:
                opponent_stones += 2.5  # apply komi
            
            if player_stones > opponent_stones:
                self.reward = 10.0
            elif player_stones < opponent_stones:
                self.reward = -10.0
            else:
                self.reward = 0.0
        else:
            next_state_features = extract_state_features(self.curr_board, self.piece_type)
            legal_moves = get_all_legal_moves(self.curr_board, self.prev_board, self.piece_type)

            if not legal_moves or legal_moves == ["PASS"]:
                max_next_q = 0.0
            else:
                next_q_values = []
                for move in legal_moves:
                    action_features = extract_action_features(self.curr_board, move, self.piece_type)
                    next_q_values.append(self.q_function.predict(next_state_features, action_features))
                max_next_q = max(next_q_values) if next_q_values else 0.0
            target = self.reward + self.gamma * max_next_q
        
        error = self.q_function.update(
            self.last_state_features, 
            self.last_action_features, 
            target, 
            self.alpha
        )

        self.training_errors.append(error)

        experience = (
            self.last_state_features, 
            self.last_action_features, 
            self.reward, 
            next_state_features, 
            self.game_end
        )

        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

        if len(self.replay_buffer) >= self.min_replay_size:
            self.replay_experience()


    def replay_experience(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        recent_count = self.batch_size // 2
        random_count = self.batch_size - recent_count

        recent_indices = list(range(len(self.replay_buffer) - recent_count, len(self.replay_buffer)))

        if len(self.replay_buffer) > recent_count:
            all_indices = list(range(len(self.replay_buffer) - recent_count))
            random_indices = random.sample(all_indices, min(random_count, len(all_indices)))
        else:
            random_indices = []
        
        batch_indices = recent_indices + random_indices

        for idx in batch_indices:
            state_features, action_features, reward, _, game_end = self.replay_buffer[idx]
            if game_end:
                target = reward
            else:
                max_next_q = 0.0
                target = reward + self.gamma * max_next_q
            self.q_function.update(state_features, action_features, target, self.alpha * 0.5)

# doesn't need to worry about other util functions, only worry about how things are calculated


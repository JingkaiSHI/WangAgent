import numpy as np
####################################################
# Game board logic helper function
# includes:
# get_group
# count_liberty
# is_suicide
# violates_ko
# get_all_legal_moves
####################################################

X_TYPE, O_TYPE = 1, 2

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

####################################################
# IO util function to read and write from input.txt
####################################################

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
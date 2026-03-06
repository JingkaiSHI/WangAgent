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
    """Get connected group using host's logic"""
    from host import GO
    test_go = GO(5)
    test_go.board = [row[:] for row in board]
    return test_go.ally_dfs(i, j)

def count_liberties(board, i, j, player):
    """Count liberties using host's logic"""
    from host import GO
    test_go = GO(5)
    test_go.board = [row[:] for row in board]
    return sum(1 for _ in test_go.detect_neighbor_ally(i, j))

def is_suicide(board, i, j, player):
    """Check for suicide move using host's logic"""
    from host import GO
    test_go = GO(5)
    test_go.board = [row[:] for row in board]
    test_go.previous_board = None
    # First check if stone has liberty after placement
    temp_board = [row[:] for row in board]
    temp_board[i][j] = player
    temp_go = GO(5)
    temp_go.board = temp_board
    # If move would have liberties, it's not suicide
    if temp_go.find_liberty(i, j):
        return False
    # Check if it would capture opponent stones (not suicide)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == 3-player:
            # Check if opponent group would have no liberties
            if not test_go.find_liberty(ni, nj):
                return False
    return True

def violates_ko(current_board, prev_board, move, player):
    """Check for ko violation using host's logic"""
    from host import GO
    test_go = GO(5)
    test_go.board = [row[:] for row in current_board]
    test_go.previous_board = prev_board
    # Calculate died_pieces for proper Ko detection
    opponent = 3 - player
    test_go.died_pieces = []
    for x in range(5):
        for y in range(5):
            if current_board[x][y] == opponent and not test_go.find_liberty(x, y):
                group = test_go.ally_dfs(x, y)
                for piece in group:
                    if piece not in test_go.died_pieces:
                        test_go.died_pieces.append(piece)
    # Use host's ko check
    i, j = move
    return not test_go.valid_place_check(i, j, player, test_check=True)

def get_all_legal_moves(curr_board, prev_board, player):
    """Get all legal moves using host's logic"""
    from host import GO
    test_go = GO(5)
    test_go.board = [row[:] for row in curr_board]
    test_go.previous_board = prev_board if prev_board else None
    
    legal_moves = []
    for i in range(5):
        for j in range(5):
            if curr_board[i][j] == 0 and test_go.valid_place_check(i, j, player):
                legal_moves.append((i, j))
    legal_moves.append("PASS")
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
    

def detect_urgent_threats(board, piece_type):
    """Detect urgent threats requiring immediate response"""
    opponent = 3 - piece_type
    urgent_defense_moves = []
    urgent_attack_moves = []
    
    # Detect stones in atari (capture threats)
    for i in range(5):
        for j in range(5):
            # Look for our stones in atari
            if board[i][j] == piece_type and count_liberties(board, i, j, piece_type) == 1:
                # Find the liberty position
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == 0:
                        urgent_defense_moves.append((ni, nj))
                        break
            
            # Look for opponent stones we can capture
            if board[i][j] == opponent and count_liberties(board, i, j, opponent) == 1:
                # Find the liberty position
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == 0:
                        urgent_attack_moves.append((ni, nj))
                        break
    
    return urgent_defense_moves, urgent_attack_moves
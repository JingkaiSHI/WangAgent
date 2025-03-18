import numpy as np
from go_helper import get_group, count_liberties

####################################################
# strategic move for learning helper function
# functions includes:
# extract_state_features
# extract_action_features
####################################################
def extract_state_features(board, piece_type):
    """Extract all relevant state features from board position"""
    opponent = 3 - piece_type
    
    # Material count
    my_pieces = sum(row.count(piece_type) for row in board) / 25.0
    opponent_pieces = sum(row.count(opponent) for row in board) / 25.0
    
    # Strategic position
    center_mine = sum(1 for i in range(1, 4) for j in range(1, 4) if board[i][j] == piece_type) / 9.0
    center_opponent = sum(1 for i in range(1, 4) for j in range(1, 4) if board[i][j] == opponent) / 9.0
    edge_mine = sum(1 for i in range(5) for j in range(5) 
                   if board[i][j] == piece_type and (i == 0 or i == 4 or j == 0 or j == 4)) / 16.0
    edge_opponent = sum(1 for i in range(5) for j in range(5)
                       if board[i][j] == opponent and (i == 0 or i == 4 or j == 0 or j == 4)) / 16.0
    
    # Tactical features
    my_liberties = 0
    opp_liberties = 0
    my_groups = 0
    visited = set()
    
    for i in range(5):
        for j in range(5):
            if board[i][j] == piece_type and (i, j) not in visited:
                group = get_group(board, i, j, piece_type)
                visited.update(group)
                my_groups += 1
                my_liberties += count_liberties(board, i, j, piece_type)
            elif board[i][j] == opponent and (i, j) not in visited:
                visited.add((i, j))
                opp_liberties += count_liberties(board, i, j, opponent)
    
    # Normalize liberties
    my_lib_ratio = my_liberties / (my_pieces * 25.0 + 0.1)
    opp_lib_ratio = opp_liberties / (opponent_pieces * 25.0 + 0.1)
    
    # Atari detection (stones with 1 liberty)
    my_atari = sum(1 for i in range(5) for j in range(5) 
                  if board[i][j] == piece_type and count_liberties(board, i, j, piece_type) == 1) / 5.0
    opp_atari = sum(1 for i in range(5) for j in range(5)
                   if board[i][j] == opponent and count_liberties(board, i, j, opponent) == 1) / 5.0
    
    # Control of key points
    center_control = float(board[2][2] == piece_type) - float(board[2][2] == opponent)
    corners_control = sum(1 for i, j in [(0,0), (0,4), (4,0), (4,4)] 
                         if board[i][j] == piece_type) / 4.0 - \
                      sum(1 for i, j in [(0,0), (0,4), (4,0), (4,4)]
                         if board[i][j] == opponent) / 4.0
    
    # Territory control
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
                    territory += 0.2  # Scale down from the original
                elif opp_adjacent > 0 and my_adjacent == 0:
                    territory -= 0.2  # Scale down from the original
    
    # Color indicator - removed intentionally
    
    return np.array([
        my_pieces, 
        opponent_pieces,
        center_mine, 
        center_opponent,
        edge_mine, 
        edge_opponent,
        my_lib_ratio, 
        opp_lib_ratio,
        my_atari, 
        opp_atari,
        float(my_groups) / 5.0,
        center_control,
        corners_control,
        territory / 5.0  # Normalized territory
    ])


def extract_action_features(board, action, player):
    """Extract features for a specific action"""
    if action == "PASS":
        # For PASS moves, return special features
        stones_diff = sum(row.count(player) for row in board) - sum(row.count(3-player) for row in board)
        # Pass feature: only encourage passing when ahead or in very late game
        return np.array([0.0, 0.0, 0.0, 0.0, float(stones_diff > 2), 0.0])
    
    i, j = action
    opponent = 3 - player
    temp_board = [row.copy() for row in board]
    temp_board[i][j] = player
    
    # Capture feature
    captures = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < 5 and 0 <= nj < 5 and board[ni][nj] == opponent:
            if count_liberties(temp_board, ni, nj, opponent) == 0:
                group = get_group(board, ni, nj, opponent)
                captures += len(group)
    
    # Liberty feature
    liberty = count_liberties(temp_board, i, j, player) / 4.0
    
    # Connection feature
    connects = sum(1 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                  if 0 <= i+di < 5 and 0 <= j+dj < 5 and board[i+di][j+dj] == player) / 4.0
    
    # Position feature
    is_center = 1.0 if (i == 2 and j == 2) else 0.0  # True center
    is_middle = 1.0 if (0 < i < 4 and 0 < j < 4) else 0.0  # Middle area
    is_corner = 1.0 if (i, j) in [(0,0), (0,4), (4,0), (4,4)] else 0.0  # Corner
    
    return np.array([
        float(captures) / 5.0,
        liberty,
        connects,
        is_center,
        is_middle,
        is_corner
    ])
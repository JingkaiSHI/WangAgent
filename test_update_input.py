import os
from my_trainer import trainer
from my_player3 import parse_input, X_TYPE, O_TYPE

# Utility function: Write a new input.txt with the specified state.
def write_input_file(piece_type, prev_board, curr_board):
    with open("input.txt", "w") as f:
        f.write(f"{piece_type}\n")
        for row in prev_board:
            f.write(''.join(str(cell) for cell in row) + "\n")
        for row in curr_board:
            f.write(''.join(str(cell) for cell in row) + "\n")

# Test 1: Basic Move (No capture)
def test_update_input_basic():
    piece_type = X_TYPE  # agent's turn
    empty_board = [[0]*5 for _ in range(5)]
    # Write initial state: both prev_board and curr_board are empty.
    write_input_file(piece_type, empty_board, empty_board)
    
    # Write a valid move "2,2" to output.txt.
    with open("output.txt", "w") as f:
        f.write("2,2")
    
    # Create a trainer instance with fixed filenames.
    test_trainer = trainer(player=None, opponent=None,
                           board_dir="input.txt",
                           action_dir="output.txt",
                           num_episodes=1, step_limit=24, komi=2.5)
    test_trainer.cur_piece = X_TYPE
    
    # Call update_input() to simulate the move.
    test_trainer.update_input()
    
    new_piece_type, new_prev_board, new_curr_board = parse_input("input.txt")
    # Expected: new_prev_board equals the old current board (empty),
    # new_curr_board should have a stone (1) at (2,2).
    expected_curr = ['00000', '00000', '00100', '00000', '00000']
    new_curr_str = [''.join(str(cell) for cell in row) for row in new_curr_board]
    assert new_piece_type == O_TYPE, f"Expected piece type {O_TYPE}, got {new_piece_type}"
    assert new_prev_board == empty_board, "Expected previous board to be empty"
    assert new_curr_str == expected_curr, f"Expected current board {expected_curr}, got {new_curr_str}"
    print("Test 1 (basic move) passed.")

# Test 2: Capture Situation
def test_update_input_capture():
    piece_type = X_TYPE  # agent's turn
    prev_board = [[0]*5 for _ in range(5)]
    # Set up a board where an opponent stone (2) is nearly surrounded.
    # Current board:
    # Row0: 00000
    # Row1: 11100
    # Row2: 1 2 0 0 0   -> opponent stone at (2,1) has one liberty at (2,2)
    # Row3: 11100
    # Row4: 00000
    curr_board = [
        [0,0,0,0,0],
        [1,1,1,0,0],
        [1,2,0,0,0],
        [1,1,1,0,0],
        [0,0,0,0,0]
    ]
    write_input_file(piece_type, prev_board, curr_board)
    
    # Write move: agent places stone at (2,2) to capture the opponent stone.
    with open("output.txt", "w") as f:
        f.write("2,2")
    
    test_trainer = trainer(player=None, opponent=None,
                           board_dir="input.txt",
                           action_dir="output.txt",
                           num_episodes=1, step_limit=24, komi=2.5)
    test_trainer.cur_piece = X_TYPE
    
    test_trainer.update_input()
    
    new_piece_type, new_prev_board, new_curr_board = parse_input("input.txt")
    # After move, row 2 should be:
    # Originally row2: [1,2,0,0,0]
    # After placing 1 at (2,2): [1,2,1,0,0]
    # Then capture: opponent stone at (2,1) should be removed, row becomes: [1,0,1,0,0]
    expected_curr = ['00000', '11100', '10100', '11100', '00000']
    new_curr_str = [''.join(str(cell) for cell in row) for row in new_curr_board]
    assert new_piece_type == O_TYPE, f"Expected piece type {O_TYPE}, got {new_piece_type}"
    assert new_curr_str == expected_curr, f"Expected current board {expected_curr}, got {new_curr_str}"
    print("Test 2 (capture) passed.")

# Test 3: Illegal Move (Out-of-bounds)
def test_update_input_illegal():
    piece_type = X_TYPE
    empty_board = [[0]*5 for _ in range(5)]
    write_input_file(piece_type, empty_board, empty_board)
    
    # Write an illegal move "5,5" to output.txt (indexes 0-4 are valid).
    with open("output.txt", "w") as f:
        f.write("5,5")
    
    test_trainer = trainer(player=None, opponent=None,
                           board_dir="input.txt",
                           action_dir="output.txt",
                           num_episodes=1, step_limit=24, komi=2.5)
    test_trainer.cur_piece = X_TYPE
    
    test_trainer.update_input()
    
    assert test_trainer.game_end == True, "Expected game_end to be True for an illegal move."
    print("Test 3 (illegal move) passed.")

if __name__ == "__main__":
    test_update_input_basic()
    test_update_input_capture()
    test_update_input_illegal()
    
    # Optionally, clean up the files after testing.
    if os.path.exists("input.txt"):
        os.remove("input.txt")
    if os.path.exists("output.txt"):
        os.remove("output.txt")
    
    print("All update_input tests passed.")


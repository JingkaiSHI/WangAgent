import os
import random
from my_player3 import X_TYPE, O_TYPE, parse_input, write_move
from my_trainer import trainer  # Assumes QLearningAgent is imported in my_player3
# Assuming QLearningAgent is part of the module we test:
from my_player3 import QLearningAgent
from my_trainer import read_output
from my_player3 import get_all_legal_moves  # Import if needed

# Utility function: Write an input file with a given state.
def write_input_file(piece_type, board):
    # For this test, we treat both previous and current boards as the same.
    with open("input.txt", "w") as f:
        f.write(f"{piece_type}\n")
        for row in board:
            f.write(''.join(str(cell) for cell in row) + "\n")
        for row in board:
            f.write(''.join(str(cell) for cell in row) + "\n")

# Utility function: Read the output move from output.txt.
def read_output_file():
    with open("output.txt", "r") as f:
        line = f.readline().strip()
        if line == "PASS":
            return "PASS"
        parts = line.split(',')
        return int(parts[0]), int(parts[1])

# Test 1: Exploration Mode (epsilon=1)
def test_take_action_exploration():
    # Create a simple board state: empty board.
    board = [[0]*5 for _ in range(5)]
    write_input_file(X_TYPE, board)
    
    # Create an agent with epsilon = 1 (always explore).
    agent = QLearningAgent(epsilon=1.0, alpha=0.1, gamma=0.8, piece_type=X_TYPE)
    # Manually load state:
    agent.load_cur_state("input.txt")
    # Set a fixed random seed so that our random choice is reproducible.
    random.seed(42)
    
    agent.take_action()
    action = read_output_file()
    
    # The legal moves for an empty board will be all cells plus "PASS".
    # Our implementation excludes "PASS" during exploration if possible.
    legal_moves = get_all_legal_moves(agent.curr_board, agent.prev_board, agent.piece_type)
    if len(legal_moves) > 1 and "PASS" in legal_moves:
        legal_moves = [m for m in legal_moves if m != "PASS"]
    
    assert action in legal_moves, f"Action {action} not in legal moves {legal_moves}"
    print("Test take_action exploration passed. Action chosen:", action)

# Test 2: Exploitation Mode (epsilon=0)
def test_take_action_exploitation():
    # Create a simple board state: empty board.
    board = [[0]*5 for _ in range(5)]
    write_input_file(X_TYPE, board)
    
    # Create an agent with epsilon = 0 (always exploit).
    agent = QLearningAgent(epsilon=0.0, alpha=0.1, gamma=0.8, piece_type=X_TYPE)
    agent.load_cur_state("input.txt")
    
    # Manually pre-populate the Q-table for the current state to favor a specific move.
    state_key = ''.join(''.join(str(cell) for cell in row) for row in agent.curr_board) + str(agent.piece_type)
    agent.q_table[state_key] = {}
    # Let's favor move (3, 3) by setting its Q-value higher than others.
    legal_moves = get_all_legal_moves(agent.curr_board, agent.prev_board, agent.piece_type)
    for move in legal_moves:
        agent.q_table[state_key][move] = 0.0
    agent.q_table[state_key][(3, 3)] = 1.0  # Best move

    agent.take_action()
    action = read_output_file()
    assert action == (3, 3), f"Expected action (3,3), got {action}"
    print("Test take_action exploitation passed. Action chosen:", action)

if __name__ == "__main__":
    # Remove any existing files first.
    
    test_take_action_exploration()
    test_take_action_exploitation()
    
    print("All take_action tests passed.")

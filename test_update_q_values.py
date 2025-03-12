import os
from my_player3 import X_TYPE, O_TYPE, QLearningAgent, parse_input, write_move, get_all_legal_moves

# Utility function: Write an input file with the given board state.
def write_input_file(piece_type, board):
    # For this test, we set both previous and current boards to be the same.
    with open("input.txt", "w") as f:
        f.write(f"{piece_type}\n")
        for row in board:
            f.write(''.join(str(cell) for cell in row) + "\n")
        for row in board:
            f.write(''.join(str(cell) for cell in row) + "\n")

def test_update_q_value():
    # Create an agent with fixed parameters.
    agent = QLearningAgent(epsilon=0.1, alpha=0.1, gamma=0.8, piece_type=X_TYPE)
    # Create an initial board (empty board).
    board = [[0]*5 for _ in range(5)]
    write_input_file(X_TYPE, board)
    
    # Set up initial state s (the current board state and piece type).
    s = ''.join(''.join(str(cell) for cell in row) for row in board) + str(X_TYPE)
    # Set the agent's last_state to s.
    agent.last_state = s
    # Set the agent's last action to (2,2) (the move taken).
    agent.last_action = (2, 2)
    # Set an example reward.
    agent.reward = 5.0  # e.g., the agent received a reward of 5.
    
    # Pre-populate Q-table for state s.
    agent.q_table[s] = { (2,2): 0.0 }  # initial Q(s, (2,2)) is 0.
    
    # Now, simulate the new board state s' after the move:
    # Let's assume the move (2,2) was applied to the board.
    new_board = [row.copy() for row in board]
    new_board[2][2] = X_TYPE  # Agent placed a stone at (2,2).
    agent.curr_board = new_board
    # For simplicity, let the previous board remain the same.
    agent.prev_board = board
    
    # Compute next state s' encoding.
    next_state = ''.join(''.join(str(cell) for cell in row) for row in new_board) + str(X_TYPE)
    # Initialize Q-values for s' with two moves:
    agent.q_table[next_state] = { (1,1): 2.0, (2,2): 3.0 }  # so max Q(s', a') = 3.0
    
    # Expected update:
    # current_q = 0.0, reward = 5.0, gamma = 0.8, max_next_q = 3.0, alpha = 0.1.
    # new_q = 0.0 + 0.1 * (5.0 + 0.8*3.0 - 0.0) = 0.1 * (5.0 + 2.4) = 0.1 * 7.4 = 0.74
    expected_new_q = 0.74
    
    agent.update_q_value()
    updated_q = agent.q_table[s][(2,2)]
    assert abs(updated_q - expected_new_q) < 1e-6, f"Expected Q-value {expected_new_q}, got {updated_q}"
    print("Test update_q_value passed, updated Q-value:", updated_q)

if __name__ == "__main__":
    # Run the test.
    test_update_q_value()
    
    
    print("All update_q_value tests passed.")

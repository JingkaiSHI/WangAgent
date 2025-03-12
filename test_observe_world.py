import os
from my_player3 import X_TYPE, O_TYPE, QLearningAgent, get_group, count_liberties

def test_observe_world():
    # Create an agent and set its fixed piece type.
    agent = QLearningAgent(epsilon=0.1, alpha=0.1, gamma=0.8, piece_type=X_TYPE)
    agent.agent_piece = X_TYPE  # fixed perspective for evaluation

    # Define s: board after our move.
    # For example, an empty board with our stone at (2,2)
    s_board = [[0]*5 for _ in range(5)]
    s_board[2][2] = X_TYPE

    # Define s': board after opponent's move.
    # Let's simulate a scenario where the opponent places a stone at (0,0) with no effect on our group.
    s_prime_board = [row.copy() for row in s_board]
    s_prime_board[0][0] = O_TYPE

    # Set these boards in the agent.
    agent.prev_board = s_board
    agent.curr_board = s_prime_board

    # For our simple evaluation:
    # Evaluate s_board for X_TYPE:
    #   Our stone at (2,2) should have 4 liberties (neighbors: (1,2), (3,2), (2,1), (2,3)).
    #   So evaluation(s) = 1 + 0.5*4 = 3.0
    # Evaluate s_prime_board for X_TYPE:
    #   Our stone remains unchanged, so evaluation(s') should still be 3.0.
    # Hence, expected reward = 3.0 - 3.0 = 0.
    expected_reward = 0.0
    
    # Call observe_world.
    agent.observe_world()
    
    # Compute expected next state encoding.
    expected_next_state = ''.join(''.join(str(cell) for cell in row) for row in s_prime_board) + str(agent.agent_piece)
    
    assert abs(agent.reward - expected_reward) < 1e-6, f"Expected reward {expected_reward}, got {agent.reward}"
    assert agent.next_state == expected_next_state, f"Expected next state {expected_next_state}, got {agent.next_state}"
    print("Test observe_world passed. Reward:", agent.reward, "Next state:", agent.next_state)

if __name__ == "__main__":
    test_observe_world()

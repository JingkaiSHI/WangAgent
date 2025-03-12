import os
from my_player3 import QLearningAgent, X_TYPE, O_TYPE, parse_input
from random_player import RandomPlayer
from my_trainer import trainer

# Simple territory evaluation function:
def evaluate_territory(board, player):
    return sum(row.count(player) for row in board)

def play_game(agent, opponent, board_dir="input.txt", action_dir="output.txt", step_limit=24, komi=2.5):
    """
    Simulate one full game between the agent and the opponent.
    Uses the trainer’s update_input and should_end routines.
    Returns 1 if the agent wins, 0 if it loses.
    """
    # Create a temporary trainer instance for a single game.
    game_trainer = trainer(agent, opponent, board_dir, action_dir, num_episodes=1, step_limit=step_limit, komi=komi)
    # Reset the board to the initial state.
    game_trainer.reset_input()  # Assumes reset_input() is implemented as shown earlier.
    game_trainer.cur_step = step_limit
    game_trainer.game_end = False

    while not game_trainer.game_end:
        if agent.piece_type == X_TYPE:
            # Agent goes first.
            game_trainer.cur_piece = X_TYPE
            agent.load_cur_state(board_dir)
            agent.take_action()
            game_trainer.update_input()
            game_trainer.cur_step -= 1

            game_trainer.cur_piece = O_TYPE
            opponent.load_cur_state(board_dir)
            opponent.select_move()
            game_trainer.update_input()
            game_trainer.cur_step -= 1
        else:
            # Opponent goes first.
            game_trainer.cur_piece = O_TYPE
            opponent.load_cur_state(board_dir)
            opponent.select_move()
            game_trainer.update_input()
            game_trainer.cur_step -= 1

            game_trainer.cur_piece = X_TYPE
            agent.load_cur_state(board_dir)
            agent.take_action()
            game_trainer.update_input()
            game_trainer.cur_step -= 1
        
        # Check for end-of-game conditions.
        if not game_trainer.game_end:
            game_trainer.game_end = game_trainer.should_end()

    # Once game ends, evaluate final board state.
    piece_type, prev_board, curr_board = parse_input(board_dir)
    score_X = evaluate_territory(curr_board, X_TYPE)
    score_O = evaluate_territory(curr_board, O_TYPE)
    # Add komi bonus to White.
    score_O += (komi if O_TYPE == O_TYPE else 0)

    # Determine winner from the agent's perspective.
    if agent.piece_type == X_TYPE:
        return 1 if score_X > score_O else 0
    else:
        return 1 if score_O > score_X else 0

def train_and_assess():
    # Create your Q-learning agent.
    agent = QLearningAgent(epsilon=0.1, alpha=0.1, gamma=0.8, piece_type=X_TYPE)
    agent.agent_piece = X_TYPE  # Use this field for reward evaluation.
    
    # Create a RandomPlayer as the opponent.
    opponent = RandomPlayer()
    opponent.player = O_TYPE  # Set opponent's piece type.
    
    # Create a trainer instance for training.
    train_trainer = trainer(agent, opponent, board_dir="input.txt", action_dir="output.txt",
                            num_episodes=1000, step_limit=24, komi=2.5)
    # Train the agent.
    print("Starting training...")
    train_trainer.train()
    print("Training completed.")

    # Now, assess performance over a number of test games.
    test_games = 20
    wins = 0
    for _ in range(test_games):
        outcome = play_game(agent, opponent)
        wins += outcome
    win_rate = wins / test_games
    print("Assessment: Agent win rate over {} test games: {:.2f}".format(test_games, win_rate))

if __name__ == "__main__":
    train_and_assess()

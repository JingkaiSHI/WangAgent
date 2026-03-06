# WangAgent

A reinforcement learning agent trained to play Go on a 5×5 board. Named after a character from Arknights.

## Overview

WangAgent is an intelligent Go-playing agent that combines multiple AI techniques to learn and play Go effectively on a smaller 5×5 board. The agent uses:

- **Q-Learning** with a linear function approximator for decision-making
- **Monte Carlo Tree Search (MCTS)** for enhanced move selection
- **Feature extraction** to understand board state and move consequences
- **Progressive training** with curriculum learning from weaker opponents to stronger ones

## Project Structure

```
WangGoAgent/
├── my_player3.py              # Main Q-learning agent with LinearQFunction
├── my_trainer.py              # Training utilities and game management
├── training/                  # Training pipeline with progressive difficulty
├── MCTS_module/               # Monte Carlo Tree Search implementation
│   ├── mcts.py               # MCTS core algorithm
│   ├── go_mcts_board.py      # Go board representation for MCTS
│   └── mcts_test.py          # MCTS tests
├── host.py                    # Go game implementation (5×5 board)
├── go_helper.py               # Board logic utilities (liberties, groups, moves)
├── feature_extract_module.py  # State and action feature extraction
├── log_module.py              # Logging and diagnostics
├── visualize.py               # Training performance visualization
├── visualize_training.py      # Additional training metrics visualization
├── train.py                   # Main training entry point
├── random_player.py           # Random baseline player
└── logs/                      # Training logs and board states
```

## Key Features

### Q-Learning Agent (`my_player3.py`)
- **LinearQFunction**: Learns 28 state and action features including:
  - Board control (center, edge, corners)
  - Stone safety (liberties, groups)
  - Tactical features (captures, pre-atari detection)
  - Eye potential and cutting points
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation during training
- **Move Selection**: Uses learned Q-values with occasional MCTS guidance

### MCTS Module (`MCTS_module/`)
- **UCB-based Node Selection**: Balances exploration and exploitation
- **Tactical Awareness**: Enhanced bonuses for capture and defensive moves
- **Parallel Playouts**: Monte Carlo simulations for deep lookahead
- **Integration**: Works alongside Q-learning for improved decision-making

### Training Pipeline (`training/`)
- **Progressive Training**: Multi-stage curriculum:
  1. Learn against random player
  2. Self-play matches
  3. Face improving versions of itself
- **Configurable Parameters**: Epsilon decay, learning rate, discount factor, and more
- **Performance Tracking**: Win rate, episode rewards, and Q-value statistics

### Board Representation (`host.py`, `go_helper.py`)
- Full Go rule implementation for 5×5 boards:
  - Legal move validation
  - Capture detection
  - Suicide rule enforcement
  - Ko rule prevention
  - Komi scoring (board_size / 2)

### Feature Extraction (`feature_extract_module.py`)
Extracts relevant features from board states and moves:
- **State Features** (14): Player/opponent stones, liberties, groups, territory, etc.
- **Advanced Features** (6): Pre-atari, cutting points, eye potential
- **Action Features** (8): Capture value, liberty changes, connectivity, distance metrics

## Getting Started

### Prerequisites
- Python 3.8+
- NumPy
- Matplotlib (for visualization)

### Training the Agent

```bash
python train.py
```

This will:
1. Create a new Q-learning agent
2. Run progressive training through multiple stages
3. Generate training logs in `logs/` directory
4. Save results and performance metrics

### Testing Against Random Opponent

```bash
python random_player.py
```

### Visualizing Training Results

```bash
python visualize.py
python visualize_training.py
```

## Agent Architecture

```
Input: Board State (5×5 grid)
  ↓
Feature Extraction (28 features)
  ↓
Q-Function Evaluation (Linear combination of features)
  ↓
Move Selection (Epsilon-greedy or MCTS enhanced)
  ↓
Output: Move (position or pass)
```

## Training Configuration

Key hyperparameters in `training/`:
- **Initial Epsilon**: 1.0 (full exploration)
- **Epsilon Decay**: 0.995 per episode
- **Learning Rate (Alpha)**: 0.1
- **Discount Factor (Gamma)**: 0.99
- **Episodes per Stage**: Progressive increases

## Game Rules

WangGoAgent plays standard Go rules on a 5×5 board:
- Players alternate placing stones
- Captured stones are removed when surrounded
- Suicide moves are illegal
- Ko rule prevents immediate recapture
- Game ends when both players pass
- Score = stones + territory + komi (board_size/2)

## Performance

Training tracks:
- **Win Rate**: Against opponents of increasing strength
- **Episode Rewards**: Cumulative score per game
- **Epsilon Decay**: Exploration schedule
- **Q-Value Statistics**: Average and distribution of learned values

## Files Not Included

The following directories are excluded from version control:
- `MCTS_Module/` - Compiled cache and test outputs
- `logs/` - Training session logs
- `results/` - Saved results and metrics
- `submission_agent/` - Private submission materials

## Future Improvements

- [ ] Neural network function approximation (replace linear Q-function)
- [ ] Larger board support (7×7, 9×9)
- [ ] Policy gradient methods
- [ ] Multi-agent competitive training
- [ ] Distributed training for faster convergence

## License

This is a personal project for Go AI research.

## Acknowledgments

- Named after Wang from Arknights
- Inspired by AlphaGo's combination of neural networks and MCTS
- Go rule implementation based on standard conventions

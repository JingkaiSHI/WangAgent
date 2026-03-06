import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingLogger:
    """Comprehensive logging system for Go agent training."""
    
    def __init__(self, log_dir="logs", viz_dir="visualizations"):
        # Create directories
        self.log_dir = log_dir
        self.viz_dir = viz_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Initialize timestamp for log files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize log files
        self.episode_log_path = f"{log_dir}/episodes_{self.timestamp}.txt"
        self.weights_log_path = f"{log_dir}/weights_{self.timestamp}.txt"
        self.board_log_path = f"{log_dir}/boards_{self.timestamp}.txt"
        
        # Create header for files
        with open(self.episode_log_path, 'w') as f:
            f.write("# Training Episode Log\n")
            f.write("phase,episode,total_steps,result,reward,q_value\n")
            
        with open(self.weights_log_path, 'w') as f:
            f.write("# Q-Function Weights Log\n")
            f.write("phase,episode,step,weights\n")
            
        with open(self.board_log_path, 'w') as f:
            f.write("# Board States Log\n")
            f.write("phase,episode,step,player,board,action,q_value\n")
            
    def log_episode(self, phase, episode_num, total_steps, result, reward, q_value):
        """Log episode results."""
        with open(self.episode_log_path, 'a') as f:
            f.write(f"{phase},{episode_num},{total_steps},{result},{reward},{q_value}\n")
    
    def log_weights(self, phase, episode_num, step, weights):
        """Log Q-function weights."""
        with open(self.weights_log_path, 'a') as f:
            weights_str = ",".join([f"{w:.6f}" for w in weights])
            f.write(f"{phase},{episode_num},{step},[{weights_str}]\n")
            
            # Occasionally visualize weight distribution
            if episode_num % 500 == 0:
                self.visualize_weights(weights, phase, episode_num)
    
    def log_board(self, phase, episode_num, step, player, board, action, q_value):
        """Log board state and action."""
        board_str = json.dumps(board)
        with open(self.board_log_path, 'a') as f:
            f.write(f"{phase},{episode_num},{step},{player},{board_str},{action},{q_value}\n")
            
        # Occasionally visualize board states
        if episode_num % 500 == 0 and step <= 5:  # Log early moves in milestone episodes
            self.visualize_board(board, action, phase, episode_num, step)
    

    def visualize_weights(self, weights, phase, episode_num):
        """Create visualization of weight distribution."""
        plt.figure(figsize=(14, 10))
        
        # Check number of weights to determine which feature set is being used
        if len(weights) == 20:  # Original feature set
            # Original feature names for readability
            feature_names = [
                # State features (first 14)
                "S:Player_Stones", "S:Opponent_Stones", "S:Player_Liberties", 
                "S:Opponent_Liberties", "S:Player_Groups", "S:Opponent_Groups",
                "S:Center_Control", "S:Edge_Control", "S:Corner_Control",
                "S:Player_Territory", "S:Opponent_Territory", "S:Ko_Potential",
                "S:Last_Move_Distance", "S:Board_Fill_Ratio",
                # Action features (last 6)
                "A:Captures", "A:Self_Atari", "A:Distance_From_Center", 
                "A:Proximity_To_Own", "A:Proximity_To_Enemy", "A:Is_Pass"
            ]
            
            # Plot state features and action features separately
            state_weights = weights[:14]
            action_weights = weights[14:]
            
            plt.subplot(2, 1, 1)
            plt.bar(range(len(state_weights)), state_weights)
            plt.xticks(range(len(state_weights)), feature_names[:14], rotation=45, ha="right")
            plt.title(f"State Feature Weights (Phase {phase}, Episode {episode_num})")
            plt.grid(axis='y')
            
            plt.subplot(2, 1, 2)
            plt.bar(range(len(action_weights)), action_weights)
            plt.xticks(range(len(action_weights)), feature_names[14:], rotation=45, ha="right")
            plt.title("Action Feature Weights")
            plt.grid(axis='y')
            
        elif len(weights) == 28:  # Enhanced feature set
            # Updated feature names for the enhanced set
            feature_names = [
                # Original state features (14)
                "S:Player_Stones", "S:Opponent_Stones", "S:Player_Liberties", 
                "S:Opponent_Liberties", "S:Player_Groups", "S:Opponent_Groups",
                "S:Center_Control", "S:Edge_Control", "S:Corner_Control",
                "S:Player_Territory", "S:Opponent_Territory", "S:Ko_Potential",
                "S:Last_Move_Distance", "S:Board_Fill_Ratio",
                
                # NEW state features (6)
                "S:My_Pre_Atari", "S:Opp_Pre_Atari", "S:My_Cutting_Points",
                "S:Opp_Cutting_Points", "S:My_Eye_Potential", "S:Opp_Eye_Potential",
                
                # Original action features (6)
                "A:Captures", "A:Liberty", "A:Connects", 
                "A:Is_Center", "A:Is_Edge", "A:Distance_From_Center",
                
                # NEW action features (2)
                "A:Self_Atari_Move", "A:Protects_Vulnerable"
            ]
            
            # Divide into 3 charts for better visibility
            state_weights_original = weights[:14]
            state_weights_new = weights[14:20]  # 6 new state features
            action_weights = weights[20:]  # 8 action features
            
            plt.subplot(3, 1, 1)
            plt.bar(range(len(state_weights_original)), state_weights_original)
            plt.xticks(range(len(state_weights_original)), feature_names[:14], rotation=45, ha="right")
            plt.title(f"Original State Feature Weights (Phase {phase}, Episode {episode_num})")
            plt.grid(axis='y')
            
            plt.subplot(3, 1, 2)
            plt.bar(range(len(state_weights_new)), state_weights_new)
            plt.xticks(range(len(state_weights_new)), feature_names[14:20], rotation=45, ha="right")
            plt.title("New State Feature Weights")
            plt.grid(axis='y')
            
            plt.subplot(3, 1, 3)
            plt.bar(range(len(action_weights)), action_weights)
            plt.xticks(range(len(action_weights)), feature_names[20:], rotation=45, ha="right")
            plt.title("Action Feature Weights")
            plt.grid(axis='y')
        
        else:
            # For any other weight count, just use generic labels
            plt.bar(range(len(weights)), weights)
            plt.xticks(range(len(weights)), [f"Feature {i}" for i in range(len(weights))], rotation=45, ha="right")
            plt.title(f"Q-Function Weights (Phase {phase}, Episode {episode_num})")
            plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/weights_phase{phase}_ep{episode_num}.png")
        plt.close()
    
    def visualize_board(self, board, action, phase, episode_num, step):
        """Create visualization of board state and selected move."""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Draw board grid
        for i in range(6):
            ax.axhline(y=i, color='black', linewidth=1)
            ax.axvline(x=i, color='black', linewidth=1)
        
        # Draw stones
        for i in range(5):
            for j in range(5):
                if board[i][j] == 1:  # Black
                    ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.4, color='black'))
                elif board[i][j] == 2:  # White
                    ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.4, color='white', edgecolor='black'))
        
        # Highlight selected move if not PASS
        if action != "PASS" and isinstance(action, tuple):
            i, j = action
            ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.2, color='red'))
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.invert_yaxis()  # Invert y-axis to match board coordinates
        ax.set_title(f"Phase {phase}, Episode {episode_num}, Step {step}")
        
        # Save the visualization
        plt.savefig(f"{self.viz_dir}/board_phase{phase}_ep{episode_num}_step{step}.png")
        plt.close()
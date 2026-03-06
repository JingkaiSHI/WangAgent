import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_episode_log(log_file):
    """Load and process episode log data."""
    return pd.read_csv(log_file, skiprows=1)

# Replace the load_weights_log function:

# Replace the load_weights_log function:

def load_weights_log(log_file):
    """Load and process weights log data."""
    try:
        # First try standard CSV parsing
        df = pd.read_csv(log_file, skiprows=1)
        
        # Parse weight vectors from string
        def parse_weights(weight_str):
            try:
                # Remove brackets and split by comma
                return np.array([float(w) for w in weight_str.strip('[]').split(',')])
            except:
                # Return empty array if parsing fails
                return np.array([])
        
        df['parsed_weights'] = df['weights'].apply(parse_weights)
        return df
        
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        # Create an empty dataframe with required columns as fallback
        return pd.DataFrame(columns=['phase', 'episode', 'step', 'weights', 'parsed_weights'])

# Replace the plot_weight_evolution function with this more robust version:

def plot_weight_evolution(weights_df, feature_names=None):
    """Plot how weights evolve over training."""
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(20)]
    
    # Check if dataframe has expected columns
    if 'episode' not in weights_df.columns or 'parsed_weights' not in weights_df.columns:
        print("Error: Weight dataframe missing required columns")
        return
    
    # Extract weights for each feature across episodes
    feature_weights = {name: [] for name in feature_names}
    episodes = sorted(weights_df['episode'].unique())
    
    # Debug info
    print(f"Found {len(episodes)} unique episodes in the data")
    if not episodes:
        print("Error: No episodes found in the data")
        return
    
    valid_features = []
    
    # First, check which features have data across all episodes
    for name in feature_names:
        has_data = True
        for episode in episodes:
            episode_rows = weights_df[weights_df['episode'] == episode]
            if episode_rows.empty:
                print(f"Warning: No data for episode {episode}")
                has_data = False
                break
                
            # Get first row for this episode
            first_row = episode_rows.iloc[0]
            weights_array = first_row['parsed_weights']
            
            # Map feature name to index
            try:
                idx = feature_names.index(name)
                if idx >= len(weights_array):
                    has_data = False
                    break
            except ValueError:
                has_data = False
                break
        
        if has_data:
            valid_features.append(name)
    
    print(f"Found {len(valid_features)} features with complete data")
    
    # Now extract the weights for valid features
    for episode in episodes:
        episode_rows = weights_df[weights_df['episode'] == episode]
        if episode_rows.empty:
            continue
            
        episode_weights = episode_rows.iloc[0]['parsed_weights']
        
        for name in valid_features:
            idx = feature_names.index(name)
            if idx < len(episode_weights):
                feature_weights[name].append(episode_weights[idx])
    
    # Plot only features with data
    plt.figure(figsize=(14, 10))
    
    # Group features by type for clearer visualization
    feature_groups = [
        {'name': 'Material', 'features': ['S:Player_Stones', 'S:Opponent_Stones', 'S:Player_Groups', 'S:Opponent_Groups']},
        {'name': 'Liberty', 'features': ['S:Player_Liberties', 'S:Opponent_Liberties']},
        {'name': 'Position', 'features': ['S:Center_Control', 'S:Edge_Control', 'S:Corner_Control']},
        {'name': 'Territory', 'features': ['S:Player_Territory', 'S:Opponent_Territory', 'S:Board_Fill_Ratio']},
        {'name': 'Other State', 'features': ['S:Ko_Potential', 'S:Last_Move_Distance']},
        {'name': 'Tactical', 'features': ['A:Captures', 'A:Self_Atari', 'A:Distance_From_Center', 
                                        'A:Proximity_To_Own', 'A:Proximity_To_Enemy', 'A:Is_Pass']}
    ]
    
    # Use a different color for each feature group
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_groups)))
    
    for i, group in enumerate(feature_groups):
        for j, name in enumerate(group['features']):
            if name in valid_features and feature_weights[name]:
                plt.plot(episodes, feature_weights[name], 
                         label=name, 
                         color=colors[i],
                         linestyle=['-', '--', ':', '-.'][j % 4],
                         linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.title('Feature Weight Evolution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("visualizations/weight_evolution.png")
    plt.close()

def plot_phase_performance(episode_df):
    """Plot performance metrics by training phase."""
    # Group by phase and calculate metrics
    phase_metrics = episode_df.groupby('phase').agg({
        'result': 'mean',  # Win rate
        'reward': 'mean',  # Average reward
        'q_value': 'mean'  # Average Q-value
    }).reset_index()
    
    # Plot metrics
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 3, 1)
    plt.bar(phase_metrics['phase'], phase_metrics['result'])
    plt.xlabel('Phase')
    plt.ylabel('Win Rate')
    plt.title('Win Rate by Phase')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    
    plt.subplot(1, 3, 2)
    plt.bar(phase_metrics['phase'], phase_metrics['reward'])
    plt.xlabel('Phase')
    plt.ylabel('Average Reward')
    plt.title('Reward by Phase')
    plt.grid(axis='y')
    
    plt.subplot(1, 3, 3)
    plt.bar(phase_metrics['phase'], phase_metrics['q_value'])
    plt.xlabel('Phase')
    plt.ylabel('Average Q-Value')
    plt.title('Q-Value by Phase')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig("visualizations/phase_performance.png")
    plt.close()

# Replace the single log selection with this code:

# In the main function, replace how we process the weight logs:

def main():
    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Find log files
    log_dir = "logs"
    episode_logs = [f for f in os.listdir(log_dir) if f.startswith("episodes_")]
    weights_logs = [f for f in os.listdir(log_dir) if f.startswith("weights_")]
    
    if not episode_logs or not weights_logs:
        print("No log files found")
        return
    
    # Sort logs by timestamp to maintain phase order
    weights_logs.sort()
    
    print(f"Found {len(weights_logs)} weight log files")
    
    # Process each weight file as a separate phase
    all_weights_data = []
    for phase_num, log_file in enumerate(weights_logs, 1):
        print(f"Processing: {log_file} as Phase {phase_num}")
        weights_df = load_weights_log(os.path.join(log_dir, log_file))
        
        # Explicitly set the phase number based on file order
        weights_df['phase'] = phase_num
        
        all_weights_data.append(weights_df)
    
    # Combine all dataframes
    combined_weights_df = pd.concat(all_weights_data, ignore_index=True)
    print(f"Combined data has {len(combined_weights_df)} rows")
    
    # Check how many unique phases we have
    phases = sorted(combined_weights_df['phase'].unique())
    print(f"Found {len(phases)} unique phases: {phases}")
    
    # For episodes, just use the latest for performance metrics
    latest_episode_log = sorted(episode_logs)[-1]
    episode_df = load_episode_log(os.path.join(log_dir, latest_episode_log))
    
    # Feature names for interpretation
    feature_names = [
        # State features
        "S:Player_Stones", "S:Opponent_Stones", "S:Player_Liberties", 
        "S:Opponent_Liberties", "S:Player_Groups", "S:Opponent_Groups",
        "S:Center_Control", "S:Edge_Control", "S:Corner_Control",
        "S:Player_Territory", "S:Opponent_Territory", "S:Ko_Potential",
        "S:Last_Move_Distance", "S:Board_Fill_Ratio",
        # Action features
        "A:Captures", "A:Self_Atari", "A:Distance_From_Center", 
        "A:Proximity_To_Own", "A:Proximity_To_Enemy", "A:Is_Pass"
    ]
    
    # Generate visualizations
    plot_weight_evolution(combined_weights_df, feature_names)
    plot_phase_performance(episode_df)
    plot_comprehensive_weight_evolution(combined_weights_df, feature_names)
    
    print("Analysis complete. Visualizations saved.")
    
# Add this new function after your existing plot_weight_evolution function

# Fix the plot_comprehensive_weight_evolution function:

def plot_comprehensive_weight_evolution(weights_df, feature_names=None):
    """Create a comprehensive visualization of weight evolution across all phases."""
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(20)]
    
    # Check if dataframe has expected columns
    if 'phase' not in weights_df.columns or 'episode' not in weights_df.columns or 'parsed_weights' not in weights_df.columns:
        print("Error: Weight dataframe missing required columns for comprehensive plotting")
        return
    
    # Get unique phases and prepare for sequential plotting
    phases = sorted(weights_df['phase'].unique())
    
    if not phases:
        print("Error: No phase information found in weight data")
        return
        
    print(f"Found {len(phases)} unique phases: {phases}")
    
    # Process data to create continuous episode numbering across phases
    all_episodes = []
    phase_boundaries = [0]  # Start with 0
    phase_data = {}
    
    # First, collect all episode numbers and weights
    for phase in phases:
        phase_df = weights_df[weights_df['phase'] == phase]
        phase_data[phase] = phase_df
        phase_episodes = sorted(phase_df['episode'].unique())
        print(f"Phase {phase}: Found {len(phase_episodes)} episodes")
        
        # Adjust episode numbers to be continuous across phases
        if all_episodes:
            phase_episodes = [ep + all_episodes[-1] + 100 for ep in phase_episodes]  # Gap between phases
        
        all_episodes.extend(phase_episodes)
        phase_boundaries.append(len(all_episodes))
    
    # Prepare weight data across all phases
    feature_weights = {name: [] for name in feature_names}
    episode_markers = []
    
    # Collect all valid weights for each feature
    valid_features = []
    for name in feature_names:
        valid = True
        for phase in phases:
            phase_df = phase_data[phase]
            # Check if this feature has any valid data in this phase
            if phase_df.empty:
                valid = False
                break
                
            # Check first row as sample
            first_row = phase_df.iloc[0]
            weights_array = first_row['parsed_weights']
            idx = feature_names.index(name)
            if idx >= len(weights_array):
                valid = False
                break
        
        if valid:
            valid_features.append(name)
    
    print(f"Found {len(valid_features)} features with complete data")
    
    # Now construct the episode markers and feature weights
    for phase in phases:
        phase_df = phase_data[phase]
        phase_episodes = sorted(phase_df['episode'].unique())
        base_offset = 0 if len(episode_markers) == 0 else episode_markers[-1] + 100
        
        for episode in phase_episodes:
            episode_rows = phase_df[phase_df['episode'] == episode]
            if episode_rows.empty:
                continue
                
            episode_weights = episode_rows.iloc[0]['parsed_weights']
            episode_markers.append(base_offset + episode)
            
            for name in valid_features:
                idx = feature_names.index(name)
                if idx < len(episode_weights):
                    feature_weights[name].append(episode_weights[idx])
    
    # Create plots by feature category
    fig = plt.figure(figsize=(20, 16))
    
    # Define feature groups for organized visualization
    feature_groups = [
        # Group 1: Material-related features
        {
            'title': 'Material Features',
            'features': ['S:Player_Stones', 'S:Opponent_Stones', 'S:Player_Groups', 'S:Opponent_Groups'],
            'position': 1
        },
        # Group 2: Liberty-related features
        {
            'title': 'Liberty & Territory Features',
            'features': ['S:Player_Liberties', 'S:Opponent_Liberties', 'S:Player_Territory', 'S:Opponent_Territory', 'S:Board_Fill_Ratio'],
            'position': 2
        },
        # Group 3: Positional features
        {
            'title': 'Positional Features',
            'features': ['S:Center_Control', 'S:Edge_Control', 'S:Corner_Control', 'S:Last_Move_Distance', 'S:Ko_Potential'],
            'position': 3
        },
        # Group 4: Action features
        {
            'title': 'Tactical Features',
            'features': ['A:Captures', 'A:Self_Atari', 'A:Distance_From_Center', 'A:Proximity_To_Own', 'A:Proximity_To_Enemy', 'A:Is_Pass'],
            'position': 4
        }
    ]
    
    for group in feature_groups:
        ax = fig.add_subplot(2, 2, group['position'])
        
        for feature in group['features']:
            if feature in valid_features and len(feature_weights[feature]) > 0:
                ax.plot(episode_markers, feature_weights[feature], label=feature, linewidth=2)
        
        # Add phase boundary vertical lines
        for phase_idx in range(1, len(phases)):
            boundary_episode = episode_markers[phase_boundaries[phase_idx]-1]
            ax.axvline(x=boundary_episode, color='red', linestyle='--', alpha=0.7)
            ax.text(boundary_episode + 50, ax.get_ylim()[1] * 0.9, f"Phase {phase_idx+1}", 
                   horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(group['title'], fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add phase background colors
        colors = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']
        for p_idx, phase in enumerate(phases):
            if p_idx < len(phases)-1 and p_idx < len(phase_boundaries)-1:
                start = episode_markers[phase_boundaries[p_idx]] if p_idx < len(phase_boundaries)-1 and phase_boundaries[p_idx] < len(episode_markers) else 0
                end = episode_markers[phase_boundaries[p_idx+1]-1] if p_idx+1 < len(phase_boundaries) and phase_boundaries[p_idx+1]-1 < len(episode_markers) else 0
                if start < end:  # Ensure valid span
                    ax.axvspan(start-50, end+50, alpha=0.15, color=colors[p_idx % len(colors)])
    
    plt.suptitle('Complete Evolution of Weights Across All Training Phases', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig("visualizations/complete_weight_evolution.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
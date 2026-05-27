import sys
import os
import random
import torch
import matplotlib.pyplot as plt

# Append parent directory to sys.path for robust modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.packing_env import Packing
from agents.q_agent import QLearning, train_q_episode
from utils.data import load_2dvsbpp_instance, scan_dataset_limits, split_dataset_files
from utils.io import save_torch_checkpoint

def save_agent(agent, save_path):
    state_dict = {
        'qtable': agent.q_table,
        'gamma': agent.gamma,
        'lr': agent.alpha
    }
    save_torch_checkpoint(state_dict, save_path)

if __name__ == '__main__':
    # === Resolve Root Paths ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    dataset_dir = os.path.join(root_dir, '2dvsbpp')
    checkpoint_path = os.path.join(root_dir, 'ql_train.pth')

    # === Config ===
    episodes = 1000

    # === Scan and Split Dataset ===
    max_w, max_h, max_items, max_bin_types, dataset_files = scan_dataset_limits(dataset_dir)
    if not dataset_files:
        raise FileNotFoundError(f"No valid dataset files (.txt) found in: {dataset_dir}")
    
    train_files, val_files = split_dataset_files(dataset_files, train_ratio=0.8, seed=42)
    print(f"Dataset Split -> Total: {len(dataset_files)}, Train: {len(train_files)} (80%), Val: {len(val_files)} (20%)")
    print(f"Dataset Limits -> Max Width: {max_w}, Max Height: {max_h}, Max Items: {max_items}, Max Bin Types: {max_bin_types}")

    # === Initialize Q-Learning Agent ===
    agent = QLearning(num_items=max_items)

    # === Resume Training if checkpoint exists ===
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}' to resume training...")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            agent.q_table = checkpoint['qtable']
            agent.gamma = checkpoint.get('gamma', agent.gamma)
            agent.alpha = checkpoint.get('lr', agent.alpha)
            print(f"Checkpoint loaded successfully with {len(agent.q_table)} state-action pairs. Resuming...")
        except Exception as e:
            print(f"Failed to load checkpoint to resume: {e}. Starting training from scratch.")

    rewards = []
    best_rw = float('-inf')
    best_env_state = None

    print("Training Q-Learning...")
    for episode in range(episodes):
        file_path = random.choice(train_files)
        file_name = os.path.basename(file_path)
        bins, items = load_2dvsbpp_instance(file_path)
        
        env = Packing(bin_types=bins, items_or_height=items, max_width=max_w, max_height=max_h, max_items=max_items)
        agent.epsilon = max(0.1, 1 - episode / (episodes * 0.9))  # Decrease epsilon gradually
        total_rw = train_q_episode(env, agent)

        if total_rw > best_rw:
            best_rw = total_rw
            best_env_state = {
                'placed_items': env.placed_items.copy(),
                'total_bin_cost': env.total_bin_cost,
                'width': env.width,
                'height': env.height,
                'bin_types': env.bin_types,
                'items': env.items,
                'file_name': file_name
            }

        rewards.append(total_rw)
        print(f"Episode {episode + 1}/{episodes} [{file_name}], Reward={total_rw:.2f}, Cost={env.total_bin_cost:.1f}, Bins={env.opened_bins_count}")
        save_agent(agent, save_path=checkpoint_path)

    # === Render Best Episode ===
    if best_env_state is not None:
        env = Packing(
            bin_types=best_env_state['bin_types'], 
            items_or_height=best_env_state['items'],
            max_width=max_w, 
            max_height=max_h, 
            max_items=max_items
        )
        env.placed_items = best_env_state['placed_items']
        env.total_bin_cost = best_env_state['total_bin_cost']
        env.width = best_env_state['width']
        env.height = best_env_state['height']
        
        print(f"\nBest Episode File: {best_env_state['file_name']}")
        print(f"Best reward: {best_rw:.2f}, Items placed: {len(env.placed_items)}, Bins opened: {env.opened_bins_count}, Cost: {env.total_bin_cost}")
        env.render()

    # === Plot training curve ===
    plt.figure()
    plt.plot(rewards)
    plt.title('Q-Learning - Reward Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.show()

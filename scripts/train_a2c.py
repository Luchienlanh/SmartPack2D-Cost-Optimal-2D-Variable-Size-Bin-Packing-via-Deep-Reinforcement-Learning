import sys
import os
import random
import torch
import matplotlib.pyplot as plt

# Append parent directory to sys.path for robust modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.packing_env import Packing
from agents.a2c_agent import A2CNetwork, train_a2c_episode
from utils.device import device
from utils.data import load_2dvsbpp_instance, scan_dataset_limits, split_dataset_files
from utils.io import save_torch_checkpoint
from utils.amp import AMP_ENABLED, load_scaler_state

def save_agent(agent, save_path):
    state_dict = {
        'actor_net_state_dict': agent.actor.state_dict(),
        'critic_net_state_dict': agent.critic.state_dict(),
        'frame_net_state_dict': agent.frame_net.state_dict(),
        'item_net_state_dict': agent.item_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'gamma': agent.gamma,
        'scaler_state_dict': agent.scaler.state_dict(),
    }
    save_torch_checkpoint(state_dict, save_path)

if __name__ == '__main__':
    # === Resolve Root Paths ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    dataset_dir = os.path.join(root_dir, '2dvsbpp')
    checkpoint_path = os.path.join(root_dir, 'a2c_train.pth')

    # === Config ===
    episodes = 100
    batch_size = 8

    # === Scan and Split Dataset ===
    max_w, max_h, max_items, max_bin_types, dataset_files = scan_dataset_limits(dataset_dir)
    if not dataset_files:
        raise FileNotFoundError(f"No valid dataset files (.txt) found in: {dataset_dir}")
    
    train_files, val_files = split_dataset_files(dataset_files, train_ratio=0.8, seed=42)
    print(f"Dataset Split -> Total: {len(dataset_files)}, Train: {len(train_files)} (80%), Val: {len(val_files)} (20%)")
    print(f"Dataset Limits -> Max Width: {max_w}, Max Height: {max_h}, Max Items: {max_items}, Max Bin Types: {max_bin_types}")

    action_size = (max_items * 2) + max_bin_types

    # === Initialize A2C Agent ===
    agent = A2CNetwork(
        height=max_h,
        width=max_w,
        action_size=action_size,
        num_items=max_items,
        gamma=0.99,
        lr=1e-4
    ).to(device)

    # === Resume Training if checkpoint exists ===
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}' to resume training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.actor.load_state_dict(checkpoint['actor_net_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_net_state_dict'])
            agent.frame_net.load_state_dict(checkpoint['frame_net_state_dict'])
            agent.item_net.load_state_dict(checkpoint['item_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            load_scaler_state(agent.scaler, checkpoint)
            print("Checkpoint loaded successfully. Resuming training...")
        except Exception as e:
            print(f"Failed to load checkpoint to resume: {e}. Starting training from scratch.")

    print(f"Training A2C on: {device} | AMP={'enabled' if AMP_ENABLED else 'disabled'}")
    rewards = []
    best_rw = float('-inf')
    best_env_state = None

    for ep in range(episodes):
        file_path = random.choice(train_files)
        file_name = os.path.basename(file_path)
        bins, items = load_2dvsbpp_instance(file_path)
        
        env = Packing(bin_types=bins, items_or_height=items, max_width=max_w, max_height=max_h, max_items=max_items)
        total_reward, sum_loss = train_a2c_episode(env, agent, batch_size=batch_size)
        
        if total_reward > best_rw:
            best_rw = total_reward
            best_env_state = {
                'placed_items': env.placed_items.copy(),
                'total_bin_cost': env.total_bin_cost,
                'width': env.width,
                'height': env.height,
                'bin_types': env.bin_types,
                'items': env.items,
                'file_name': file_name
            }

        rewards.append(total_reward)
        print(f"Episode {ep + 1}/{episodes} [{file_name}], Reward={total_reward:.2f}, Loss={sum_loss:.4f}, Cost={env.total_bin_cost:.1f}, Bins={env.opened_bins_count}")
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
    plt.plot(rewards, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("A2C - Reward over episodes")
    plt.grid()
    plt.show()

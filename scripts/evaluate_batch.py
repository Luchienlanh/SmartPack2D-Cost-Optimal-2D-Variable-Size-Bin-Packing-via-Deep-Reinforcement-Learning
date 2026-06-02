import sys
import os
import random
import time
import torch
import numpy as np

# Append parent directory to sys.path for robust modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.packing_env import Packing
from utils.device import device
from utils.data import load_2dvsbpp_instance, scan_dataset_limits, split_dataset_files
from utils.heuristics import run_ffd_heuristic
from utils.metrics import summarize_packing
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CNetwork
from agents.pg_agent import PolicyNetwork

def load_agent_safely(agent_class, path, max_w, max_h, max_items, max_bin_types):
    """Dynamically initializes and loads checkpoints for PPO, A2C, or PG agents."""
    if not os.path.exists(path):
        return None
    action_size = (max_items * 2) + max_bin_types
    agent = agent_class(height=max_h, width=max_w, action_size=action_size, num_items=max_items).to(device)
    try:
        ckpt = torch.load(path, map_location=device)
        for key, subnet in [('frame_net_state_dict', getattr(agent, 'frame_net', None)),
                            ('item_net_state_dict', getattr(agent, 'item_net', None)),
                            ('actor_state_dict', getattr(agent, 'actor', None)),
                            ('actor_net_state_dict', getattr(agent, 'actor', None)),
                            ('critic_state_dict', getattr(agent, 'critic', None)),
                            ('critic_net_state_dict', getattr(agent, 'critic', None)),
                            ('policy_state_dict', getattr(agent, 'policy_head', None))]:
            if key in ckpt and subnet is not None:
                subnet.load_state_dict(ckpt[key])
        agent.eval()
        print(f"Loaded checkpoint: {path}")
        return agent
    except Exception as e:
        print(f"Failed to load checkpoint '{path}': {e}")
        return None

def evaluate_rl_agent(agent_type, agent, file_path, max_w, max_h, max_items, max_bin_types):
    """Evaluates a trained RL agent on a single file using deterministic greedy actions."""
    bins, items = load_2dvsbpp_instance(file_path)
    env = Packing(bin_types=bins, items_or_height=items, max_width=max_w, max_height=max_h, max_items=max_items)
    action_space = [(i, rot) for i in range(max_items) for rot in [False, True]] + [("open", b_idx) for b_idx in range(len(bins))]
    
    start_time = time.time()
    max_steps = max(1, env.num_items * 3 + len(bins) * 2)
    steps = 0
    while not env.is_done() and steps < max_steps:
        steps += 1
        valid_idx = env.get_valid_actions(action_space)
        if not valid_idx:
            break
            
        frame, remain = env.get_state()
        frame_4d = frame.unsqueeze(0).unsqueeze(0).float().to(device)
        remain_2d = remain.view(1, -1).float().to(device)
        
        with torch.no_grad():
            logits = agent(frame_4d, remain_2d)[0] if agent_type in ['PPO', 'A2C'] else agent(frame_4d, remain_2d)
        logits = logits.squeeze(0)
        
        mask = torch.full_like(logits, -1e9)
        mask[valid_idx] = 0.0
        a_idx = torch.argmax(logits + mask).item()
        env.place(action_space[a_idx])
        
    exec_time = time.time() - start_time
    result = summarize_packing(env, total_items=len(items), exec_time=exec_time)
    result["truncated"] = steps >= max_steps and not env.is_done()
    return result

def print_summary_table(results):
    print("\n" + "="*80 + "\n                    📊 BATCH EVALUATION COMPARISON RESULTS\n" + "="*80)
    print("| Method / Solver | Avg Cost 📉 | Avg Bins 📦 | Space Util (%) 📈 | Placement Success % | Avg Time (s) ⚡ |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for name, m in results.items():
        if m is None:
            print(f"| **{name}** | *Checkpoint not found* | | | | |")
        else:
            print(f"| **{name}** | {m['cost']:.1f} | {m['bins_opened']:.2f} | {m['utilization']:.1f}% | {m['success_rate']:.1f}% | {m['time']:.4f}s |")
    print("="*80 + "\n")

if __name__ == '__main__':
    # === Resolve Root Paths ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    dataset_dir = os.path.join(root_dir, '2dvsbpp')

    max_w, max_h, max_items, max_bin_types, dataset_files = scan_dataset_limits(dataset_dir)
    if not dataset_files:
        print("No dataset files found!")
        exit(1)
        
    _, val_files = split_dataset_files(dataset_files, train_ratio=0.8, seed=42)
    eval_files = random.sample(val_files, min(len(val_files), 30))
    print(f"Evaluating solvers across {len(eval_files)} validation files...")
    
    # Initialize and load agents dynamically using root directory checkpoints
    agents = {
        'RL Agent (PPO)': load_agent_safely(PPOAgent, os.path.join(root_dir, 'ppo_model.pth'), max_w, max_h, max_items, max_bin_types),
        'RL Agent (A2C)': load_agent_safely(A2CNetwork, os.path.join(root_dir, 'a2c_train.pth'), max_w, max_h, max_items, max_bin_types),
        'RL Agent (Policy Gradient)': load_agent_safely(PolicyNetwork, os.path.join(root_dir, 'train_plc.pth'), max_w, max_h, max_items, max_bin_types)
    }
    
    solvers = {'FFD Heuristic': []}
    for name in agents:
        solvers[name] = []
        
    for idx, f_path in enumerate(eval_files):
        bins, items = load_2dvsbpp_instance(f_path)
        solvers['FFD Heuristic'].append(run_ffd_heuristic(bins, items, max_width=max_w, max_height=max_h, max_items=max_items))
        
        for name, agent in agents.items():
            if agent is not None:
                try:
                    solvers[name].append(evaluate_rl_agent(name.split()[-1][1:-1], agent, f_path, max_w, max_h, max_items, max_bin_types))
                except Exception as e:
                    print(f"Failed to evaluate {name} on {os.path.basename(f_path)}: {e}")
        if (idx + 1) % 5 == 0 or (idx + 1) == len(eval_files):
            print(f"Evaluated {idx + 1}/{len(eval_files)} files...")
            
    summary = {}
    for name, res_list in solvers.items():
        if not res_list:
            summary[name] = None
        else:
            summary[name] = {
                'cost': np.mean([r['cost'] for r in res_list]),
                'bins_opened': np.mean([r['bins_opened'] for r in res_list]),
                'utilization': np.mean([r['utilization'] for r in res_list]),
                'success_rate': np.mean([r['success_rate'] for r in res_list]),
                'time': np.mean([r['time'] for r in res_list])
            }
    print_summary_table(summary)

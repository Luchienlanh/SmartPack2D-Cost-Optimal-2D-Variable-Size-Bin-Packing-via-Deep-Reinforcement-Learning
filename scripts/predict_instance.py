import argparse
import json
import os
import sys
import time

import torch

# Append parent directory to sys.path for robust modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.a2c_agent import A2CNetwork
from agents.pg_agent import PolicyNetwork
from agents.ppo_agent import PPOAgent
from env.packing_env import Packing
from utils.data import load_2dvsbpp_instance, scan_dataset_limits
from utils.device import device
from utils.metrics import summarize_packing


AGENTS = {
    "pg": (PolicyNetwork, "train_plc.pth"),
    "policy_gradient": (PolicyNetwork, "train_plc.pth"),
    "a2c": (A2CNetwork, "a2c_train.pth"),
    "ppo": (PPOAgent, "ppo_model.pth"),
}


def load_agent(agent_name, checkpoint_path, max_h, max_w, max_items, max_bin_types):
    agent_class, _ = AGENTS[agent_name]
    action_size = (max_items * 2) + max_bin_types
    agent = agent_class(
        height=max_h,
        width=max_w,
        action_size=action_size,
        num_items=max_items,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    for key, module_name in [
        ("frame_net_state_dict", "frame_net"),
        ("item_net_state_dict", "item_net"),
        ("policy_state_dict", "policy_head"),
        ("actor_state_dict", "actor"),
        ("actor_net_state_dict", "actor"),
        ("critic_state_dict", "critic"),
        ("critic_net_state_dict", "critic"),
    ]:
        module = getattr(agent, module_name, None)
        if key in checkpoint and module is not None:
            module.load_state_dict(checkpoint[key])

    agent.eval()
    return agent


def predict(agent_name, agent, bins, items, max_w, max_h, max_items, max_steps=None):
    env = Packing(
        bin_types=bins,
        items_or_height=items,
        max_width=max_w,
        max_height=max_h,
        max_items=max_items,
    )
    action_space = [(i, rot) for i in range(max_items) for rot in [False, True]]
    action_space.extend(("open", b_idx) for b_idx in range(len(bins)))

    if max_steps is None:
        max_steps = max(1, env.num_items * 3 + len(bins) * 2)

    chosen_actions = []
    start_time = time.time()
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
            output = agent(frame_4d, remain_2d)
            logits = output[0] if agent_name in {"ppo", "a2c"} else output
            logits = logits.squeeze(0).float()

        mask = torch.full_like(logits, -1e9)
        mask[valid_idx] = 0.0
        action_idx = torch.argmax(logits + mask).item()
        action = action_space[action_idx]
        success, reward = env.place(action)
        chosen_actions.append((action, success, reward))

    metrics = summarize_packing(env, total_items=len(items), exec_time=time.time() - start_time)
    metrics["steps"] = steps
    metrics["truncated"] = steps >= max_steps and not env.is_done()
    return env, metrics, chosen_actions


def print_report(instance_path, checkpoint_path, agent_name, env, metrics, actions, show_actions):
    print(f"Agent: {agent_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Instance: {instance_path}")
    print(f"Device: {device}")
    print(
        "Metrics: "
        f"cost={metrics['cost']:.1f}, "
        f"bins={metrics['bins_opened']}, "
        f"utilization={metrics['utilization']:.2f}%, "
        f"success={metrics['success_rate']:.1f}%, "
        f"placed={metrics['placed_count']}/{metrics['total_count']}, "
        f"steps={metrics['steps']}, "
        f"time={metrics['time']:.4f}s"
    )
    if metrics["truncated"]:
        print("Warning: prediction stopped at max_steps before all items were placed.")

    print("\nOpened bins:")
    for idx, bin_cfg in enumerate(env.opened_bins, start=1):
        print(
            f"  Bin {idx}: "
            f"{bin_cfg['width']}x{bin_cfg['height']}, "
            f"cost={bin_cfg['cost']}"
        )

    print("\nPlacements:")
    print("  item | bin | x | y | h | w | rotated")
    for item_idx, x, y, h, w, rotated, bin_number in env.placed_items:
        print(f"  {item_idx:4d} | {bin_number:3d} | {x:3d} | {y:3d} | {h:3d} | {w:3d} | {rotated}")

    if show_actions:
        print("\nActions:")
        for step, (action, success, reward) in enumerate(actions, start=1):
            print(f"  {step:4d}: action={action}, success={success}, reward={reward:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run a trained SmartPack2D checkpoint on one 2DVSBPP instance.")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="ppo")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--dataset-dir", default="2dvsbpp")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text report.")
    parser.add_argument("--show-actions", action="store_true")
    parser.add_argument("--render", action="store_true", help="Show matplotlib render after printing prediction.")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(root_dir, dataset_dir)

    instance_path = args.instance
    if not os.path.isabs(instance_path):
        instance_path = os.path.join(root_dir, instance_path)

    _, default_checkpoint = AGENTS[args.agent]
    checkpoint_path = args.checkpoint or os.path.join(root_dir, default_checkpoint)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(root_dir, checkpoint_path)

    max_w, max_h, max_items, max_bin_types, _ = scan_dataset_limits(dataset_dir)
    bins, items = load_2dvsbpp_instance(instance_path)
    agent = load_agent(args.agent, checkpoint_path, max_h, max_w, max_items, max_bin_types)
    env, metrics, actions = predict(
        args.agent,
        agent,
        bins,
        items,
        max_w=max_w,
        max_h=max_h,
        max_items=max_items,
        max_steps=args.max_steps,
    )

    if args.json:
        print(json.dumps({
            "agent": args.agent,
            "checkpoint": checkpoint_path,
            "instance": instance_path,
            "metrics": {k: v for k, v in metrics.items() if k not in {"opened_bins", "placed_items"}},
            "opened_bins": env.opened_bins,
            "placements": [
                {
                    "item": int(item_idx),
                    "bin": int(bin_number),
                    "x": int(x),
                    "y": int(y),
                    "height": int(h),
                    "width": int(w),
                    "rotated": bool(rotated),
                }
                for item_idx, x, y, h, w, rotated, bin_number in env.placed_items
            ],
        }, indent=2))
    else:
        print_report(instance_path, checkpoint_path, args.agent, env, metrics, actions, args.show_actions)

    if args.render:
        env.render()


if __name__ == "__main__":
    main()

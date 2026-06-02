import torch
import torch.nn as nn
import torch.optim as optim
from utils.amp import autocast_context, create_grad_scaler
from utils.device import device

class PolicyNetwork(nn.Module):
    def __init__(self, height, width, action_size, num_items, gamma=0.99, lr=1e-4):
        super().__init__()
        self.height = height
        self.width = width
        self.action_size = action_size
        self.num_items = num_items
        self.gamma = gamma
        self.lr = lr
        
        # frame network
        self.frame_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        ).to(device)

        # items network
        self.item_net = nn.Sequential(
            nn.Linear(num_items * 2, 128),
            nn.ReLU()
        ).to(device)

        # combined 2 features network
        self.combined_size = self._calc_combined_size()
        self.policy_head = nn.Sequential(
            nn.Linear(self.combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scaler = create_grad_scaler()

    def _calc_combined_size(self):
        with torch.no_grad():
            dummy_frame = torch.zeros((1, 1, self.height, self.width), device=device)
            frame_out = self.frame_net(dummy_frame)
            frame_dim = frame_out.shape[1]

            dummy_items = torch.zeros((1, self.num_items * 2), device=device)
            item_out = self.item_net(dummy_items)
            item_dim = item_out.shape[1]

        return frame_dim + item_dim

    def forward(self, frame, items_tensor):
        """
        frame: (batch,1,H,W) trên GPU
        items_tensor: (batch, num_items*2) trên GPU
        return logits: (batch, action_size)
        """
        f_feat = self.frame_net(frame)
        i_feat = self.item_net(items_tensor)
        combined = torch.cat((f_feat, i_feat), dim=1)
        logits = self.policy_head(combined)
        return logits

    def select_action(self, state, valid_actions):
        frame, items = state
        frame = frame.unsqueeze(0).unsqueeze(0).float().to(device)     # (1,1,H,W)
        items = items.view(1, -1).float().to(device)                   # (1, N*2)

        with autocast_context():
            logits = self.forward(frame, items)
        logits = logits.float().squeeze(0)                             # (action_size,)
        probs = torch.softmax(logits, dim=-1)

        # Mask
        mask = torch.zeros_like(probs)
        if valid_actions:
            mask[valid_actions] = 1.0
        else:
            mask[:] = 1.0

        masked_probs = probs * mask
        if masked_probs.sum() <= 1e-8:
            masked_probs = mask
        masked_probs = masked_probs / masked_probs.sum()

        dist = torch.distributions.Categorical(masked_probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob

    def update_policy(self, log_probs, rewards):
        # Tính discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy loss
        policy_loss = []
        for lp, ret in zip(log_probs, returns):
            policy_loss.append(-lp * ret)
        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_one_episode(self, env, batch_size=16):
        max_items = getattr(env, 'max_items', env.num_items)
        bin_types = getattr(env, 'bin_types', [{'width': env.width, 'height': env.height, 'cost': 100.0}])
        action_space = [(i, rot) for i in range(max_items) for rot in [False, True]] + [("open", b_idx) for b_idx in range(len(bin_types))]
        env.reset()

        frame_cpu, items_cpu = env.get_state()  # CPU
        done = False
        total_reward = 0

        log_probs = []
        rewards = []

        while not done:
            valid_actions = env.get_valid_actions(action_space)
            if not valid_actions:
                break

            action_idx, log_prob = self.select_action((frame_cpu, items_cpu), valid_actions)
            success, reward = env.place(action_space[action_idx])

            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

            frame_cpu, items_cpu = env.get_state()  # CPU
            if len(log_probs) >= batch_size:
                self.update_policy(log_probs, rewards)
                log_probs = []
                rewards = []

            if env.is_done():
                done = True

        if log_probs:
            self.update_policy(log_probs, rewards)
        return total_reward

def train_pg_episode(env, agent, batch_size=16):
    """Functional wrapper for compatibility and consistency."""
    return agent.train_one_episode(env, batch_size=batch_size)

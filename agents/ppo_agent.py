import torch
import torch.nn as nn
import torch.optim as optim
from utils.amp import autocast_context, create_grad_scaler
from utils.device import device

class PPOAgent(nn.Module):
    def __init__(self, height, width, action_size, num_items,
                 gamma=0.99, lr=3e-4, eps_clip=0.2, K_epochs=4, gae_lambda=0.95):
        super().__init__()
        self.height = height
        self.width = width
        self.action_size = action_size
        self.num_items = num_items
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lr = lr
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        # --- Frame CNN (thêm MaxPool để giảm tham số) ---
        # Input 100x100 -> Pool -> 50x50 -> Pool -> 25x25 -> Pool -> 12x12
        self.frame_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        ).to(device)

        # --- Item MLP ---
        self.item_net = nn.Sequential(
            nn.Linear(num_items * 2, 128),
            nn.ReLU()
        ).to(device)

        self.combined_size = self._calc_combined_size()

        # --- Actor: output logits (KHÔNG có Softmax) ---
        self.actor = nn.Sequential(
            nn.Linear(self.combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(device)

        # --- Critic ---
        self.critic = nn.Sequential(
            nn.Linear(self.combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scaler = create_grad_scaler()
        self.MseLoss = nn.MSELoss()

    def _calc_combined_size(self):
        with torch.no_grad():
            dummy_frame = torch.zeros((1, 1, self.height, self.width), device=device)
            frame_dim = self.frame_net(dummy_frame).shape[1]
            dummy_items = torch.zeros((1, self.num_items * 2), device=device)
            item_dim = self.item_net(dummy_items).shape[1]
        return frame_dim + item_dim

    def forward(self, frame_4d, item_2d):
        """
        frame_4d: (batch, 1, H, W)
        item_2d:  (batch, num_items*2)
        returns: logits (batch, action_size), values (batch, 1)
        """
        f_feat = self.frame_net(frame_4d)
        i_feat = self.item_net(item_2d)
        combined = torch.cat((f_feat, i_feat), dim=1)
        logits = self.actor(combined)
        values = self.critic(combined)
        return logits, values

    def select_action(self, state, valid_actions):
        """
        Chọn action từ policy, dùng logit masking chuẩn.
        valid_actions: list các index hợp lệ trong action_space.
        """
        frame, remain = state
        frame_4d = frame.unsqueeze(0).unsqueeze(0).float().to(device)
        remain_2d = remain.view(1, -1).float().to(device)

        with torch.no_grad(), autocast_context():
            logits, val = self.forward(frame_4d, remain_2d)
        logits = logits.float()
        val = val.float()
        logits = logits.squeeze(0)  # (action_size,)
        val = val.squeeze(0)

        # --- Action Masking trên logit (chuẩn PPO) ---
        mask = torch.full_like(logits, -1e9)
        if valid_actions:
            mask[valid_actions] = 0.0
        else:
            mask[:] = 0.0  # fallback: cho phép tất cả
        masked_logits = logits + mask

        dist = torch.distributions.Categorical(logits=masked_logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob, val

    def evaluate(self, frames, items, actions, valid_masks):
        """
        Đánh giá batch cho PPO update.
        frames:      (B, 1, H, W)
        items:       (B, num_items*2)
        actions:     (B,)
        valid_masks: (B, action_size) - 0.0 cho valid, -1e9 cho invalid
        """
        with autocast_context():
            logits, values = self.forward(frames, items)
        logits = logits.float()
        values = values.float()
        masked_logits = logits + valid_masks
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Tính Generalized Advantage Estimation (GAE) cho toàn bộ buffer.
        rewards:    list[float]
        values:     list[float] (detached)
        dones:      list[bool]
        next_value: float (V(s_{T+1}), 0 nếu terminal)
        returns: advantages (Tensor), returns (Tensor)
        """
        gae = 0
        advantages = []
        values_ext = values + [next_value]

        for t in reversed(range(len(rewards))):
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * values_ext[t + 1] * non_terminal - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=device)
        return advantages, returns


class PPOMemory:
    """Buffer lưu trữ kinh nghiệm cho 1 episode."""
    def __init__(self):
        self.frames = []
        self.items = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.valid_masks = []

    def store(self, frame, item, action, log_prob, value, reward, done, valid_mask):
        self.frames.append(frame.detach().cpu())
        self.items.append(item.detach().cpu())
        self.actions.append(action)
        self.log_probs.append(log_prob.detach().cpu())
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.valid_masks.append(valid_mask.detach().cpu())

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


def train_ppo_episode(env, agent, memory, batch_size=32):
    """
    Chạy 1 episode, thu thập kinh nghiệm, sau đó cập nhật PPO chuẩn:
    1. Tính GAE trên toàn bộ buffer
    2. Lặp K_epochs
    3. Trong mỗi epoch, shuffle & chia mini-batch
    """
    max_items = getattr(env, 'max_items', env.num_items)
    bin_types = getattr(env, 'bin_types', [{'width': env.width, 'height': env.height, 'cost': 100.0}])
    action_space = [(i, rot) for i in range(max_items) for rot in [False, True]] + [("open", b_idx) for b_idx in range(len(bin_types))]
    env.reset()
    frame, remain = env.get_state()
    done = False
    total_reward = 0
    max_steps = max(1, env.num_items * 3 + len(bin_types) * 2)
    steps = 0
    memory.clear()

    # === Phase 1: Thu thập kinh nghiệm (Rollout) ===
    while not done and steps < max_steps:
        steps += 1
        valid_idx = env.get_valid_actions(action_space)
        if not valid_idx:
            done = True
            break

        # Tạo valid_mask cho logit masking
        valid_mask = torch.full((agent.action_size,), -1e9, device=device)
        valid_mask[valid_idx] = 0.0

        a_idx, log_prob, val = agent.select_action((frame, remain), valid_idx)
        action = action_space[a_idx]
        success, reward = env.place(action)

        next_frame, next_remain = env.get_state()

        # Lưu vào buffer (detach value để tính GAE)
        memory.store(
            frame=frame.unsqueeze(0).float(),       # (1, H, W)
            item=remain.view(-1).float(),            # (num_items*2,)
            action=a_idx,
            log_prob=log_prob.detach(),
            value=val.item(),
            reward=reward,
            done=False,
            valid_mask=valid_mask
        )

        frame, remain = next_frame, next_remain
        total_reward += reward
        done = env.is_done()

    if steps >= max_steps and not done:
        total_reward -= 10.0

    # Đánh dấu bước cuối cùng là terminal
    if len(memory) > 0:
        memory.dones[-1] = True

    # === Phase 2: Tính GAE cho toàn bộ buffer ===
    if len(memory) == 0:
        return total_reward, 0.0

    next_value = 0.0  # Terminal => V(s_T+1) = 0
    advantages, returns = agent.compute_gae(
        memory.rewards, memory.values, memory.dones, next_value
    )

    # Chuẩn hóa advantage (giảm variance)
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Chuyển buffer sang tensor
    buf_frames = torch.stack(memory.frames)       # (T, 1, H, W), kept on CPU
    buf_items = torch.stack(memory.items)         # (T, num_items*2), kept on CPU
    buf_actions = torch.tensor(memory.actions, dtype=torch.long)
    buf_old_logprobs = torch.stack(memory.log_probs)
    buf_valid_masks = torch.stack(memory.valid_masks)

    # === Phase 3: PPO Update - K epochs x mini-batches ===
    total_loss = 0.0
    num_samples = len(memory)

    for _ in range(agent.K_epochs):
        # Shuffle indices mỗi epoch
        perm = torch.randperm(num_samples)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            idx = perm[start:end]

            # Lấy mini-batch
            mb_frames = buf_frames[idx].to(device)
            mb_items = buf_items[idx].to(device)
            mb_actions = buf_actions[idx].to(device)
            mb_old_logprobs = buf_old_logprobs[idx].to(device)
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]
            mb_valid_masks = buf_valid_masks[idx].to(device)

            # Evaluate lại policy mới
            new_logprobs, state_values, dist_entropy = agent.evaluate(
                mb_frames, mb_items, mb_actions, mb_valid_masks
            )

            # PPO Clipped Surrogate Loss
            ratios = torch.exp(new_logprobs - mb_old_logprobs)
            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - agent.eps_clip, 1 + agent.eps_clip) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value Loss
            critic_loss = agent.MseLoss(state_values, mb_returns)

            # Entropy Bonus (khuyến khích khám phá)
            entropy_loss = -dist_entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            agent.optimizer.zero_grad(set_to_none=True)
            agent.scaler.scale(loss).backward()
            # Gradient clipping để ổn định huấn luyện
            agent.scaler.unscale_(agent.optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            agent.scaler.step(agent.optimizer)
            agent.scaler.update()

            total_loss += loss.item()

    return total_reward, total_loss

import torch
import torch.nn as nn
import torch.optim as optim
from utils.amp import autocast_context, create_grad_scaler
from utils.device import device

class A2CNetwork(nn.Module):
    def __init__(self, height, width, action_size, num_items, gamma=0.99, lr=1e-4):
        super().__init__()
        self.height = height
        self.width  = width
        self.action_size= action_size
        self.num_items= num_items
        self.gamma= gamma
        self.lr = lr

        self.frame_net = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten()
        ).to(device)

        self.item_net = nn.Sequential(
            nn.Linear(num_items*2,128),
            nn.ReLU()
        ).to(device)

        self.combined_size = self._calc_combined_size()

        self.actor = nn.Sequential(
            nn.Linear(self.combined_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_size)
        ).to(device)

        self.critic = nn.Sequential(
            nn.Linear(self.combined_size,128),
            nn.ReLU(),
            nn.Linear(128,1)
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.frame_net.parameters())
            + list(self.item_net.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=self.lr
        )
        self.scaler = create_grad_scaler()

    def _calc_combined_size(self):
        with torch.no_grad():
            dummy_frame = torch.zeros((1,1,self.height,self.width), device=device)
            out_frame = self.frame_net(dummy_frame)
            frame_dim = out_frame.shape[1]

            dummy_items = torch.zeros((1,self.num_items*2), device=device)
            out_items = self.item_net(dummy_items)
            item_dim = out_items.shape[1]
        return frame_dim + item_dim

    def forward(self, frame_4d, item_2d):
        feat_f = self.frame_net(frame_4d)
        feat_i = self.item_net(item_2d)
        combined = torch.cat((feat_f, feat_i), dim=1)
        logits = self.actor(combined)
        value = self.critic(combined)
        return logits, value

    def select_action(self, state, valid_idx):
        frame, remain = state
        # => frame_cpu shape(H,W)
        frame_4d = frame.unsqueeze(0).unsqueeze(0).float().to(device)
        # => (1,1,H,W)

        remain_np = remain.detach().cpu().numpy().reshape(1, -1) # => (1, N*2)
        remain_t = torch.from_numpy(remain_np).float().to(device)

        with autocast_context():
            logits, val = self.forward(frame_4d, remain_t)
        logits = logits.float()
        val = val.float()
        logits = logits.squeeze(0)
        val = val.squeeze(0)

        probs = torch.softmax(logits, dim=-1)

        mask = torch.zeros_like(probs)
        if valid_idx:
            mask[valid_idx] = 1.0
        else:
            mask[:] = 1.0

        masked_probs = probs * mask
        if masked_probs.sum() < 1e-8:
            masked_probs = mask
        masked_probs = masked_probs / masked_probs.sum()

        dist = torch.distributions.Categorical(masked_probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob, val

    def update_ac(self, log_probs, values, rewards, next_val, done):
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        returns = []
        G = 0 if done else next_val.item()
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        advantage = returns - values.squeeze(-1)
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def train_one_episode(self, env, batch_size=32):
        max_items = getattr(env, 'max_items', env.num_items)
        bin_types = getattr(env, 'bin_types', [{'width': env.width, 'height': env.height, 'cost': 100.0}])
        action_space = [(i, rot) for i in range(max_items) for rot in [False, True]] + [("open", b_idx) for b_idx in range(len(bin_types))]
        env.reset()
        done = False
        frame, remain = env.get_state()
        total_reward = 0
        transitions = []
        sum_loss = 0

        while not done:
            valid_idx = env.get_valid_actions(action_space)
            if not valid_idx:
                done = True
                break

            a_idx, log_p, val = self.select_action((frame, remain), valid_idx)
            action = action_space[a_idx]
            success, reward = env.place(action)
            next_frame, next_remain = env.get_state()

            transitions.append((log_p, val, reward))
            frame, remain = next_frame, next_remain
            total_reward += reward
            done = env.is_done()

            if len(transitions) == batch_size or done:
                if done:
                    next_val = torch.tensor(0.0, device=device)
                else:
                    nf_4d = next_frame.unsqueeze(0).unsqueeze(0).float().to(device)
                    nr_np = next_remain.detach().cpu().numpy().reshape(1, -1)
                    nr_t = torch.from_numpy(nr_np).float().to(device)

                    with torch.no_grad(), autocast_context():
                        logits2, val2 = self.forward(nf_4d, nr_t)
                        next_val = val2.squeeze(0)

                lgs, vals, rws = zip(*transitions)
                sum_loss += self.update_ac(lgs, vals, rws, next_val, done)
                transitions = []

        if len(transitions) > 0:
            next_val = torch.tensor(0.0, device=device)
            lgs, vals, rws = zip(*transitions)
            sum_loss += self.update_ac(lgs, vals, rws, next_val, True)

        return total_reward, sum_loss

def train_a2c_episode(env, agent, batch_size=32):
    """Functional wrapper for compatibility and consistency."""
    return agent.train_one_episode(env, batch_size)

import random

class QLearning:
    def __init__(self, num_items, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.num_items = num_items
        self.gamma = gamma  # Discount rate
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}

    def get_state(self, packing_env):
        frame, remain = packing_env.get_state()
        # Ép Tensor sang string tuple để có thể hash làm key cho Dictionary
        # LƯU Ý: State không gian 100x100 rất lớn, RAM sẽ bị đầy rất nhanh khi dùng Q-Table.
        return (str(frame.flatten().tolist()), str(remain.flatten().tolist()))

    def choose_action(self, state, packing_env):
        max_items = getattr(packing_env, 'max_items', self.num_items)
        bin_types = getattr(packing_env, 'bin_types', [{'width': packing_env.width, 'height': packing_env.height, 'cost': 100.0}])
        action_space = [(i, rot) for i in range(max_items) for rot in [True, False]] + [("open", b_idx) for b_idx in range(len(bin_types))]
        valid_actions = packing_env.get_valid_actions(action_space)

        if not valid_actions:
            return None  

        if random.random() < self.epsilon:  # Exploration
            return random.choice([action_space[i] for i in valid_actions])

        # Exploitation
        q_values = [self.q_table.get((state, action_space[i]), 0) for i in valid_actions]
        max_q = max(q_values)
        best_actions = [action_space[i] for i, q_val in zip(valid_actions, q_values) if q_val == max_q]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, packing_env):
        current_Q = self.q_table.get((state, action), 0)

        max_items = getattr(packing_env, 'max_items', self.num_items)
        bin_types = getattr(packing_env, 'bin_types', [{'width': packing_env.width, 'height': packing_env.height, 'cost': 100.0}])
        action_space = [(i, rot) for i in range(max_items) for rot in [True, False]] + [("open", b_idx) for b_idx in range(len(bin_types))]
        next_valid_actions = packing_env.get_valid_actions(action_space)
        next_q_values = [self.q_table.get((next_state, action), 0) for action in next_valid_actions]
        max_next_q = max(next_q_values) if next_q_values else 0

        # Q-learning update
        self.q_table[(state, action)] = current_Q + self.alpha * (
            reward + self.gamma * max_next_q - current_Q
        )

def train_q_episode(env, agent):
    """
    Trains Q-learning agent for one episode.
    """
    env.reset()
    state = agent.get_state(env)
    total_rw = 0
    max_steps = max(1, env.num_items * 3)
    steps = 0

    while not env.is_done() and steps < max_steps:
        steps += 1
        action = agent.choose_action(state, env)
        if action is None:
            break

        success, reward = env.place(action)
        next_state = agent.get_state(env)

        agent.update_q_table(state, action, reward, next_state, env)
        state = next_state
        total_rw += reward

    return total_rw

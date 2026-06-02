import torch
import matplotlib.pyplot as plt
from utils.device import device

class Packing:
    def __init__(self, bin_types, items_or_height, items=None, max_width=100, max_height=100, max_items=200):
        # Packing(width, height, items)
        if isinstance(bin_types, int):
            width = bin_types
            height = items_or_height
            self.bin_types = [{'width': width, 'height': height, 'cost': 100.0}]
            self.items = items
        else:
            self.bin_types = bin_types
            self.items = items_or_height

        self.num_items = len(self.items)
        self.max_width = max_width
        self.max_height = max_height
        self.max_items = max_items

        # Start with default bin type 0
        self.active_bin_type = 0
        self.width = self.bin_types[0]['width']
        self.height = self.bin_types[0]['height']

        # Initialize the frame
        self.frame = torch.zeros((self.height, self.width), requires_grad=False)
        self.reset()

    def reset(self):
        # Modern VS-BPP reset
        self.placed_items = []
        self.opened_bins = [self.bin_types[0]]
        self.opened_bins_count = 1
        self.total_bin_cost = self.bin_types[0]['cost']
        
        # Reset current active frame to default bin type
        self.width = self.bin_types[0]['width']
        self.height = self.bin_types[0]['height']
        self.frame = torch.zeros((self.height, self.width), requires_grad=False)
        self.empty_positions = [(0, 0)]
        
        # Initialize remaining items with padding [0, 0]
        self.remain_items = [[0, 0] for _ in range(self.max_items)]
        for i, item in enumerate(self.items):
            if i < self.max_items:
                self.remain_items[i] = list(item)

    def open_new_bin(self, bin_type_idx):
        """Closes current bin and opens a new one of bin_type_idx."""
        bin_cfg = self.bin_types[bin_type_idx]
        self.width = bin_cfg['width']
        self.height = bin_cfg['height']
        
        # Create fresh empty frame and positions for the new bin
        self.frame = torch.zeros((self.height, self.width), requires_grad=False)
        self.empty_positions = [(0, 0)]
        
        self.opened_bins_count += 1
        self.opened_bins.append(bin_cfg)
        self.total_bin_cost += bin_cfg['cost']
        
        # Return negative cost penalty (scaled by 100)
        penalty = - (bin_cfg['cost'] / 100.0)
        return penalty

    def can_place(self, x, y, h, w):
        # check condition to place the item
        if x + h > self.height or y + w > self.width:
            return False
        return torch.all(self.frame[x:x + h, y:y + w] == 0) 

    def update_empty_positions(self, x, y, h, w):
        new_positions = []
        if y + w < self.width:
            new_positions.append((x, y + w))
        if x + h < self.height:
            new_positions.append((x + h, y))

        self.empty_positions = [
            pos for pos in self.empty_positions
            if not (x <= pos[0] < x + h and y <= pos[1] < y + w)
        ]
        self.empty_positions.extend(new_positions)

    def place(self, action):
        # Check if action is to open a new bin: action format ("open", bin_type_idx)
        if action[0] == "open":
            bin_type_idx = action[1]
            penalty = self.open_new_bin(bin_type_idx)
            return True, penalty

        # Else, standard placement action: action format (item_index, rotated)
        index, rotated = action
        if index >= self.max_items or self.remain_items[index] == [0, 0]:
            return False, -5 

        h, w = self.remain_items[index]
        if rotated:
            h, w = w, h

        max_possible_adjacency = 2 * (h + w)
        max_penalty = self.width * self.height

        best_position = None
        best_reward = -float('inf')

        for x, y in self.empty_positions:
            if self.can_place(x, y, h, w):
                # calculate reward elements
                adjacency_bonus = self.calc_adjacency(x, y, h, w) / max_possible_adjacency
                square_bonus = self.calc_square_bonus(h, w)
                space_penalty = self.calc_space_penalty(x, y, h, w) / max_penalty

                remaining_space = (self.width * self.height - torch.sum(self.frame).item())
                ratio_s = (h * w) / remaining_space if remaining_space > 0 else 0

                reward = (
                        ratio_s * 0.7 
                        + adjacency_bonus * 0.2
                        + square_bonus * 0.1
                        - space_penalty * 0.1
                )

                # penalty if not square or adjacency 
                if square_bonus < 0.5 and adjacency_bonus < 0.3:
                    reward -= 0.2
                
                # update best reward 
                if reward > best_reward:
                    best_reward = reward
                    best_position = (x, y)

        # place item
        if best_position:
            x, y = best_position
            with torch.no_grad():
                self.frame[x:x + h, y:y + w] = 1
            self.remain_items[index] = [0, 0]
            self.placed_items.append((index, x, y, h, w, rotated, self.opened_bins_count))
            self.update_empty_positions(x, y, h, w)
            return True, best_reward

        return False, -5

    # function to calculate the adjacency bonus
    def calc_adjacency(self, x, y, h, w):
        bonus = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            for i in range(h):
                for j in range(w):
                    nx, ny = x + i + dx, y + j + dy
                    if 0 <= nx < self.height and 0 <= ny < self.width:
                        if self.frame[nx, ny] == 1:
                            bonus += 1
        return bonus

    # function to calculate the square bonus
    def calc_square_bonus(self, h, w):
        return min(h, w) / max(h, w)

    # function to calculate the space penalty
    def calc_space_penalty(self, x, y, h, w):
        penalty = 0
        for dx in range(h):
            for dy in range(w):
                nx, ny = x + dx, y + dy
                if nx < self.height and ny < self.width and self.frame[nx, ny] == 0:
                    penalty += 1
        return penalty

    # All items are placed ?
    def is_done(self):
        return all(self.remain_items[i] == [0, 0] for i in range(self.num_items))

    def get_state(self):
        # Create padded grid state of shape (max_height, max_width) initialized to 1.0 (occupied)
        padded_frame = torch.ones((self.max_height, self.max_width), dtype=torch.float32)
        # Copy the active frame into the top-left region
        padded_frame[:self.height, :self.width] = self.frame
        
        remain_items_tensor = torch.tensor(self.remain_items, dtype=torch.float32)
        return padded_frame, remain_items_tensor 

    # Find valid actions
    def get_valid_actions(self, action_space, allow_rotation=True):
        placement_actions = []
        open_actions = []
        for idx, act in enumerate(action_space):
            # Check action type
            if act[0] == "open":
                # Opening a new bin is valid only when the active bin cannot
                # place any remaining item. This prevents agents from learning
                # to spam "open" actions instead of packing.
                if not self.is_done() and self.bin_type_can_fit_remaining(act[1], allow_rotation=allow_rotation):
                    open_actions.append(idx)
            else:
                # Placement action: act is (item_idx, rotated)
                item_idx = act[0]
                rotated = act[1]
                if rotated and not allow_rotation:
                    continue
                if item_idx < self.num_items and self.remain_items[item_idx] != [0, 0]:
                    h, w = self.remain_items[item_idx]
                    if rotated:
                        h, w = w, h
                    if any(self.can_place(x, y, h, w) for x, y in self.empty_positions):
                        placement_actions.append(idx)

        return placement_actions if placement_actions else open_actions

    def bin_type_can_fit_remaining(self, bin_type_idx, allow_rotation=True):
        if bin_type_idx >= len(self.bin_types):
            return False

        bin_cfg = self.bin_types[bin_type_idx]
        bin_h = bin_cfg['height']
        bin_w = bin_cfg['width']

        for item_idx in range(self.num_items):
            h, w = self.remain_items[item_idx]
            if [h, w] == [0, 0]:
                continue
            if h <= bin_h and w <= bin_w:
                return True
            if allow_rotation and w <= bin_h and h <= bin_w:
                return True
        return False

    def render(self):
        # Find all bins used
        bins_used = set(range(1, self.opened_bins_count + 1))
        if self.placed_items and len(self.placed_items[0]) > 6:
            bins_used.update(item[6] for item in self.placed_items)
        num_plots = len(bins_used)
        
        fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), squeeze=False)
        
        for b_idx, b_num in enumerate(sorted(bins_used)):
            ax = axes[0, b_idx]
            # Find items placed in this bin
            items_in_bin = [item for item in self.placed_items if len(item) > 6 and item[6] == b_num]
            if not items_in_bin and b_num == 1:
                items_in_bin = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in self.placed_items]
                
            for index, x, y, l, w, rotated in [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in items_in_bin]:
                color = 'blue' if rotated else 'pink'
                rect = plt.Rectangle((x, y), l, w, edgecolor='black', facecolor=color, alpha=0.5, linewidth=1)
                ax.add_patch(rect)
                ax.text(x + l / 2, y + w / 2, f"{l}×{w}", ha='center', va='center', fontsize=8)
                
            bin_cfg = self.opened_bins[b_num - 1] if b_num - 1 < len(self.opened_bins) else {'height': self.height, 'width': self.width}
            ax.set_xlim(0, bin_cfg['height'])
            ax.set_ylim(0, bin_cfg['width'])
            ax.set_xlabel('Length')
            ax.set_ylabel('Width')
            ax.set_title(f'Bin {b_num}')
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            
        plt.suptitle(f'VS-BPP Multi-Bin Packing ({num_plots} Bins Opened, Total Cost={self.total_bin_cost})')
        plt.tight_layout()
        plt.show()

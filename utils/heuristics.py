import time
from env.packing_env import Packing

def run_ffd_heuristic(bin_types, items, max_width=300, max_height=300, max_items=200):
    """
    Solves a 2DVSBPP instance using the First Fit Decreasing (FFD) heuristic.
    Maintains a replica environment to guarantee identical physical constraints and rules.
    
    Returns a dictionary of execution and optimization metrics.
    """
    start_time = time.time()
    
    # Initialize the replica environment with the selected bins and items
    env = Packing(bin_types=bin_types, items_or_height=items, max_width=max_width, max_height=max_height, max_items=max_items)
    
    # 1. Sort the items in decreasing order of area (height * width)
    # We keep their original index so we can play the corresponding actions in the environment
    sorted_items = sorted(enumerate(items), key=lambda x: x[1][0] * x[1][1], reverse=True)
    
    # Track the configuration of all opened bins to calculate utilization precisely
    opened_bins = [bin_types[0]]
    
    # 2. Iterate through sorted items and pack them sequentially
    for orig_idx, (w, h) in sorted_items:
        placed = False
        
        # Try to place the item in the currently active bin
        # We test rotation = False (no rotation) and then rotation = True (rotated)
        for rotated in [False, True]:
            item_w, item_h = (h, w) if rotated else (w, h)
            
            # Search empty positions bottom-left first (which is how FFD operates)
            for x, y in sorted(env.empty_positions, key=lambda p: (p[0], p[1])):
                if env.can_place(x, y, item_h, item_w):
                    env.place((orig_idx, rotated))
                    placed = True
                    break
            if placed:
                break
                
        # If the item cannot fit in the currently active bin, we must open a new one
        if not placed:
            # Find the cheapest bin type that is large enough to fit the piece (either rotated or not)
            cheapest_bin_idx = None
            cheapest_cost = float('inf')
            
            for b_idx, b_cfg in enumerate(bin_types):
                for rot in [False, True]:
                    iw, ih = (h, w) if rot else (w, h)
                    if iw <= b_cfg['width'] and ih <= b_cfg['height']:
                        if b_cfg['cost'] < cheapest_cost:
                            cheapest_cost = b_cfg['cost']
                            cheapest_bin_idx = b_idx
            
            if cheapest_bin_idx is not None:
                # Open the new bin in the environment
                env.place(("open", cheapest_bin_idx))
                opened_bins.append(bin_types[cheapest_bin_idx])
                
                # Place the item in the fresh grid at (0, 0)
                for rot in [False, True]:
                    iw, ih = (h, w) if rot else (w, h)
                    if env.can_place(0, 0, ih, iw):
                        env.place((orig_idx, rot))
                        break
            else:
                # Fallback: Piece is too large for any available bin type
                # Standard VSBPP behavior is to skip the item or fail placement
                pass
                
    exec_time = time.time() - start_time
    
    # Calculate precise Space Utilization Rate (%)
    total_piece_area = sum(item[3] * item[4] for item in env.placed_items)
    total_bin_area = sum(b['width'] * b['height'] for b in opened_bins)
    utilization = (total_piece_area / total_bin_area * 100) if total_bin_area > 0 else 0.0
    
    return {
        'cost': env.total_bin_cost,
        'bins_opened': env.opened_bins_count,
        'utilization': utilization,
        'placed_count': len(env.placed_items),
        'total_count': len(items),
        'success_rate': (len(env.placed_items) / len(items)) * 100 if len(items) > 0 else 0.0,
        'time': exec_time,
        'placed_items': env.placed_items,
        'opened_bins': opened_bins
    }

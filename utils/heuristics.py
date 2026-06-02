import time
from env.packing_env import Packing
from utils.metrics import summarize_packing

def run_ffd_heuristic(bin_types, items, max_width=300, max_height=300, max_items=200, allow_rotation=True):
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
    rotations = [False, True] if allow_rotation else [False]
    
    # 2. Iterate through sorted items and pack them sequentially
    for orig_idx, (h, w) in sorted_items:
        placed = False
        
        # Try to place the item in the currently active bin
        # We test rotation = False (no rotation) and then rotation = True (rotated)
        for rotated in rotations:
            item_h, item_w = (w, h) if rotated else (h, w)
            
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
                for rot in rotations:
                    ih, iw = (w, h) if rot else (h, w)
                    if iw <= b_cfg['width'] and ih <= b_cfg['height']:
                        if b_cfg['cost'] < cheapest_cost:
                            cheapest_cost = b_cfg['cost']
                            cheapest_bin_idx = b_idx
            
            if cheapest_bin_idx is not None:
                # Open the new bin in the environment
                env.place(("open", cheapest_bin_idx))
                
                # Place the item in the fresh grid at (0, 0)
                for rot in rotations:
                    ih, iw = (w, h) if rot else (h, w)
                    if env.can_place(0, 0, ih, iw):
                        env.place((orig_idx, rot))
                        break
            else:
                # Fallback: Piece is too large for any available bin type
                # Standard VSBPP behavior is to skip the item or fail placement
                pass
                
    exec_time = time.time() - start_time
    
    return summarize_packing(env, total_items=len(items), exec_time=exec_time)

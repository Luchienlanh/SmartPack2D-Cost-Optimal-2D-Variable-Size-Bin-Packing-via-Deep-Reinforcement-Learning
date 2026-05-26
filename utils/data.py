import os
import pandas as pd

def load_items_from_csv(filename):
    """Loads items from a CSV file searching in direct path or parent directories."""
    for path in [filename, os.path.join("..", filename), os.path.join("..", "..", filename)]:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            data = pd.read_csv(path)
            h = next((c for c in data.columns if c.lower() == 'height'), 'Height')
            w = next((c for c in data.columns if c.lower() == 'width'), 'Width')
            q = next((c for c in data.columns if c.lower() == 'quantity'), 'Quantity')
            return [(r[h], r[w]) for _, r in data.iterrows() for _ in range(int(r[q]))]
    raise FileNotFoundError(f"CSV file '{filename}' not found.")

def load_2dvsbpp_instance(file_path):
    """Parses a 2DVSBPP instance text file, skipping the Z dimension."""
    with open(file_path, 'r') as f:
        lines = [line.strip().split() for line in f if line.strip()]
        
    num_pieces = int(lines[0][0])
    num_bins = int(lines[0][1])
    
    # Parse Bins & Pieces (skipping the Z column)
    bins = [{'width': int(lines[i][0]), 'height': int(lines[i][1]), 'cost': float(lines[i][3])} 
            for i in range(1, num_bins + 1)]
    items = [(int(lines[i][1]), int(lines[i][2])) 
             for i in range(num_bins + 1, num_bins + 1 + num_pieces)]
    return bins, items

def scan_dataset_limits(dataset_dir):
    """Scans dataset files to calculate absolute scaling and shape limits."""
    import glob
    dataset_files = glob.glob(os.path.join(dataset_dir, '*.txt'))
    max_w, max_h, max_items, max_bin_types = 100, 100, 200, 5
    valid_files = []
    
    for fp in dataset_files:
        if os.path.basename(fp).lower() == 'readme.txt' or os.path.getsize(fp) == 0:
            continue
        try:
            bins, items = load_2dvsbpp_instance(fp)
            if not bins or not items:
                continue
            max_w = max(max_w, max(b['width'] for b in bins))
            max_h = max(max_h, max(b['height'] for b in bins))
            max_items = max(max_items, len(items))
            max_bin_types = max(max_bin_types, len(bins))
            valid_files.append(fp)
        except Exception:
            continue
    return max_w, max_h, max_items, max_bin_types, valid_files

def split_dataset_files(dataset_files, train_ratio=0.8, seed=42):
    """Splits dataset files deterministically into train and validation sets."""
    import random
    temp_list = sorted(list(dataset_files))
    random.Random(seed).shuffle(temp_list)
    split_idx = int(len(temp_list) * train_ratio)
    return temp_list[:split_idx], temp_list[split_idx:]

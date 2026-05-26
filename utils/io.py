import os
import torch

def get_project_root():
    # File is at Notebook_Scripts/utils/io.py
    # utils/ is parent, Notebook_Scripts/ is parent of utils, project root is parent of Notebook_Scripts
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def resolve_model_path(filename):
    """
    Resolves the model path. If filename is absolute, returns it.
    Otherwise, places it in the project root directory.
    """
    if os.path.isabs(filename):
        return filename
    return os.path.join(get_project_root(), filename)

def save_torch_checkpoint(state_dict, filename):
    path = resolve_model_path(filename)
    torch.save(state_dict, path)
    print(f"Saved checkpoint to: {path}")

def load_torch_checkpoint(filename):
    path = resolve_model_path(filename)
    if not os.path.exists(path):
        # Fallback: check if it's in the current working directory or relative to it
        if os.path.exists(filename):
            path = filename
        else:
            # Fallback to Notebook_Scripts directory
            notebook_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fallback_path = os.path.join(notebook_dir, filename)
            if os.path.exists(fallback_path):
                path = fallback_path
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {filename}")
    
    print(f"Loading checkpoint from: {path}")
    return torch.load(path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None, weights_only=False)

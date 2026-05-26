import os
import torch

# Suppress visual studio compile warnings or Intel OpenMP runtime library warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

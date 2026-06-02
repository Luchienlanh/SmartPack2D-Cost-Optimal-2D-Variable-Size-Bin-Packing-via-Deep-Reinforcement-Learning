from contextlib import nullcontext

import torch

from utils.device import device


AMP_ENABLED = device.type == "cuda"
AMP_DTYPE = torch.float16


def autocast_context():
    if not AMP_ENABLED:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=AMP_DTYPE)
    return torch.cuda.amp.autocast(dtype=AMP_DTYPE)


def create_grad_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=AMP_ENABLED)
        except TypeError:
            return torch.amp.GradScaler(enabled=AMP_ENABLED)
    return torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)


def load_scaler_state(scaler, checkpoint):
    state = checkpoint.get("scaler_state_dict")
    if state:
        scaler.load_state_dict(state)

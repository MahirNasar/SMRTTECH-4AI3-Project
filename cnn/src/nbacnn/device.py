# file: src/nbacnn/device.py
import random
import numpy as np, torch

def pick_device(pref="auto"):
    """'auto' -> mps if available, else cpu. 'cpu' / 'mps' force a choice."""
    if pref == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "mps":
        raise RuntimeError("MPS requested but not available (check arm64 Python + PyTorch build)")
    return torch.device("cpu")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)   # also seeds MPS

def sync(device):
    """Wait for queued GPU work to finish (needed for honest timing on MPS)."""
    if device.type == "mps":
        torch.mps.synchronize()
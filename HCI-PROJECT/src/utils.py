import os
import random
from pathlib import Path

import numpy as np
import yaml


def on_kaggle() -> bool:
    return os.path.exists("/kaggle/working")


def get_root() -> Path:
    """Project root: /kaggle/working/HCI-PROJECT on Kaggle, else current repo."""
    if on_kaggle():
        return Path("/kaggle/working/HCI-PROJECT")
    return Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/config.yaml") -> dict:
    root = get_root()
    with open(root / path, "r") as f:
        cfg = yaml.safe_load(f)
    base = Path("/kaggle/working") if on_kaggle() else root
    cfg["paths"]["data_root"] = str(base / cfg["paths"]["data_root"])
    cfg["paths"]["cache_dir"] = str(base / cfg["paths"]["cache_dir"])
    cfg["paths"]["results_dir"] = str(base / cfg["paths"]["results_dir"])
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
import os
import random
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # allow running some utilities without torch installed
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

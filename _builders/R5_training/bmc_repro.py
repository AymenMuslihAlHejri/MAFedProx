#!/usr/bin/env python3
"""bmc_repro.py - seeding, environment capture, frozen configs (matrix v2, K, O)."""
from __future__ import annotations
import hashlib, json, platform, random, sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

TRAINING_SEEDS = (101, 202, 303)
SPLIT_SEED, BOOTSTRAP_SEED = 20260929, 20260930


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generator(seed: int, stream: int) -> torch.Generator:
    """Independent, reproducible stream per (training seed, client/stream index)."""
    return torch.Generator().manual_seed(seed * 1000 + stream)


def worker_init_fn(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def env_versions() -> Dict[str, Any]:
    def v(mod):
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "unknown")
        except Exception:
            return "not installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": v("numpy"), "scipy": v("scipy"), "PIL": v("PIL"), "cv2": v("cv2"),
            "albumentations": v("albumentations"), "segmentation_models_pytorch": v("segmentation_models_pytorch")}


def config_sha256(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def write_once_json(path: Path, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)

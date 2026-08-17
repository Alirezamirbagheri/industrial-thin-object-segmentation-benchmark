from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def model_options(config: dict, model_name: str) -> dict[str, Any]:
    return dict(
        config.get("training", {})
        .get("model_overrides", {})
        .get(model_name, {})
    )


def run_name(
    model_name: str,
    fraction: float,
    seed: int,
    loss_profile: str,
) -> str:
    return (
        f"{model_name}__frac_{fraction:.2f}"
        f"__seed_{seed}__loss_{loss_profile}"
    )


def stable_ids_hash(ids: list[str]) -> str:
    joined = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

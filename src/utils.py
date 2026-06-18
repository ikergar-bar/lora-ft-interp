"""Shared helpers reused across milestones (seed, device, dtype, config, logging).

See lora-sentiment-interpretability-spec.md §2 (hardware), §4 (reproducibility), §9.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Project paths (this file lives in <repo>/src/).
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train.yaml"


def set_seed(seed: int) -> None:
    """Fix global seeds for reproducibility (spec §4, §8 determinism)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if available, else CPU (CPU fallback allowed by spec §2)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(precision: str | None, device: torch.device) -> torch.dtype:
    """Resolve the compute dtype, honouring the spec's bf16/fp16 rule (§2).

    bf16 only makes sense on Ampere+ (capability >= 8.0) GPUs. On older cards
    (Turing/Pascal) or CPU we fall back to fp16/fp32. If ``precision`` requests
    bf16 on unsupported hardware we downgrade to fp16 rather than fail.
    """
    if device.type != "cuda":
        # fp16 on CPU is slow/unsupported for many ops; use fp32 on CPU.
        return torch.float32

    requested = (precision or "fp16").lower()
    capability = torch.cuda.get_device_capability(0)
    supports_bf16 = capability[0] >= 8

    if requested == "bf16":
        return torch.bfloat16 if supports_bf16 else torch.float16
    if requested in ("fp16", "float16"):
        return torch.float16
    if requested in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML config (configs/train.yaml)."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def log_result(name: str, payload: dict[str, Any]) -> Path:
    """Persist a result dict as JSON under results/ and echo it to stdout."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"_logged_at": datetime.now(timezone.utc).isoformat(), **payload}
    out_path = RESULTS_DIR / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[log_result] wrote {out_path}")
    print(json.dumps(payload, indent=2))
    return out_path

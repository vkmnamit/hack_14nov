#!/usr/bin/env python3
"""Fine-tune a stateful, autoregressive ConvLSTM against ANUGA sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_convlstm import (
    BACKEND_DIR,
    DATA_PATH,
    MODEL_DIR,
    NORMALIZATION_PATH,
    ConvLSTMSurrogate,
    calculate_metrics,
    run_rollout,
    verify_normalization,
)


REPO_DIR = BACKEND_DIR.parent
INITIAL_WEIGHTS = MODEL_DIR / "convlstm_surrogate.pt"
OUTPUT_WEIGHTS = MODEL_DIR / "convlstm_multistep.pt"
HISTORY_PATH = MODEL_DIR / "convlstm_multistep_training.json"
TIMESTEPS = 60
CELL_AREA_M2 = 30.0**2
WET_THRESHOLD_M = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION_PATH)
    parser.add_argument("--initial-weights", type=Path, default=INITIAL_WEIGHTS)
    parser.add_argument("--output-weights", type=Path, default=OUTPUT_WEIGHTS)
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mass-loss-weight", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )


def normalized_static_inputs(
    data: Any,
    normalization: Any,
    scenario_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    infiltration = torch.as_tensor(
        data["infiltration_loss_mm_hr"][scenario_index],
        dtype=torch.float32,
        device=device,
    )
    elevation = torch.as_tensor(
        data["elevation_m"],
        dtype=torch.float32,
        device=device,
    )
    manning = torch.as_tensor(
        data["manning_n"][scenario_index],
        dtype=torch.float32,
        device=device,
    )
    infiltration_norm = (
        infiltration - float(normalization["infil_mean"])
    ) / (float(normalization["infil_std"]) + 1e-8)
    elevation_norm = (
        elevation - float(normalization["elev_mean"])
    ) / (float(normalization["elev_std"]) + 1e-8)
    manning_norm = (
        manning - float(normalization["mann_mean"])
    ) / (float(normalization["mann_std"]) + 1e-8)
    return infiltration_norm, elevation_norm, manning_norm



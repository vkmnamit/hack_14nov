#!/usr/bin/env python3
"""Compare ConvLSTM rollouts with ANUGA validation scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
DATA_PATH = (
    BACKEND_DIR
    / "data"
    / "processed"
    / "majuli_training_v2"
    / "majuli_anuga_convlstm_v2.npz"
)
MODEL_DIR = REPO_DIR / "model"
WEIGHTS_PATH = MODEL_DIR / "convlstm_surrogate.pt"
NORMALIZATION_PATH = MODEL_DIR / "normalization_config.npz"
OUTPUT_DIR = DATA_PATH.parent / "convlstm_evaluation"
TARGET_MINUTES = (1, 5, 15, 30, 60)
WET_THRESHOLD_M = 0.01
CELL_AREA_M2 = 30.0**2


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 32,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            batch, _, height, width = x.shape
            hidden_channels = self.gates.out_channels // 4
            h = x.new_zeros((batch, hidden_channels, height, width))
            c = torch.zeros_like(h)
        else:
            h, c = state

        gates = self.gates(torch.cat((x, h), dim=1))
        ingate, forgetgate, cellgate, outgate = torch.chunk(gates, 4, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        c_next = forgetgate * c + ingate * cellgate
        h_next = outgate * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class ConvLSTMSurrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(in_channels=7, hidden_channels=32)
        self.output_layer = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_state(x_seq)
        return output

    def forward_with_state(
        self,
        x_seq: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        for frame in range(x_seq.shape[1]):
            h_state, state = self.cell(x_seq[:, frame], state)
        return self.output_layer(h_state), state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Scenario split to evaluate.",
    )
    parser.add_argument(
        "--carry-hidden-state",
        action="store_true",
        help="Carry ConvLSTM state between autoregressive rollout steps.",
    )
    return parser.parse_args()


def verify_normalization(data: Any, normalization: Any) -> None:
    pairs = {
        "state_mean": "normalization_state_mean",
        "state_std": "normalization_state_std",
        "rain_mean": "normalization_raw_rainfall_mean_mm_hr",
        "rain_std": "normalization_raw_rainfall_std_mm_hr",
        "infil_mean": "normalization_infiltration_mean_mm_hr",
        "infil_std": "normalization_infiltration_std_mm_hr",
        "elev_mean": "normalization_elevation_mean_m",
        "elev_std": "normalization_elevation_std_m",
        "mann_mean": "normalization_manning_mean",
        "mann_std": "normalization_manning_std",
    }
    for model_key, data_key in pairs.items():
        if not np.allclose(
            normalization[model_key],
            data[data_key],
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(f"Normalization mismatch for {model_key}")


def make_model(weights_path: Path) -> ConvLSTMSurrogate:
    model = ConvLSTMSurrogate()
    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model



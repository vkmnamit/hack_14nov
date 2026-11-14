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



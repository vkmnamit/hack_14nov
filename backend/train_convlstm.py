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


def train_epoch(
    model: ConvLSTMSurrogate,
    optimizer: torch.optim.Optimizer,
    data: Any,
    normalization: Any,
    scenario_indices: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    teacher_forcing_ratio: float,
    chunk_length: int,
    mass_loss_weight: float,
) -> float:
    model.train()
    state_mean = torch.as_tensor(
        normalization["state_mean"],
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    state_std = torch.as_tensor(
        normalization["state_std"],
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    chunk_loss_total = 0.0
    step_count = 0

    for scenario_index in rng.permutation(scenario_indices):
        scenario_index = int(scenario_index)
        truth = torch.as_tensor(
            data["state_sequence"][scenario_index],
            dtype=torch.float32,
            device=device,
        )
        truth_norm = (truth - state_mean) / (state_std + 1e-8)
        rainfall = torch.as_tensor(
            data["raw_rainfall_mm_hr"][scenario_index, :TIMESTEPS],
            dtype=torch.float32,
            device=device,
        )
        effective_rainfall = torch.as_tensor(
            data["rainfall_effective_m_s"][scenario_index, :TIMESTEPS],
            dtype=torch.float32,
            device=device,
        )
        rainfall_norm = (
            rainfall - float(normalization["rain_mean"])
        ) / (float(normalization["rain_std"]) + 1e-8)
        static = torch.stack(
            normalized_static_inputs(
                data,
                normalization,
                scenario_index,
                device,
            )
        )

        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
        feedback = truth_norm[0:1]
        for chunk_start in range(0, TIMESTEPS, chunk_length):
            chunk_end = min(chunk_start + chunk_length, TIMESTEPS)
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []

            for frame in range(chunk_start, chunk_end):
                if frame == 0 or rng.random() < teacher_forcing_ratio:
                    current_norm = truth_norm[frame : frame + 1]
                else:
                    current_norm = feedback

                model_input = torch.cat(
                    (
                        current_norm,
                        rainfall_norm[frame].unsqueeze(0).unsqueeze(0),
                        static.unsqueeze(0),
                    ),
                    dim=1,
                ).unsqueeze(1)
                prediction_norm, hidden_state = model.forward_with_state(
                    model_input,
                    hidden_state,
                )
                target_norm = truth_norm[frame + 1 : frame + 2]

                depth_loss = F.mse_loss(
                    prediction_norm[:, 0],
                    target_norm[:, 0],
                )
                target_wet = (
                    truth[frame + 1, 0] >= WET_THRESHOLD_M
                ).unsqueeze(0).unsqueeze(0)
                momentum_error = (
                    prediction_norm[:, 1:] - target_norm[:, 1:]
                ).square()
                momentum_loss = (
                    (momentum_error * target_wet).sum()
                    / (2 * target_wet.sum().clamp_min(1))
                )

                predicted_depth = (
                    prediction_norm[:, 0] * state_std[:, 0] + state_mean[:, 0]
                ).clamp_min(0.0)
                current_depth = (
                    current_norm[:, 0] * state_std[:, 0] + state_mean[:, 0]
                ).clamp_min(0.0)
                predicted_volume_change = (
                    (predicted_depth.sum() - current_depth.sum())
                    * CELL_AREA_M2
                )
                expected_volume_change = (
                    effective_rainfall[frame].sum()
                    * 60.0
                    * CELL_AREA_M2
                )
                mass_scale = expected_volume_change.abs().clamp_min(1.0)
                mass_loss = (
                    (predicted_volume_change - expected_volume_change)
                    / mass_scale
                ).square()
                losses.append(
                    depth_loss + momentum_loss + mass_loss_weight * mass_loss
                )

                minimum_normalized_depth = (
                    -state_mean[0, 0, 0, 0] / state_std[0, 0, 0, 0]
                )
                feedback = torch.cat(
                    (
                        prediction_norm[:, 0:1].clamp_min(
                            minimum_normalized_depth
                        ),
                        prediction_norm[:, 1:],
                    ),
                    dim=1,
                )

            chunk_loss = torch.stack(losses).mean()
            if not torch.isfinite(chunk_loss):
                raise RuntimeError(
                    f"Non-finite training loss in scenario {scenario_index}, "
                    f"frames {chunk_start}-{chunk_end}"
                )
            chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            chunk_steps = chunk_end - chunk_start
            chunk_loss_total += float(chunk_loss.detach().cpu()) * chunk_steps
            step_count += chunk_steps
            hidden_state = tuple(value.detach() for value in hidden_state)
            feedback = feedback.detach()

    return chunk_loss_total / step_count



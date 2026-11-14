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

try:
    from .evaluate_convlstm import (
        BACKEND_DIR,
        DATA_PATH,
        MODEL_DIR,
        NORMALIZATION_PATH,
        ConvLSTMSurrogate,
        calculate_metrics,
        predict_next_state,
        run_rollout,
        verify_normalization,
    )
except ImportError:
    from evaluate_convlstm import (
        BACKEND_DIR,
        DATA_PATH,
        MODEL_DIR,
        NORMALIZATION_PATH,
        ConvLSTMSurrogate,
        calculate_metrics,
        predict_next_state,
        run_rollout,
        verify_normalization,
    )


REPO_DIR = BACKEND_DIR.parent
INITIAL_WEIGHTS = MODEL_DIR / "convlstm_surrogate.pt"
OUTPUT_WEIGHTS = MODEL_DIR / "convlstm_hybrid.pt"
HISTORY_PATH = MODEL_DIR / "convlstm_hybrid_training.json"
TIMESTEPS = 60
CELL_AREA_M2 = 30.0**2
WET_THRESHOLD_M = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION_PATH)
    parser.add_argument("--initial-weights", type=Path, default=INITIAL_WEIGHTS)
    parser.add_argument(
        "--initialization-mode",
        choices=("absolute", "hybrid_residual"),
        default="absolute",
        help="Whether initial weights predict states or residual corrections.",
    )
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
        elevation_m = torch.as_tensor(
            data["elevation_m"],
            dtype=torch.float32,
            device=device,
        )
        manning = torch.as_tensor(
            data["manning_n"][scenario_index],
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
                prediction_norm, hidden_state = predict_next_state(
                    model,
                    model_input,
                    current_norm,
                    effective_rainfall[frame].unsqueeze(0).unsqueeze(0),
                    elevation_m,
                    manning,
                    state_mean,
                    state_std,
                    hidden_state,
                    physics_informed=True,
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


def validate(
    model: ConvLSTMSurrogate,
    data: Any,
    normalization: Any,
    scenario_indices: np.ndarray,
    carry_hidden_state: bool = True,
    physics_informed: bool = False,
) -> dict[str, float]:
    model.eval()
    metrics_by_minute: dict[int, list[dict[str, float | None]]] = {
        15: [],
        30: [],
        60: [],
    }
    for scenario_index in scenario_indices:
        prediction, _, _, _ = run_rollout(
            model,
            data,
            normalization,
            int(scenario_index),
            carry_hidden_state=carry_hidden_state,
            physics_informed=physics_informed,
        )
        truth = data["state_sequence"][int(scenario_index)]
        for minute in metrics_by_minute:
            metrics_by_minute[minute].append(
                calculate_metrics(prediction[minute], truth[minute])
            )

    result: dict[str, float] = {}
    for minute, scenario_metrics in metrics_by_minute.items():
        for metric_name in (
            "depth_mae",
            "wet_area_csi",
            "wet_depth_mean_relative_error_pct",
            "water_volume_relative_error_pct",
        ):
            values = [
                metrics[metric_name]
                for metrics in scenario_metrics
                if metrics[metric_name] is not None
            ]
            result[f"{metric_name}_at_{minute}min"] = (
                float(np.mean(values)) if values else float("nan")
            )
    return result


def validation_score(
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> float:
    depth_ratio = (
        metrics["depth_mae_at_60min"]
        / max(baseline["depth_mae_at_60min"], 1e-6)
    )
    volume_ratio = (
        metrics["water_volume_relative_error_pct_at_60min"]
        / max(baseline["water_volume_relative_error_pct_at_60min"], 1.0)
    )
    csi_miss_ratio = (
        (1.0 - metrics["wet_area_csi_at_60min"])
        / max(1.0 - baseline["wet_area_csi_at_60min"], 0.05)
    )
    return (depth_ratio + volume_ratio + csi_miss_ratio) / 3.0


def save_checkpoint(model: ConvLSTMSurrogate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.chunk_length < 1:
        raise ValueError("epochs, patience, and chunk length must be positive")
    if args.mass_loss_weight < 0:
        raise ValueError("mass-loss weight cannot be negative")

    device = select_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ConvLSTMSurrogate()
    initial_state = torch.load(
        args.initial_weights,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(initial_state, strict=True)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as data, np.load(
        args.normalization,
        allow_pickle=False,
    ) as normalization:
        verify_normalization(data, normalization)
        train_indices = np.flatnonzero(data["split"] == "train")
        validation_indices = np.flatnonzero(data["split"] == "validation")
        if train_indices.size == 0 or validation_indices.size == 0:
            raise ValueError("Dataset must contain train and validation scenarios")

        initial_checkpoint_validation = validate(
            model,
            data,
            normalization,
            validation_indices,
            carry_hidden_state=True,
            physics_informed=args.initialization_mode == "hybrid_residual",
        )
        initial_residual_weight = model.output_layer.weight.detach().clone()
        initial_residual_bias = model.output_layer.bias.detach().clone()
        with torch.no_grad():
            torch.nn.init.zeros_(model.output_layer.weight)
            torch.nn.init.zeros_(model.output_layer.bias)
        physics_baseline_validation = validate(
            model,
            data,
            normalization,
            validation_indices,
            carry_hidden_state=True,
            physics_informed=True,
        )
        best_validation_score = 1.0
        best_epoch = 0
        if args.initialization_mode == "hybrid_residual":
            initial_score = validation_score(
                initial_checkpoint_validation,
                physics_baseline_validation,
            )
            if initial_score < best_validation_score:
                with torch.no_grad():
                    model.output_layer.weight.copy_(initial_residual_weight)
                    model.output_layer.bias.copy_(initial_residual_bias)
                best_validation_score = initial_score
        save_checkpoint(model, args.output_weights)

        history: dict[str, Any] = {
            "device": str(device),
            "training_scenarios": int(train_indices.size),
            "validation_scenarios": int(validation_indices.size),
            "validation_selection_metric": (
                "mean of depth-MAE ratio, water-volume-error ratio, and "
                "CSI-miss ratio relative to the physics-only validation "
                "baseline; volume baseline has a 1 percentage-point floor"
            ),
            "initial_checkpoint_validation": initial_checkpoint_validation,
            "physics_baseline_validation": physics_baseline_validation,
            "chunk_length": args.chunk_length,
            "mass_loss_weight": args.mass_loss_weight,
            "initialization_mode": args.initialization_mode,
            "physics_solver": {
                "method": "hydrostatic-reconstruction Rusanov flux",
                "gravity_m_s2": 9.81,
                "cell_size_m": 30.0,
                "cfl": 0.45,
                "rainfall_units": "effective metres per second",
                "friction": "Manning",
                "boundaries": "reflective",
                "correction": "ConvLSTM residual in normalized state units",
            },
            "initial_weights": str(args.initial_weights),
            "best_weights": str(args.output_weights),
            "best_epoch": best_epoch,
            "best_validation_score": best_validation_score,
            "epochs": [],
        }
        epochs_without_improvement = 0

        print(
            f"Training on {device}; {train_indices.size} train and "
            f"{validation_indices.size} ANUGA validation scenarios"
        )
        for epoch in range(args.epochs):
            started = time.perf_counter()
            progress = epoch / max(args.epochs - 1, 1)
            teacher_forcing_ratio = 0.5 * (1.0 - progress)
            train_loss = train_epoch(
                model,
                optimizer,
                data,
                normalization,
                train_indices,
                device,
                rng,
                teacher_forcing_ratio,
                args.chunk_length,
                args.mass_loss_weight,
            )
            validation_metrics = validate(
                model,
                data,
                normalization,
                validation_indices,
                physics_informed=True,
            )
            score = validation_score(
                validation_metrics,
                physics_baseline_validation,
            )
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "teacher_forcing_ratio": teacher_forcing_ratio,
                "validation": validation_metrics,
                "validation_score": score,
                "seconds": time.perf_counter() - started,
            }
            history["epochs"].append(epoch_record)
            improved = score < best_validation_score
            if improved:
                best_validation_score = score
                epochs_without_improvement = 0
                history["best_epoch"] = epoch + 1
                save_checkpoint(model, args.output_weights)
            else:
                epochs_without_improvement += 1

            args.history.write_text(
                json.dumps(history, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train loss {train_loss:.6f} | "
                f"val 60-min depth MAE "
                f"{validation_metrics['depth_mae_at_60min']:.5f} m | "
                f"CSI {validation_metrics['wet_area_csi_at_60min']:.3f} | "
                f"volume error "
                f"{validation_metrics['water_volume_relative_error_pct_at_60min']:.1f}% | "
                f"score {score:.3f} | "
                f"teacher forcing {teacher_forcing_ratio:.2f} | "
                f"{epoch_record['seconds']:.1f}s"
            )
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation score did not improve.")
                break

        history["best_validation_score"] = best_validation_score
        args.history.write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Best checkpoint: {args.output_weights}")
        print(f"Training history: {args.history}")


if __name__ == "__main__":
    main()

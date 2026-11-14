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


def run_rollout(
    model: ConvLSTMSurrogate,
    data: Any,
    normalization: Any,
    scenario_index: int,
    carry_hidden_state: bool = False,
) -> tuple[np.ndarray, float]:
    truth = data["state_sequence"][scenario_index]
    rainfall = data["raw_rainfall_mm_hr"][scenario_index]
    infiltration = data["infiltration_loss_mm_hr"][scenario_index]
    elevation = data["elevation_m"]
    manning = data["manning_n"][scenario_index]

    infiltration_norm = (
        infiltration - normalization["infil_mean"]
    ) / (normalization["infil_std"] + 1e-8)
    elevation_norm = (
        elevation - normalization["elev_mean"]
    ) / (normalization["elev_std"] + 1e-8)
    manning_norm = (
        manning - normalization["mann_mean"]
    ) / (normalization["mann_std"] + 1e-8)

    current = truth[0].copy()
    predicted = [current.copy()]
    device = next(model.parameters()).device
    hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
    start = time.perf_counter()
    with torch.inference_mode():
        for frame in range(60):
            state_norm = (
                current - normalization["state_mean"][:, None, None]
            ) / (normalization["state_std"][:, None, None] + 1e-8)
            rainfall_norm = (
                rainfall[frame] - normalization["rain_mean"]
            ) / (normalization["rain_std"] + 1e-8)
            model_input = np.empty((1, 1, 7, 50, 50), dtype=np.float32)
            model_input[0, 0, 0:3] = state_norm
            model_input[0, 0, 3] = rainfall_norm
            model_input[0, 0, 4] = infiltration_norm
            model_input[0, 0, 5] = elevation_norm
            model_input[0, 0, 6] = manning_norm

            model_input_tensor = torch.from_numpy(model_input).to(device)
            if carry_hidden_state:
                output, hidden_state = model.forward_with_state(
                    model_input_tensor,
                    hidden_state,
                )
            else:
                output = model(model_input_tensor)
            output_norm = output[0].cpu().numpy()
            current = (
                output_norm * (normalization["state_std"][:, None, None] + 1e-8)
                + normalization["state_mean"][:, None, None]
            )
            current[0] = np.maximum(current[0], 0.0)
            predicted.append(current.copy())

    return np.stack(predicted), time.perf_counter() - start


def mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def calculate_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    channel_names = ("depth", "x_momentum", "y_momentum")
    for channel, name in enumerate(channel_names):
        error = predicted[channel] - truth[channel]
        result[f"{name}_mae"] = float(np.mean(np.abs(error)))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
        result[f"{name}_bias"] = float(np.mean(error))

    predicted_depth = predicted[0]
    truth_depth = truth[0]
    depth_error = predicted_depth - truth_depth
    truth_wet = truth_depth >= WET_THRESHOLD_M
    predicted_wet = predicted_depth >= WET_THRESHOLD_M
    shared_wet = truth_wet & predicted_wet
    wet_intersection = np.logical_and(truth_wet, predicted_wet).sum()
    wet_union = np.logical_or(truth_wet, predicted_wet).sum()

    result["wet_depth_mae_m"] = mean_or_none(np.abs(depth_error[truth_wet]))
    result["wet_depth_mean_relative_error_pct"] = (
        float(np.mean(np.abs(depth_error[truth_wet]) / truth_depth[truth_wet]) * 100)
        if truth_wet.any()
        else None
    )
    result["wet_area_csi"] = (
        float(wet_intersection / wet_union) if wet_union else 1.0
    )
    result["wet_area_precision"] = (
        float(wet_intersection / predicted_wet.sum())
        if predicted_wet.any()
        else None
    )
    result["wet_area_recall"] = (
        float(wet_intersection / truth_wet.sum()) if truth_wet.any() else None
    )

    truth_volume = float(truth_depth.sum() * CELL_AREA_M2)
    predicted_volume = float(predicted_depth.sum() * CELL_AREA_M2)
    result["anuga_water_volume_m3"] = truth_volume
    result["convlstm_water_volume_m3"] = predicted_volume
    result["water_volume_relative_error_pct"] = (
        float(abs(predicted_volume - truth_volume) / truth_volume * 100)
        if truth_volume > 0
        else None
    )

    if shared_wet.any():
        truth_u = truth[1, shared_wet] / truth_depth[shared_wet]
        truth_v = truth[2, shared_wet] / truth_depth[shared_wet]
        predicted_u = predicted[1, shared_wet] / predicted_depth[shared_wet]
        predicted_v = predicted[2, shared_wet] / predicted_depth[shared_wet]
        truth_speed = np.hypot(truth_u, truth_v)
        predicted_speed = np.hypot(predicted_u, predicted_v)
        result["velocity_u_mae_m_s"] = float(np.mean(np.abs(predicted_u - truth_u)))
        result["velocity_v_mae_m_s"] = float(np.mean(np.abs(predicted_v - truth_v)))
        result["velocity_vector_mae_m_s"] = float(
            np.mean(np.hypot(predicted_u - truth_u, predicted_v - truth_v))
        )
        result["speed_mae_m_s"] = float(
            np.mean(np.abs(predicted_speed - truth_speed))
        )
    else:
        result["velocity_u_mae_m_s"] = None
        result["velocity_v_mae_m_s"] = None
        result["velocity_vector_mae_m_s"] = None
        result["speed_mae_m_s"] = None

    return result


def scenario_parameters(data: Any, index: int) -> dict[str, Any]:
    rainfall_profile = data["rainfall_profiles_mm_hr"][index]
    infiltration = data["infiltration_loss_mm_hr"][index]
    manning = data["manning_n"][index]
    return {
        "scenario_index": int(index),
        "scenario_id": int(data["scenario_ids"][index]),
        "scenario_name": str(data["scenario_names"][index]),
        "temporal_profile": str(data["temporal_profiles"][index]),
        "spatial_pattern": str(data["spatial_patterns"][index]),
        "initial_condition": str(data["initial_condition_names"][index]),
        "rainfall_profile_mm_hr": [float(x) for x in rainfall_profile],
        "infiltration_min_mm_hr": float(np.min(infiltration)),
        "infiltration_mean_mm_hr": float(np.mean(infiltration)),
        "infiltration_max_mm_hr": float(np.max(infiltration)),
        "manning_min": float(np.min(manning)),
        "manning_mean": float(np.mean(manning)),
        "manning_max": float(np.max(manning)),
        "elevation_min_m": float(np.min(data["elevation_m"])),
        "elevation_mean_m": float(np.mean(data["elevation_m"])),
        "elevation_max_m": float(np.max(data["elevation_m"])),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    metric_names = [
        key
        for key in rows[0]
        if key not in {"scenario_index", "scenario_id", "scenario_name", "split", "minute"}
        and isinstance(rows[0][key], (int, float))
    ]
    aggregates: dict[str, float | None] = {}
    for name in metric_names:
        values = [row[name] for row in rows if row[name] is not None]
        aggregates[name] = float(np.mean(values)) if values else None
    return aggregates


def build_grouped_summary(
    rows: list[dict[str, Any]],
    parameters_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    dimensions = (
        "temporal_profile",
        "spatial_pattern",
        "initial_condition",
        "infiltration_mean_mm_hr",
        "manning_mean",
    )
    grouped: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            scenario = parameters_by_index[row["scenario_index"]]
            value = scenario[dimension]
            label = f"{value:.3f}" if isinstance(value, float) else str(value)
            groups.setdefault(label, []).append(row)
        grouped[dimension] = {
            label: {
                "scenario_count": len(
                    {row["scenario_index"] for row in group_rows}
                ),
                "metrics": aggregate_rows(group_rows),
            }
            for label, group_rows in sorted(groups.items())
        }
    return grouped


def save_spatial_report(
    output_dir: Path,
    split: str,
    scenario_info: list[dict[str, Any]],
    predictions: np.ndarray,
    references: np.ndarray,
) -> None:
    np.savez_compressed(
        output_dir / f"{split}_spatial_fields.npz",
        scenario_indices=np.asarray(
            [item["scenario_index"] for item in scenario_info],
            dtype=np.int16,
        ),
        scenario_ids=np.asarray(
            [item["scenario_id"] for item in scenario_info],
            dtype=np.int16,
        ),
        minutes=np.asarray(TARGET_MINUTES, dtype=np.int16),
        predicted_convlstm=predictions,
        expected_anuga=references,
        signed_error=predictions - references,
    )

    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for scenario_number, scenario in enumerate(scenario_info):
        fig, axes = plt.subplots(
            len(TARGET_MINUTES),
            4,
            figsize=(16, 3.6 * len(TARGET_MINUTES)),
            constrained_layout=True,
        )
        for row_index, minute in enumerate(TARGET_MINUTES):
            pred = predictions[scenario_number, row_index]
            truth = references[scenario_number, row_index]
            pred_depth = pred[0]
            truth_depth = truth[0]
            depth_limit = max(
                float(np.max(pred_depth)),
                float(np.max(truth_depth)),
                WET_THRESHOLD_M,
            )
            depth_error = pred_depth - truth_depth
            error_limit = max(float(np.max(np.abs(depth_error))), 1e-6)
            shared_wet = (
                (pred_depth >= WET_THRESHOLD_M)
                & (truth_depth >= WET_THRESHOLD_M)
            )
            speed_error = np.full_like(pred_depth, np.nan)
            if shared_wet.any():
                pred_speed = np.hypot(
                    pred[1, shared_wet] / pred_depth[shared_wet],
                    pred[2, shared_wet] / pred_depth[shared_wet],
                )
                truth_speed = np.hypot(
                    truth[1, shared_wet] / truth_depth[shared_wet],
                    truth[2, shared_wet] / truth_depth[shared_wet],
                )
                speed_error[shared_wet] = pred_speed - truth_speed
            speed_limit = max(
                float(np.nanmax(np.abs(speed_error)))
                if np.isfinite(speed_error).any()
                else 0.0,
                1e-6,
            )

            panels = (
                (truth_depth, "ANUGA expected depth (m)", "Blues", 0, depth_limit),
                (pred_depth, "ConvLSTM predicted depth (m)", "Blues", 0, depth_limit),
                (depth_error, "Depth error: pred - ANUGA (m)", "coolwarm", -error_limit, error_limit),
                (speed_error, "Speed error on shared wet cells (m/s)", "coolwarm", -speed_limit, speed_limit),
            )
            for column, (field, title, cmap, vmin, vmax) in enumerate(panels):
                axis = axes[row_index, column]
                image = axis.imshow(
                    field,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    origin="upper",
                )
                axis.set_title(f"{title}\n{minute} min")
                axis.set_xlabel("Grid column")
                if column == 0:
                    axis.set_ylabel("Grid row")
                fig.colorbar(image, ax=axis, shrink=0.78)
        safe_name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            scenario["scenario_name"],
        ).strip("_")
        fig.suptitle(
            f"Scenario {scenario['scenario_id']}: {scenario['scenario_name']}",
            fontsize=14,
        )
        fig.savefig(
            maps_dir / f"scenario_{scenario['scenario_index']:02d}_{safe_name}.png",
            dpi=120,
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = make_model(args.weights)

    with np.load(args.data, allow_pickle=False) as data, np.load(
        args.normalization,
        allow_pickle=False,
    ) as normalization:
        verify_normalization(data, normalization)
        indices = np.flatnonzero(data["split"] == args.split)
        if indices.size == 0:
            raise ValueError("Dataset contains no validation scenarios")

        parameters_by_index = {
            int(index): scenario_parameters(data, int(index))
            for index in indices
        }
        rows: list[dict[str, Any]] = []
        scenario_info: list[dict[str, Any]] = []
        predictions_by_scenario: list[np.ndarray] = []
        references_by_scenario: list[np.ndarray] = []
        rollout_times: list[float] = []

        for index in indices:
            index = int(index)
            prediction, rollout_seconds = run_rollout(
                model,
                data,
                normalization,
                index,
                carry_hidden_state=args.carry_hidden_state,
            )
            rollout_times.append(rollout_seconds)
            truth = data["state_sequence"][index]
            scenario = parameters_by_index[index]
            scenario_info.append(scenario)
            predictions_by_scenario.append(
                np.stack([prediction[minute] for minute in TARGET_MINUTES])
            )
            references_by_scenario.append(
                np.stack([truth[minute] for minute in TARGET_MINUTES])
            )

            for minute in TARGET_MINUTES:
                metrics = calculate_metrics(prediction[minute], truth[minute])
                rows.append(
                    {
                        "scenario_index": index,
                        "scenario_id": scenario["scenario_id"],
                        "scenario_name": scenario["scenario_name"],
                        "split": str(data["split"][index]),
                        "minute": minute,
                        **metrics,
                    }
                )

        predictions_array = np.stack(predictions_by_scenario)
        references_array = np.stack(references_by_scenario)
        save_spatial_report(
            args.output_dir,
            args.split,
            scenario_info,
            predictions_array,
            references_array,
        )

        csv_path = args.output_dir / "scenario_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = list(rows[0])
            fieldnames[3:3] = [
                "temporal_profile",
                "spatial_pattern",
                "initial_condition",
                "rainfall_profile_mm_hr",
                "infiltration_mean_mm_hr",
                "manning_mean",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                parameters = parameters_by_index[row["scenario_index"]]
                writer.writerow(
                    {
                        **row,
                        "temporal_profile": parameters["temporal_profile"],
                        "spatial_pattern": parameters["spatial_pattern"],
                        "initial_condition": parameters["initial_condition"],
                        "rainfall_profile_mm_hr": json.dumps(
                            parameters["rainfall_profile_mm_hr"]
                        ),
                        "infiltration_mean_mm_hr": parameters[
                            "infiltration_mean_mm_hr"
                        ],
                        "manning_mean": parameters["manning_mean"],
                    }
                )

        by_horizon = {
            str(minute): aggregate_rows(
                [row for row in rows if row["minute"] == minute]
            )
            for minute in TARGET_MINUTES
        }
        grouped = {
            str(minute): build_grouped_summary(
                [row for row in rows if row["minute"] == minute],
                parameters_by_index,
            )
            for minute in TARGET_MINUTES
        }
        report = {
            "reference": f"Stored ANUGA trajectories from {args.data.name}",
            "split": args.split,
            "scenario_count": int(indices.size),
            "weights": str(args.weights),
            "normalization": str(args.normalization),
            "forecast_minutes": list(TARGET_MINUTES),
            "wet_threshold_m": WET_THRESHOLD_M,
            "cell_area_m2": CELL_AREA_M2,
            "device": "cpu",
            "hidden_state_carried": args.carry_hidden_state,
            "torch_version": torch.__version__,
            "model_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "rollout_timing_seconds": {
                "mean": float(np.mean(rollout_times)),
                "median": float(np.median(rollout_times)),
                "min": float(np.min(rollout_times)),
                "max": float(np.max(rollout_times)),
                "median_ms_per_step": float(
                    np.median(rollout_times) * 1000 / 60
                ),
            },
            "macro_average_by_horizon": by_horizon,
            "grouped_metrics_by_horizon": grouped,
            "scenario_parameters": scenario_info,
            "metric_definitions": {
                "depth_and_momentum": "MAE, RMSE, and bias over all grid cells",
                "wet_depth_mean_relative_error_pct": (
                    "Mean absolute relative depth error on ANUGA cells with "
                    "depth >= wet threshold"
                ),
                "wet_area_csi": "Intersection over union at the wet threshold",
                "velocity": (
                    "u, v, vector, and speed MAE where both ANUGA and "
                    "ConvLSTM depths meet the wet threshold"
                ),
                "water_volume_relative_error_pct": (
                    "Absolute difference in total depth-integrated volume "
                    "divided by ANUGA volume"
                ),
                "aggregation": (
                    "Metrics are calculated per scenario, then averaged "
                    "equally across validation scenarios"
                ),
            },
        }
        json_path = args.output_dir / "summary.json"
        json_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(f"{args.split.capitalize()} scenarios: {len(indices)}")
    print(f"Summary: {json_path}")
    print(f"Per-scenario metrics: {csv_path}")
    spatial_fields_path = args.output_dir / f"{args.split}_spatial_fields.npz"
    print(f"Spatial fields: {spatial_fields_path}")
    print(f"Spatial maps: {args.output_dir / 'maps'}")
    print(
        "60-minute macro averages: "
        + json.dumps(by_horizon["60"], separators=(",", ":"))
    )


if __name__ == "__main__":
    main()

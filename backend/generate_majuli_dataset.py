#!/usr/bin/env python3
"""Generate scenario-split ANUGA sequences for a Majuli flood surrogate."""

from __future__ import annotations

import json
from pathlib import Path

import anuga
import numpy as np

from run_majuli_demo import (
    CELL_SIZE_M,
    CRS,
    GRID_CELLS,
    INFILTRATION_MM_HR,
    MANNING_N,
    OUTPUT_STEP_S,
    STORM_DURATION_S,
    load_dem,
)


BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data" / "processed" / "majuli_training"
FRAMES_PER_HOUR = int(STORM_DURATION_S // OUTPUT_STEP_S) + 1
PROFILE_INTERVAL_S = 600.0
FLOOD_DEPTH_THRESHOLD_M = 0.01

TEMPORAL_PROFILES = {
    "uniform_50": [50, 50, 50, 50, 50, 50],
    "front_loaded": [75, 65, 55, 40, 25, 15],
    "back_loaded": [15, 25, 40, 55, 65, 75],
    "double_peak": [25, 55, 75, 25, 55, 75],
    "intermittent": [0, 70, 0, 60, 0, 70],
    "moderate_variable": [30, 40, 50, 55, 40, 30],
}
SPATIAL_PATTERNS = (
    "uniform",
    "east_west_gradient",
    "north_south_gradient",
    "localized_cell",
)


def rainfall_multiplier(
    pattern_name: str,
    x: np.ndarray,
    y: np.ndarray,
    xll: float,
    yll: float,
) -> np.ndarray:
    x_fraction = np.clip(
        (x - xll) / (GRID_CELLS * CELL_SIZE_M),
        0.0,
        1.0,
    )
    y_fraction = np.clip(
        (y - yll) / (GRID_CELLS * CELL_SIZE_M),
        0.0,
        1.0,
    )
    if pattern_name == "uniform":
        return np.ones_like(x_fraction)
    if pattern_name == "east_west_gradient":
        return 0.4 + 1.2 * x_fraction
    if pattern_name == "north_south_gradient":
        return 0.4 + 1.2 * y_fraction
    if pattern_name == "localized_cell":
        raw = 0.25 + 3.0 * np.exp(
            -(
                ((x_fraction - 0.68) / 0.22) ** 2
                + ((y_fraction - 0.34) / 0.27) ** 2
            )
        )
        return raw
    raise ValueError(f"Unknown rainfall pattern {pattern_name!r}")


def rasterize(values: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    sums = np.bincount(
        cell_ids,
        weights=values,
        minlength=GRID_CELLS**2,
    )
    counts = np.bincount(cell_ids, minlength=GRID_CELLS**2)
    return (sums / counts).reshape(GRID_CELLS, GRID_CELLS)


def scenario_split(temporal_index: int, spatial_index: int) -> str:
    validation_profile = spatial_index
    test_profile = (spatial_index + 3) % len(TEMPORAL_PROFILES)
    if temporal_index == validation_profile:
        return "validation"
    if temporal_index == test_profile:
        return "test"
    return "train"


def run_scenario(
    dem: np.ndarray,
    xll: float,
    yll: float,
    profile: list[float],
    spatial_pattern: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    domain = anuga.rectangular_cross_domain(
        GRID_CELLS,
        GRID_CELLS,
        len1=GRID_CELLS * CELL_SIZE_M,
        len2=GRID_CELLS * CELL_SIZE_M,
        origin=(xll, yll),
    )
    domain.set_store(False)

    def elevation_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        columns = np.floor((x - xll) / CELL_SIZE_M).astype(np.int64)
        rows_from_south = np.floor((y - yll) / CELL_SIZE_M).astype(np.int64)
        columns = np.clip(columns, 0, GRID_CELLS - 1)
        rows_from_south = np.clip(rows_from_south, 0, GRID_CELLS - 1)
        return dem[GRID_CELLS - 1 - rows_from_south, columns]

    domain.set_quantity("elevation", function=elevation_at)
    domain.set_quantity("stage", expression="elevation")
    domain.set_quantity("friction", MANNING_N)
    reflective = anuga.Reflective_boundary(domain)
    domain.set_boundary(
        {
            "left": reflective,
            "right": reflective,
            "top": reflective,
            "bottom": reflective,
        }
    )

    x_grid, y_grid = np.meshgrid(
        xll + (np.arange(GRID_CELLS) + 0.5) * CELL_SIZE_M,
        yll + (np.arange(GRID_CELLS)[::-1] + 0.5) * CELL_SIZE_M,
    )
    pattern_mean = float(
        np.mean(rainfall_multiplier(spatial_pattern, x_grid, y_grid, xll, yll))
    )

    def net_rainfall_rate(x: np.ndarray, y: np.ndarray, time_s: float) -> np.ndarray:
        profile_index = min(
            int(max(time_s, 0.0) // PROFILE_INTERVAL_S),
            len(profile) - 1,
        )
        rainfall_mm_hr = (
            profile[profile_index]
            * rainfall_multiplier(spatial_pattern, x, y, xll, yll)
            / pattern_mean
        )
        effective_mm_hr = np.maximum(
            rainfall_mm_hr - INFILTRATION_MM_HR,
            0.0,
        )
        return effective_mm_hr / 3_600_000.0

    rainfall_operator = anuga.Rate_operator(
        domain,
        rate=net_rainfall_rate,
    )

    centroids = domain.get_centroid_coordinates(absolute=True)
    columns = np.floor((centroids[:, 0] - xll) / CELL_SIZE_M).astype(np.int64)
    rows_from_south = np.floor(
        (centroids[:, 1] - yll) / CELL_SIZE_M
    ).astype(np.int64)
    rows = GRID_CELLS - 1 - rows_from_south
    cell_ids = rows * GRID_CELLS + columns
    cell_counts = np.bincount(cell_ids, minlength=GRID_CELLS**2)
    if not np.all(cell_counts == 4):
        raise ValueError("Could not map ANUGA triangle centroids to raster cells")

    states = np.empty((FRAMES_PER_HOUR, 3, GRID_CELLS, GRID_CELLS), dtype=np.float32)
    rainfall_maps = np.empty(
        (FRAMES_PER_HOUR, GRID_CELLS, GRID_CELLS),
        dtype=np.float32,
    )
    mass_errors = np.empty(FRAMES_PER_HOUR, dtype=np.float32)
    cell_areas = domain.areas
    expected_volume = 0.0

    for frame_index, model_time in enumerate(
        domain.evolve(
            yieldstep=OUTPUT_STEP_S,
            finaltime=STORM_DURATION_S,
        )
    ):
        elevation = domain.quantities["elevation"].centroid_values
        stage = domain.quantities["stage"].centroid_values
        depth = np.maximum(stage - elevation, 0.0)
        xmomentum = domain.quantities["xmomentum"].centroid_values
        ymomentum = domain.quantities["ymomentum"].centroid_values
        states[frame_index, 0] = rasterize(depth, cell_ids)
        states[frame_index, 1] = rasterize(xmomentum, cell_ids)
        states[frame_index, 2] = rasterize(ymomentum, cell_ids)

        frame_profile_index = min(
            int(max(float(model_time), 0.0) // PROFILE_INTERVAL_S),
            len(profile) - 1,
        )
        rain_at_cells = (
            profile[frame_profile_index]
            * rainfall_multiplier(spatial_pattern, x_grid, y_grid, xll, yll)
            / pattern_mean
        )
        effective_at_cells = np.maximum(
            rain_at_cells - INFILTRATION_MM_HR,
            0.0,
        )
        rainfall_maps[frame_index] = effective_at_cells / 3_600_000.0

        volume_m3 = float(np.sum(depth * cell_areas))
        expected_volume = float(rainfall_operator.cumulative_influx)
        mass_errors[frame_index] = abs(volume_m3 - expected_volume) / max(
            expected_volume,
            1.0,
        )

    if frame_index + 1 != FRAMES_PER_HOUR:
        raise RuntimeError(
            f"Expected {FRAMES_PER_HOUR} simulation frames, got {frame_index + 1}"
        )
    return states, rainfall_maps, mass_errors


def main() -> None:
    dem, xll, yll = load_dem()
    temporal_names = list(TEMPORAL_PROFILES)
    spatial_names = list(SPATIAL_PATTERNS)
    scenario_records: list[dict[str, object]] = []
    all_states: list[np.ndarray] = []
    all_rainfall: list[np.ndarray] = []
    all_mass_errors: list[np.ndarray] = []

    for temporal_index, temporal_name in enumerate(temporal_names):
        profile = TEMPORAL_PROFILES[temporal_name]
        for spatial_index, spatial_name in enumerate(spatial_names):
            scenario_id = len(scenario_records)
            print(
                f"Scenario {scenario_id + 1:02d}/"
                f"{len(temporal_names) * len(spatial_names)}: "
                f"{temporal_name} / {spatial_name}"
            )
            states, rainfall_maps, mass_errors = run_scenario(
                dem,
                xll,
                yll,
                profile,
                spatial_name,
            )
            all_states.append(states)
            all_rainfall.append(rainfall_maps)
            all_mass_errors.append(mass_errors)
            speed = np.divide(
                np.hypot(states[:, 1], states[:, 2]),
                states[:, 0],
                out=np.zeros_like(states[:, 0]),
                where=states[:, 0] >= FLOOD_DEPTH_THRESHOLD_M,
            )
            scenario_records.append(
                {
                    "scenario_id": scenario_id,
                    "name": f"{temporal_name}__{spatial_name}",
                    "temporal_profile": temporal_name,
                    "spatial_pattern": spatial_name,
                    "rainfall_mm_hr_by_10min": profile,
                    "split": scenario_split(temporal_index, spatial_index),
                    "max_depth_m": float(states[:, 0].max()),
                    "max_speed_m_s_where_depth_ge_1cm": float(speed.max()),
                    "max_mass_balance_relative_error": float(mass_errors.max()),
                    "final_mass_balance_relative_error": float(mass_errors[-1]),
                    "final_inundated_cells_at_1cm": int(
                        np.count_nonzero(states[-1, 0] >= FLOOD_DEPTH_THRESHOLD_M)
                    ),
                }
            )

    states_array = np.stack(all_states)
    rainfall_array = np.stack(all_rainfall)
    mass_error_array = np.stack(all_mass_errors)
    scenario_ids = np.arange(len(scenario_records), dtype=np.int16)
    splits = np.asarray(
        [record["split"] for record in scenario_records],
        dtype="U10",
    )
    train_ids = np.flatnonzero(splits == "train")
    train_states = states_array[train_ids]
    state_mean = train_states.mean(axis=(0, 1, 3, 4), dtype=np.float64)
    state_std = train_states.std(axis=(0, 1, 3, 4), dtype=np.float64)
    train_rainfall = rainfall_array[train_ids]
    rainfall_mean = float(train_rainfall.mean(dtype=np.float64))
    rainfall_std = float(train_rainfall.std(dtype=np.float64))
    elevation_mean = float(dem.mean())
    elevation_std = float(dem.std())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "majuli_anuga_sequences.npz"
    times = np.arange(FRAMES_PER_HOUR, dtype=np.float32) * OUTPUT_STEP_S
    np.savez_compressed(
        archive_path,
        state_sequence=np.asarray(states_array, dtype=np.float32),
        wet_mask_1cm=np.asarray(
            states_array[:, :, 0] >= FLOOD_DEPTH_THRESHOLD_M,
            dtype=np.bool_,
        ),
        rainfall_effective_m_s=np.asarray(rainfall_array, dtype=np.float32),
        elevation_m=np.asarray(dem, dtype=np.float32),
        manning_n=np.full(dem.shape, MANNING_N, dtype=np.float32),
        times_seconds=times,
        scenario_ids=scenario_ids,
        scenario_names=np.asarray(
            [record["name"] for record in scenario_records],
            dtype="U64",
        ),
        temporal_profiles=np.asarray(
            [record["temporal_profile"] for record in scenario_records],
            dtype="U32",
        ),
        spatial_patterns=np.asarray(
            [record["spatial_pattern"] for record in scenario_records],
            dtype="U32",
        ),
        split=splits,
        mass_balance_relative_error=mass_error_array,
        state_channel_names=np.asarray(
            ["depth_m", "xmomentum_m2_s", "ymomentum_m2_s"],
            dtype="U32",
        ),
        input_channel_names=np.asarray(
            [
                "depth_m",
                "xmomentum_m2_s",
                "ymomentum_m2_s",
                "effective_rainfall_m_s",
                "elevation_m",
                "manning_n",
            ],
            dtype="U32",
        ),
        target_channel_names=np.asarray(
            ["depth_next_m", "xmomentum_next_m2_s", "ymomentum_next_m2_s"],
            dtype="U32",
        ),
        normalization_state_mean=np.asarray(state_mean, dtype=np.float32),
        normalization_state_std=np.maximum(state_std, 1e-8).astype(np.float32),
        normalization_rainfall_mean_m_s=np.float32(rainfall_mean),
        normalization_rainfall_std_m_s=np.float32(max(rainfall_std, 1e-8)),
        normalization_elevation_mean_m=np.float32(elevation_mean),
        normalization_elevation_std_m=np.float32(max(elevation_std, 1e-8)),
    )

    manifest = {
        "location": "Garamur, Majuli, Assam, India",
        "center": {"latitude": 26.980306, "longitude": 94.1574452},
        "elevation_source": (
            "NASA/USGS SRTM 1 arc-second HGT via AWS Terrain Tiles "
            "(N26E094)"
        ),
        "model": "ANUGA 4.2.0 shallow-water finite-volume solver",
        "crs": CRS,
        "grid": {
            "rows": GRID_CELLS,
            "columns": GRID_CELLS,
            "cell_size_m": CELL_SIZE_M,
            "domain_width_m": GRID_CELLS * CELL_SIZE_M,
        },
        "simulation": {
            "duration_s": STORM_DURATION_S,
            "output_interval_s": OUTPUT_STEP_S,
            "rainfall_profile_interval_s": PROFILE_INTERVAL_S,
            "infiltration_loss_mm_hr": INFILTRATION_MM_HR,
            "manning_n": MANNING_N,
            "boundary": "reflective on all sides",
            "state_channels": [
                "depth h (m)",
                "x momentum hu (m^2/s)",
                "y momentum hv (m^2/s)",
            ],
            "velocity_validity_threshold_m": FLOOD_DEPTH_THRESHOLD_M,
            "rainfall_channel": "effective rainfall after fixed loss (m/s)",
            "target_depth_mask_threshold_m": FLOOD_DEPTH_THRESHOLD_M,
        },
        "array_shapes": {
            "state_sequence": list(states_array.shape),
            "wet_mask_1cm": list(states_array[:, :, 0].shape),
            "rainfall_effective_m_s": list(rainfall_array.shape),
            "elevation_m": list(dem.shape),
            "manning_n": list(dem.shape),
            "mass_balance_relative_error": list(mass_error_array.shape),
        },
        "sequence_learning": {
            "input": (
                "Current state channels plus rainfall, elevation, and Manning "
                "roughness; concatenate/expand the static grids as channels."
            ),
            "target": "Next time-step state channels (h, hu, hv).",
            "velocity_note": (
                "Derive velocity as (hu/h, hv/h) only where depth is at least "
                "0.01 m; use wet_mask_1cm to mask velocity or momentum losses "
                "in near-dry cells."
            ),
            "split_policy": (
                "Train/validation/test splits are assigned by whole scenario, "
                "never by randomly splitting adjacent time frames."
            ),
            "scenario_counts": {
                split: int(np.count_nonzero(splits == split))
                for split in ("train", "validation", "test")
            },
            "normalization_statistics": {
                "state_mean": state_mean.tolist(),
                "state_std": np.maximum(state_std, 1e-8).tolist(),
                "rainfall_effective_mean_m_s": rainfall_mean,
                "rainfall_effective_std_m_s": max(rainfall_std, 1e-8),
                "elevation_mean_m": elevation_mean,
                "elevation_std_m": max(elevation_std, 1e-8),
                "computed_from": "training scenarios only",
            },
        },
        "accuracy_note": (
            "Mass-balance error checks numerical conservation, not real-world "
            "flood prediction accuracy. No observed Majuli flood depths or "
            "extents were used."
        ),
        "scenarios": scenario_records,
    }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    max_mass_error = float(mass_error_array.max())
    if max_mass_error > 0.01:
        raise RuntimeError(
            f"Dataset mass-balance error {max_mass_error:.2%} exceeds 1%"
        )
    print(
        f"Saved {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"Scenarios: {len(scenario_records)}; one-step transitions: "
        f"{len(scenario_records) * (FRAMES_PER_HOUR - 1)}"
    )
    print(
        "Scenario split counts: "
        + ", ".join(
            f"{name}={int(np.count_nonzero(splits == name))}"
            for name in ("train", "validation", "test")
        )
    )
    print(f"Maximum relative mass-balance error: {max_mass_error:.3e}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

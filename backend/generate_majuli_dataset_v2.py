#!/usr/bin/env python3
"""Generate parameterized ANUGA sequences for ConvLSTM training."""

from __future__ import annotations

import json
from pathlib import Path

import anuga
import numpy as np

from run_majuli_demo import (
    CELL_SIZE_M,
    CRS,
    GRID_CELLS,
    OUTPUT_STEP_S,
    STORM_DURATION_S,
    load_dem,
)


BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data" / "processed" / "majuli_training_v2"

PROFILE_INTERVAL_S = 600.0
FLOOD_DEPTH_THRESHOLD_M = 0.01
PARAMETER_LEVELS = {
    "manning_n": (0.03, 0.05, 0.08),
    "infiltration_loss_mm_hr": (0.0, 5.0, 10.0),
    "initial_condition": ("dry", "uniform_0p02m", "lowland_puddle"),
}
TEMPORAL_PROFILES = {
    "uniform_50": [50, 50, 50, 50, 50, 50],
    "front_loaded": [75, 65, 55, 40, 25, 15],
    "back_loaded": [15, 25, 40, 55, 65, 75],
    "double_peak": [25, 55, 75, 25, 55, 75],
    "short_burst_20min": [70, 70, 0, 0, 0, 0],
    "short_burst_40min": [45, 45, 45, 45, 0, 0],
}
SPATIAL_PATTERNS = (
    "uniform",
    "east_west_gradient",
    "north_south_gradient",
    "localized_cell",
)
REPLICATES_PER_RAINFALL_PATTERN = 3
FRAME_COUNT = int(STORM_DURATION_S // OUTPUT_STEP_S) + 1


def rainfall_multiplier(
    pattern_name: str,
    x: np.ndarray,
    y: np.ndarray,
    xll: float,
    yll: float,
) -> np.ndarray:
    width_m = GRID_CELLS * CELL_SIZE_M
    x_fraction = np.clip((x - xll) / width_m, 0.0, 1.0)
    y_fraction = np.clip((y - yll) / width_m, 0.0, 1.0)
    if pattern_name == "uniform":
        return np.ones_like(x_fraction)
    if pattern_name == "east_west_gradient":
        return 0.4 + 1.2 * x_fraction
    if pattern_name == "north_south_gradient":
        return 0.4 + 1.2 * y_fraction
    if pattern_name == "localized_cell":
        return 0.8 + 0.8 * np.exp(
            -(
                ((x_fraction - 0.68) / 0.22) ** 2
                + ((y_fraction - 0.34) / 0.27) ** 2
            )
        )
    raise ValueError(f"Unknown rainfall pattern {pattern_name!r}")

def initial_depth_map(
    dem: np.ndarray,
    condition: str,
) -> np.ndarray:
    if condition == "dry":
        return np.zeros_like(dem, dtype=np.float64)
    if condition == "uniform_0p02m":
        return np.full_like(dem, 0.02, dtype=np.float64)
    if condition == "lowland_puddle":
        return np.where(dem <= dem.min() + 0.10, dem.min() + 0.10 - dem, 0.0)
    raise ValueError(f"Unknown initial condition {condition!r}")

def split_for(temporal_index: int, spatial_index: int) -> str:
    validation_pattern = temporal_index % len(SPATIAL_PATTERNS)
    test_pattern = (temporal_index + 2) % len(SPATIAL_PATTERNS)
    if spatial_index == validation_pattern:
        return "validation"
    if spatial_index == test_pattern:
        return "test"
    return "train"

def rasterize(values: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    sums = np.bincount(
        cell_ids,
        weights=values,
        minlength=GRID_CELLS**2,
    )
    counts = np.bincount(cell_ids, minlength=GRID_CELLS**2)
    return (sums / counts).reshape(GRID_CELLS, GRID_CELLS)

def run_scenario(
    dem: np.ndarray,
    xll: float,
    yll: float,
    rainfall_profile_mm_hr: list[float],
    spatial_pattern: str,
    manning_n: float,
    infiltration_loss_mm_hr: float,
    initial_condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    domain = anuga.rectangular_cross_domain(
        GRID_CELLS,
        GRID_CELLS,
        len1=GRID_CELLS * CELL_SIZE_M,
        len2=GRID_CELLS * CELL_SIZE_M,
        origin=(xll, yll),
    )
    domain.set_store(False)
    initial_depth_cells = initial_depth_map(dem, initial_condition)

    def cell_indices(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        columns = np.floor((x - xll) / CELL_SIZE_M).astype(np.int64)
        rows_from_south = np.floor((y - yll) / CELL_SIZE_M).astype(np.int64)
        columns = np.clip(columns, 0, GRID_CELLS - 1)
        rows_from_south = np.clip(rows_from_south, 0, GRID_CELLS - 1)
        rows = GRID_CELLS - 1 - rows_from_south
        return rows, columns

    def elevation_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows, columns = cell_indices(x, y)
        return dem[rows, columns]

    def initial_depth_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows, columns = cell_indices(x, y)
        return initial_depth_cells[rows, columns]

    domain.set_quantity("elevation", function=elevation_at)
    domain.set_quantity(
        "stage",
        function=lambda x, y: elevation_at(x, y) + initial_depth_at(x, y),
    )
    domain.set_quantity("xmomentum", 0.0)
    domain.set_quantity("ymomentum", 0.0)
    domain.set_quantity("friction", manning_n)
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

    def rainfall_rate_m_s(
        x: np.ndarray,
        y: np.ndarray,
        time_s: float,
    ) -> np.ndarray:
        profile_index = min(
            int(max(time_s, 0.0) // PROFILE_INTERVAL_S),
            len(rainfall_profile_mm_hr) - 1,
        )
        raw_mm_hr = (
            rainfall_profile_mm_hr[profile_index]
            * rainfall_multiplier(spatial_pattern, x, y, xll, yll)
            / pattern_mean
        )
        net_mm_hr = np.maximum(raw_mm_hr - infiltration_loss_mm_hr, 0.0)
        return net_mm_hr / 3_600_000.0

    rainfall_operator = anuga.Rate_operator(domain, rate=rainfall_rate_m_s)
    centroids = domain.get_centroid_coordinates(absolute=True)
    rows, columns = cell_indices(centroids[:, 0], centroids[:, 1])
    cell_ids = rows * GRID_CELLS + columns
    counts = np.bincount(cell_ids, minlength=GRID_CELLS**2)
    if not np.all(counts == 4):
        raise ValueError("Could not map ANUGA triangle centroids to raster cells")

    initial_depth_triangles = initial_depth_at(
        centroids[:, 0],
        centroids[:, 1],
    )
    initial_volume_m3 = float(np.sum(initial_depth_triangles * domain.areas))
    states = np.empty(
        (FRAME_COUNT, 3, GRID_CELLS, GRID_CELLS),
        dtype=np.float32,
    )
    raw_rainfall = np.empty(
        (FRAME_COUNT, GRID_CELLS, GRID_CELLS),
        dtype=np.float32,
    )
    effective_rainfall = np.empty_like(raw_rainfall)
    mass_errors = np.empty(FRAME_COUNT, dtype=np.float32)

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

        profile_index = min(
            int(max(float(model_time), 0.0) // PROFILE_INTERVAL_S),
            len(rainfall_profile_mm_hr) - 1,
        )
        raw_at_cells = (
            rainfall_profile_mm_hr[profile_index]
            * rainfall_multiplier(spatial_pattern, x_grid, y_grid, xll, yll)
            / pattern_mean
        )
        net_at_cells = np.maximum(
            raw_at_cells - infiltration_loss_mm_hr,
            0.0,
        )
        raw_rainfall[frame_index] = raw_at_cells
        effective_rainfall[frame_index] = net_at_cells / 3_600_000.0

        modeled_volume_m3 = float(np.sum(depth * domain.areas))
        expected_volume_m3 = (
            initial_volume_m3 + float(rainfall_operator.cumulative_influx)
        )
        mass_errors[frame_index] = abs(
            modeled_volume_m3 - expected_volume_m3
        ) / max(expected_volume_m3, 1.0)

    if frame_index + 1 != FRAME_COUNT:
        raise RuntimeError(
            f"Expected {FRAME_COUNT} frames, got {frame_index + 1}"
        )
    return (
        states,
        raw_rainfall,
        effective_rainfall,
        states[0].copy(),
        mass_errors,
    )

def main() -> None:
    dem, xll, yll = load_dem()
    temporal_names = list(TEMPORAL_PROFILES)
    spatial_names = list(SPATIAL_PATTERNS)
    scenario_records: list[dict[str, object]] = []
    states_list: list[np.ndarray] = []
    raw_rain_list: list[np.ndarray] = []
    effective_rain_list: list[np.ndarray] = []
    initial_states_list: list[np.ndarray] = []
    mass_errors_list: list[np.ndarray] = []

    total_scenarios = (
        len(temporal_names)
        * len(spatial_names)
        * REPLICATES_PER_RAINFALL_PATTERN
    )
    for temporal_index, (temporal_name, profile) in enumerate(
        TEMPORAL_PROFILES.items()
    ):
        for spatial_index, spatial_name in enumerate(spatial_names):
            split = split_for(temporal_index, spatial_index)
            base_id = temporal_index * len(spatial_names) + spatial_index
            for replicate in range(REPLICATES_PER_RAINFALL_PATTERN):
                scenario_id = len(scenario_records)
                roughness = PARAMETER_LEVELS["manning_n"][
                    (base_id + replicate) % 3
                ]
                infiltration = PARAMETER_LEVELS["infiltration_loss_mm_hr"][
                    (base_id // 2 + 2 * replicate) % 3
                ]
                initial_condition = PARAMETER_LEVELS["initial_condition"][
                    (base_id + 2 * replicate) % 3
                ]
                print(
                    f"Scenario {scenario_id + 1:02d}/{total_scenarios}: "
                    f"{temporal_name} / {spatial_name} / "
                    f"n={roughness:.2f} / I={infiltration:.0f} / "
                    f"initial={initial_condition}"
                )
                states, raw_rain, effective_rain, initial_state, mass_errors = (
                    run_scenario(
                        dem,
                        xll,
                        yll,
                        profile,
                        spatial_name,
                        roughness,
                        infiltration,
                        initial_condition,
                    )
                )
                states_list.append(states)
                raw_rain_list.append(raw_rain)
                effective_rain_list.append(effective_rain)
                initial_states_list.append(initial_state)
                mass_errors_list.append(mass_errors)
                speed = np.divide(
                    np.hypot(states[:, 1], states[:, 2]),
                    states[:, 0],
                    out=np.zeros_like(states[:, 0]),
                    where=states[:, 0] >= FLOOD_DEPTH_THRESHOLD_M,
                )
                scenario_records.append(
                    {
                        "scenario_id": scenario_id,
                        "name": (
                            f"{temporal_name}__{spatial_name}__rep{replicate}"
                        ),
                        "temporal_profile": temporal_name,
                        "spatial_pattern": spatial_name,
                        "rainfall_mm_hr_by_10min": profile,
                        "manning_n": roughness,
                        "infiltration_loss_mm_hr": infiltration,
                        "initial_condition": initial_condition,
                        "boundary": "reflective on all sides",
                        "split": split,
                        "max_depth_m": float(states[:, 0].max()),
                        "max_speed_m_s_where_depth_ge_1cm": float(speed.max()),
                        "max_mass_balance_relative_error": float(
                            mass_errors.max()
                        ),
                        "final_mass_balance_relative_error": float(
                            mass_errors[-1]
                        ),
                        "final_inundated_cells_at_1cm": int(
                            np.count_nonzero(
                                states[-1, 0] >= FLOOD_DEPTH_THRESHOLD_M
                            )
                        ),
                    }
                )

    states_array = np.stack(states_list)
    raw_rain_array = np.stack(raw_rain_list)
    effective_rain_array = np.stack(effective_rain_list)
    initial_states = np.stack(initial_states_list)
    mass_error_array = np.stack(mass_errors_list)
    scenario_ids = np.arange(total_scenarios, dtype=np.int16)
    splits = np.asarray(
        [record["split"] for record in scenario_records],
        dtype="U10",
    )
    train_ids = np.flatnonzero(splits == "train")
    train_states = states_array[train_ids]
    state_mean = train_states.mean(axis=(0, 1, 3, 4), dtype=np.float64)
    state_std = train_states.std(axis=(0, 1, 3, 4), dtype=np.float64)
    rain_mean = float(raw_rain_array[train_ids].mean(dtype=np.float64))
    rain_std = float(raw_rain_array[train_ids].std(dtype=np.float64))
    effective_rain_mean = float(
        effective_rain_array[train_ids].mean(dtype=np.float64)
    )
    effective_rain_std = float(
        effective_rain_array[train_ids].std(dtype=np.float64)
    )
    train_infiltration = np.asarray(
        [scenario_records[index]["infiltration_loss_mm_hr"] for index in train_ids],
        dtype=np.float64,
    )
    train_roughness = np.asarray(
        [scenario_records[index]["manning_n"] for index in train_ids],
        dtype=np.float64,
    )

    infiltration_maps = np.stack(
        [
            np.full(dem.shape, record["infiltration_loss_mm_hr"], dtype=np.float32)
            for record in scenario_records
        ]
    )
    roughness_maps = np.stack(
        [
            np.full(dem.shape, record["manning_n"], dtype=np.float32)
            for record in scenario_records
        ]
    )
    times = np.arange(FRAME_COUNT, dtype=np.float32) * OUTPUT_STEP_S
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "majuli_anuga_convlstm_v2.npz"
    np.savez_compressed(
        archive_path,
        state_sequence=np.asarray(states_array, dtype=np.float32),
        initial_state=np.asarray(initial_states, dtype=np.float32),
        raw_rainfall_mm_hr=np.asarray(raw_rain_array, dtype=np.float32),
        rainfall_effective_m_s=np.asarray(
            effective_rain_array,
            dtype=np.float32,
        ),
        infiltration_loss_mm_hr=infiltration_maps,
        elevation_m=np.asarray(dem, dtype=np.float32),
        manning_n=roughness_maps,
        wet_mask_1cm=np.asarray(
            states_array[:, :, 0] >= FLOOD_DEPTH_THRESHOLD_M,
            dtype=np.bool_,
        ),
        times_seconds=times,
        scenario_ids=scenario_ids,
        scenario_names=np.asarray(
            [record["name"] for record in scenario_records],
            dtype="U80",
        ),
        temporal_profiles=np.asarray(
            [record["temporal_profile"] for record in scenario_records],
            dtype="U32",
        ),
        spatial_patterns=np.asarray(
            [record["spatial_pattern"] for record in scenario_records],
            dtype="U32",
        ),
        initial_condition_names=np.asarray(
            [record["initial_condition"] for record in scenario_records],
            dtype="U32",
        ),
        rainfall_profiles_mm_hr=np.asarray(
            [record["rainfall_mm_hr_by_10min"] for record in scenario_records],
            dtype=np.float32,
        ),
        split=splits,
        mass_balance_relative_error=mass_error_array,
        state_channel_names=np.asarray(
            ["depth_m", "xmomentum_m2_s", "ymomentum_m2_s"],
            dtype="U32",
        ),
        raw_rainfall_units=np.asarray("mm/hour"),
        effective_rainfall_units=np.asarray("m/second"),
        input_channel_names=np.asarray(
            [
                "depth_m",
                "xmomentum_m2_s",
                "ymomentum_m2_s",
                "raw_rainfall_mm_hr",
                "infiltration_loss_mm_hr",
                "elevation_m",
                "manning_n",
            ],
            dtype="U32",
        ),
        normalization_state_mean=np.asarray(state_mean, dtype=np.float32),
        normalization_state_std=np.maximum(state_std, 1e-8).astype(np.float32),
        normalization_raw_rainfall_mean_mm_hr=np.float32(rain_mean),
        normalization_raw_rainfall_std_mm_hr=np.float32(max(rain_std, 1e-8)),
        normalization_effective_rainfall_mean_m_s=np.float32(
            effective_rain_mean
        ),
        normalization_effective_rainfall_std_m_s=np.float32(
            max(effective_rain_std, 1e-8)
        ),
        normalization_infiltration_mean_mm_hr=np.float32(
            train_infiltration.mean()
        ),
        normalization_infiltration_std_mm_hr=np.float32(
            max(train_infiltration.std(), 1e-8)
        ),
        normalization_manning_mean=np.float32(train_roughness.mean()),
        normalization_manning_std=np.float32(
            max(train_roughness.std(), 1e-8)
        ),
        normalization_elevation_mean_m=np.float32(dem.mean()),
        normalization_elevation_std_m=np.float32(max(dem.std(), 1e-8)),
    )

    max_mass_error = float(mass_error_array.max())
    if max_mass_error > 0.01:
        raise RuntimeError(
            f"Dataset mass-balance error {max_mass_error:.2%} exceeds 1%"
        )
    counts_by_split = {
        split: int(np.count_nonzero(splits == split))
        for split in ("train", "validation", "test")
    }
    manifest = {
        "dataset_version": 2,
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
            "state_channels": [
                "depth h (m)",
                "x momentum hu (m^2/s)",
                "y momentum hv (m^2/s)",
            ],
            "rainfall_input": "raw spatial rainfall field in mm/hour",
            "infiltration_input": (
                "scenario-constant loss in mm/hour, applied as "
                "max(rainfall - loss, 0); not a Green-Ampt model"
            ),
            "manning_n_levels": PARAMETER_LEVELS["manning_n"],
            "infiltration_loss_levels_mm_hr": PARAMETER_LEVELS[
                "infiltration_loss_mm_hr"
            ],
            "initial_condition_types": PARAMETER_LEVELS["initial_condition"],
            "boundary": (
                "reflective on all sides; no river inflow or outflow is "
                "represented"
            ),
            "velocity_validity_threshold_m": FLOOD_DEPTH_THRESHOLD_M,
        },
        "array_shapes": {
            "state_sequence": list(states_array.shape),
            "initial_state": list(initial_states.shape),
            "raw_rainfall_mm_hr": list(raw_rain_array.shape),
            "rainfall_effective_m_s": list(effective_rain_array.shape),
            "infiltration_loss_mm_hr": list(infiltration_maps.shape),
            "manning_n": list(roughness_maps.shape),
            "elevation_m": list(dem.shape),
            "wet_mask_1cm": list(states_array[:, :, 0].shape),
            "mass_balance_relative_error": list(mass_error_array.shape),
        },
        "sequence_learning": {
            "initial_state": (
                "ANUGA rasterized state at t=0; equal to state_sequence[:, 0]."
            ),
            "input": (
                "Recent state sequence [h, hu, hv], raw rainfall, infiltration "
                "loss, DEM, and Manning n."
            ),
            "target": "Next ANUGA state [h, hu, hv].",
            "split_policy": (
                "Complete rainfall-pattern groups are assigned to a single "
                "split; time frames from one scenario are never separated."
            ),
            "scenario_counts": counts_by_split,
            "normalization_statistics": {
                "state_mean": state_mean.tolist(),
                "state_std": np.maximum(state_std, 1e-8).tolist(),
                "raw_rainfall_mean_mm_hr": rain_mean,
                "raw_rainfall_std_mm_hr": max(rain_std, 1e-8),
                "effective_rainfall_mean_m_s": effective_rain_mean,
                "effective_rainfall_std_m_s": max(effective_rain_std, 1e-8),
                "infiltration_mean_mm_hr": float(train_infiltration.mean()),
                "infiltration_std_mm_hr": float(
                    max(train_infiltration.std(), 1e-8)
                ),
                "manning_n_mean": float(train_roughness.mean()),
                "manning_n_std": float(max(train_roughness.std(), 1e-8)),
                "elevation_mean_m": float(dem.mean()),
                "elevation_std_m": float(max(dem.std(), 1e-8)),
                "computed_from": "training scenarios only",
            },
        },
        "accuracy_note": (
            "Mass balance checks numerical conservation only. Rainfall, "
            "initial states, infiltration, and parameter combinations are "
            "synthetic; no observed flood depths or river levels were used."
        ),
        "scenarios": scenario_records,
    }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Manifest saved")

if __name__ == "__main__":
    main()

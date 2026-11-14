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
    pass

if __name__ == "__main__":
    main()

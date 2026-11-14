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

def main() -> None:
    pass

if __name__ == "__main__":
    main()

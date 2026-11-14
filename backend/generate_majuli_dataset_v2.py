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


def main() -> None:
    pass

if __name__ == "__main__":
    main()

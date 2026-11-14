"""Run a parameterized ANUGA simulation and rasterize its triangle output."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


GRID_SIZE = 50
CELL_SIZE_M = 30.0
OUTPUT_STEP_S = 60.0
INFILTRATION_MM_HR = 5.0
MANNING_N = 0.05


def _rainfall_rate(pattern: str, intensity: float, duration_s: float, time_s: float) -> float:
    if time_s >= duration_s:
        return 0.0
    position = min(max(time_s / max(duration_s, 1.0), 0.0), 1.0)
    if pattern == "increasing":
        multiplier = position
    elif pattern == "peak":
        multiplier = position * 2.0 if position < 0.5 else 2.0 - position * 2.0
    else:
        multiplier = 1.0
    return max(intensity * multiplier - INFILTRATION_MM_HR, 0.0) / 3_600_000.0


def run(
    dem: np.ndarray,
    xll: float,
    yll: float,
    rainfall_intensity_mm_hr: float,
    duration_min: int,
    pattern: str,
    forecast_duration_min: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ANUGA and return minutes, rasterized depths, and rasterized velocity."""
    try:
        import anuga
    except ImportError as error:
        raise RuntimeError(
            "Live ANUGA is unavailable. Install backend/requirements.txt to enable it."
        ) from error

    domain = anuga.rectangular_cross_domain(
        GRID_SIZE,
        GRID_SIZE,
        len1=GRID_SIZE * CELL_SIZE_M,
        len2=GRID_SIZE * CELL_SIZE_M,
        origin=(xll, yll),
    )

    def elevation_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        columns = np.floor((x - xll) / CELL_SIZE_M).astype(np.int64)
        rows_from_south = np.floor((y - yll) / CELL_SIZE_M).astype(np.int64)
        columns = np.clip(columns, 0, GRID_SIZE - 1)
        rows_from_south = np.clip(rows_from_south, 0, GRID_SIZE - 1)
        return dem[GRID_SIZE - 1 - rows_from_south, columns]

    domain.set_name("varuna_live_anuga")
    domain.set_store(False)
    domain.set_quantity("elevation", function=elevation_at)
    domain.set_quantity("stage", expression="elevation")
    domain.set_quantity("friction", MANNING_N)
    reflective = anuga.Reflective_boundary(domain)
    domain.set_boundary(
        {"left": reflective, "right": reflective, "top": reflective, "bottom": reflective}
    )

    rain_duration_s = duration_min * 60.0
    forecast_duration_s = (forecast_duration_min or duration_min) * 60.0
    rate: Callable[[float], float] = lambda time_s: _rainfall_rate(
        pattern, rainfall_intensity_mm_hr, rain_duration_s, time_s
    )
    anuga.Rate_operator(domain, rate=rate)

    times: list[float] = []
    depths: list[np.ndarray] = []
    velocities_x: list[np.ndarray] = []
    velocities_y: list[np.ndarray] = []
    centroids = domain.get_centroid_coordinates(absolute=True)

    for model_time in domain.evolve(
        yieldstep=OUTPUT_STEP_S,
        finaltime=forecast_duration_s,
    ):
        elevation = domain.quantities["elevation"].centroid_values
        stage = domain.quantities["stage"].centroid_values.copy()
        depth = np.maximum(stage - elevation, 0.0)
        xmomentum = domain.quantities["xmomentum"].centroid_values.copy()
        ymomentum = domain.quantities["ymomentum"].centroid_values.copy()
        times.append(float(model_time) / 60.0)
        depths.append(depth)
        velocities_x.append(
            np.divide(xmomentum, depth, out=np.zeros_like(depth), where=depth > 1e-6)
        )
        velocities_y.append(
            np.divide(ymomentum, depth, out=np.zeros_like(depth), where=depth > 1e-6)
        )

    columns = np.floor((centroids[:, 0] - xll) / CELL_SIZE_M).astype(np.int64)
    rows_from_south = np.floor((centroids[:, 1] - yll) / CELL_SIZE_M).astype(np.int64)
    rows = GRID_SIZE - 1 - rows_from_south
    cell_ids = rows * GRID_SIZE + columns
    counts = np.bincount(cell_ids, minlength=GRID_SIZE * GRID_SIZE)
    if not np.all(counts == 4):
        raise RuntimeError("Live ANUGA triangles cannot be mapped to the DEM grid.")

    def rasterize(values: list[np.ndarray]) -> np.ndarray:
        return np.stack(
            [
                np.bincount(cell_ids, weights=frame, minlength=GRID_SIZE * GRID_SIZE)
                / counts
                for frame in values
            ]
        )

    return (
        np.asarray(times),
        rasterize(depths),
        rasterize(velocities_x),
        rasterize(velocities_y),
    )

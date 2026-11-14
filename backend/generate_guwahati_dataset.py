#!/usr/bin/env python3
"""Build a Guwahati ANUGA dataset from public SRTM and NASA POWER data."""

from __future__ import annotations

from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import urllib.request

import anuga
import numpy as np
from pyproj import Transformer


BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
OUTPUT_DIR = BACKEND_DIR / "data" / "processed" / "guwahati"
MODEL_DIR = REPO_DIR / "model"

LOCATION = "Guwahati, Assam, India"
LATITUDE = 26.1445
LONGITUDE = 91.7362
CRS = "EPSG:32646"
GRID_CELLS = 50
CELL_SIZE_M = 30.0
CELL_AREA_M2 = CELL_SIZE_M**2
OUTPUT_STEP_S = 60.0
SPINUP_HOURS = 6
FORECAST_HOURS = 1
FORECAST_STEPS = int(FORECAST_HOURS * 3600 / OUTPUT_STEP_S)
WET_THRESHOLD_M = 0.01
PARAMETER_COMBINATIONS = ((0.03, 0.0), (0.05, 5.0), (0.08, 10.0))
EVENTS_PER_YEAR = {2020: 4, 2021: 4, 2022: 4, 2023: 6, 2024: 6}
SPLIT_BY_YEAR = {
    2020: "train",
    2021: "train",
    2022: "train",
    2023: "validation",
    2024: "test",
}

TILE_NAME = "N26E091.hgt.gz"
TILE_PATH = RAW_DIR / TILE_NAME
TILE_URL = (
    "https://elevation-tiles-prod.s3.amazonaws.com/skadi/N26/N26E091.hgt.gz"
)
POWER_CACHE = RAW_DIR / "guwahati_power_precip_2020_2024.json"
POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point?"
    "parameters=PRECTOTCORR&community=AG&longitude=91.7362&latitude=26.1445"
    "&start=20200101&end=20241231&format=JSON"
)


def load_dem() -> tuple[np.ndarray, float, float]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not TILE_PATH.exists():
        print(f"Downloading public SRTM tile: {TILE_URL}")
        urllib.request.urlretrieve(TILE_URL, TILE_PATH)

    try:
        with gzip.open(TILE_PATH, "rb") as tile_file:
            tile_bytes = tile_file.read()
    except (OSError, EOFError) as error:
        raise RuntimeError(f"Unable to read SRTM tile {TILE_PATH}") from error

    expected_bytes = 3601 * 3601 * np.dtype(">i2").itemsize
    if len(tile_bytes) != expected_bytes:
        raise ValueError(
            f"Unexpected SRTM tile size: expected {expected_bytes}, "
            f"received {len(tile_bytes)}"
        )
    source = np.frombuffer(tile_bytes, dtype=">i2").reshape((3601, 3601))

    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    center_x, center_y = to_utm.transform(LONGITUDE, LATITUDE)
    width_m = GRID_CELLS * CELL_SIZE_M
    xll = center_x - width_m / 2.0
    yll = center_y - width_m / 2.0

    eastings = xll + (np.arange(GRID_CELLS) + 0.5) * CELL_SIZE_M
    northings = yll + (np.arange(GRID_CELLS)[::-1] + 0.5) * CELL_SIZE_M
    x_grid, y_grid = np.meshgrid(eastings, northings)
    to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    longitudes, latitudes = to_wgs84.transform(x_grid, y_grid)

    columns = np.rint((longitudes - 91.0) * 3600.0).astype(np.int64)
    rows = np.rint((27.0 - latitudes) * 3600.0).astype(np.int64)
    if (
        columns.min() < 0
        or columns.max() > 3600
        or rows.min() < 0
        or rows.max() > 3600
    ):
        raise ValueError("Selected simulation grid falls outside the SRTM tile")

    dem = source[rows, columns].astype(np.float64)
    if np.any(dem == -32768) or not np.all(np.isfinite(dem)):
        raise ValueError("SRTM tile contains void or non-finite elevations")
    return dem, xll, yll


def load_hourly_rainfall() -> tuple[dict[datetime, float], dict[str, object]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if POWER_CACHE.exists():
        payload = json.loads(POWER_CACHE.read_text(encoding="utf-8"))
    else:
        request = urllib.request.Request(
            POWER_URL,
            headers={"User-Agent": "GuwahatiANUGADataset/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        POWER_CACHE.write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    parameter = payload["properties"]["parameter"]["PRECTOTCORR"]
    units = payload["parameters"]["PRECTOTCORR"]["units"]
    if units != "mm/hour":
        raise ValueError(f"Unexpected NASA POWER precipitation units: {units}")
    rainfall: dict[datetime, float] = {}
    for timestamp, value in parameter.items():
        numeric_value = float(value)
        if numeric_value < 0:
            continue
        rainfall[datetime.strptime(timestamp, "%Y%m%d%H")] = numeric_value
    if not rainfall:
        raise ValueError("NASA POWER returned no valid precipitation records")
    return rainfall, payload["header"]


def select_event_times(
    rainfall: dict[datetime, float],
) -> list[tuple[datetime, str]]:
    selected: list[tuple[datetime, str]] = []
    for year, count in EVENTS_PER_YEAR.items():
        candidates: list[tuple[float, datetime]] = []
        for timestamp, current_rain in rainfall.items():
            if timestamp.year != year or timestamp.month not in (6, 7, 8, 9):
                continue
            prior_times = [
                timestamp - timedelta(hours=offset)
                for offset in range(SPINUP_HOURS, 0, -1)
            ]
            if any(time not in rainfall for time in prior_times):
                continue
            total = sum(rainfall[time] for time in prior_times) + current_rain
            if total > 0:
                candidates.append((total, timestamp))
        chosen: list[datetime] = []
        for _, timestamp in sorted(candidates, reverse=True):
            if all(
                abs(timestamp - existing) >= timedelta(hours=24)
                for existing in chosen
            ):
                chosen.append(timestamp)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(
                f"Found only {len(chosen)} separated rainy windows for {year}; "
                f"needed {count}"
            )
        selected.extend(
            (timestamp, SPLIT_BY_YEAR[year])
            for timestamp in sorted(chosen)
        )
    return selected


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
    hourly_rainfall_mm_hr: list[float],
    manning_n: float,
    infiltration_loss_mm_hr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spinup_seconds = SPINUP_HOURS * 3600.0
    forecast_seconds = FORECAST_HOURS * 3600.0
    forecast_steps = FORECAST_STEPS
    domain = anuga.rectangular_cross_domain(
        GRID_CELLS,
        GRID_CELLS,
        len1=GRID_CELLS * CELL_SIZE_M,
        len2=GRID_CELLS * CELL_SIZE_M,
        origin=(xll, yll),
    )
    domain.set_store(False)

    def cell_indices(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        columns = np.floor((x - xll) / CELL_SIZE_M).astype(np.int64)
        rows_from_south = np.floor((y - yll) / CELL_SIZE_M).astype(np.int64)
        columns = np.clip(columns, 0, GRID_CELLS - 1)
        rows_from_south = np.clip(rows_from_south, 0, GRID_CELLS - 1)
        return GRID_CELLS - 1 - rows_from_south, columns

    def elevation_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows, columns = cell_indices(x, y)
        return dem[rows, columns]

    domain.set_quantity("elevation", function=elevation_at)
    domain.set_quantity("stage", function=elevation_at)
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

    def rainfall_rate_m_s(
        _x: np.ndarray,
        _y: np.ndarray,
        time_s: float,
    ) -> np.ndarray:
        hour_index = min(
            int(max(time_s, 0.0) // 3600.0),
            len(hourly_rainfall_mm_hr) - 1,
        )
        net_rain_mm_hr = max(
            hourly_rainfall_mm_hr[hour_index] - infiltration_loss_mm_hr,
            0.0,
        )
        return np.full_like(_x, net_rain_mm_hr / 3_600_000.0)

    rainfall_operator = anuga.Rate_operator(domain, rate=rainfall_rate_m_s)
    centroids = domain.get_centroid_coordinates(absolute=True)
    rows, columns = cell_indices(centroids[:, 0], centroids[:, 1])
    cell_ids = rows * GRID_CELLS + columns
    counts = np.bincount(cell_ids, minlength=GRID_CELLS**2)
    if not np.all(counts == 4):
        raise ValueError("Could not map ANUGA triangle centroids to raster cells")

    def capture_state() -> np.ndarray:
        elevation = domain.quantities["elevation"].centroid_values
        stage = domain.quantities["stage"].centroid_values
        depth = np.maximum(stage - elevation, 0.0)
        xmomentum = domain.quantities["xmomentum"].centroid_values
        ymomentum = domain.quantities["ymomentum"].centroid_values
        return np.stack(
            (
                rasterize(depth, cell_ids),
                rasterize(xmomentum, cell_ids),
                rasterize(ymomentum, cell_ids),
            )
        ).astype(np.float32)

    for _ in domain.evolve(yieldstep=600.0, finaltime=spinup_seconds):
        pass

    states = np.empty(
        (forecast_steps + 1, 3, GRID_CELLS, GRID_CELLS),
        dtype=np.float32,
    )
    raw_rainfall = np.empty(
        (forecast_steps + 1, GRID_CELLS, GRID_CELLS),
        dtype=np.float32,
    )
    effective_rainfall = np.empty_like(raw_rainfall)
    mass_errors = np.empty(forecast_steps + 1, dtype=np.float32)
    initial_state = capture_state()
    initial_volume_m3 = float(initial_state[0].sum() * CELL_AREA_M2)
    states[0] = initial_state

    forecast_rain_mm_hr = hourly_rainfall_mm_hr[-1]
    effective_rate_m_s = max(
        forecast_rain_mm_hr - infiltration_loss_mm_hr,
        0.0,
    ) / 3_600_000.0
    raw_rainfall[:] = forecast_rain_mm_hr
    effective_rainfall[:] = effective_rate_m_s
    mass_errors[0] = 0.0
    for frame_index, _ in enumerate(
        domain.evolve(
            yieldstep=OUTPUT_STEP_S,
            finaltime=spinup_seconds + forecast_seconds,
        ),
        start=1,
    ):
        if frame_index > forecast_steps:
            raise RuntimeError("ANUGA emitted more frames than the forecast length")
        states[frame_index] = capture_state()
        elapsed_seconds = frame_index * OUTPUT_STEP_S
        expected_influx_m3 = (
            effective_rate_m_s
            * elapsed_seconds
            * GRID_CELLS**2
            * CELL_AREA_M2
        )
        modeled_volume_m3 = float(states[frame_index, 0].sum() * CELL_AREA_M2)
        mass_errors[frame_index] = abs(
            modeled_volume_m3 - initial_volume_m3 - expected_influx_m3
        ) / max(initial_volume_m3 + expected_influx_m3, 1.0)
    if frame_index != forecast_steps:
        raise RuntimeError(
            f"Expected {forecast_steps} forecast frames, got {frame_index}"
        )
    return states, raw_rainfall, effective_rainfall, mass_errors



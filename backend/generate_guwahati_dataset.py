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



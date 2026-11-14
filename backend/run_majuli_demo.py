#!/usr/bin/env python3
"""Run an ANUGA rainfall-flood demo near Garamur, Majuli, and export Cesium data."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import urllib.request

import anuga
import numpy as np
from pyproj import Transformer


BACKEND_DIR = Path(__file__).resolve().parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
OUTPUT_DIR = BACKEND_DIR / "data" / "processed" / "majuli_demo"
FRONTEND_DATA_DIR = BACKEND_DIR.parent / "frontend" / "data"

LOCATION = "Garamur, Majuli, Assam, India"
LATITUDE = 26.980306
LONGITUDE = 94.1574452
CRS = "EPSG:32646"
CELL_SIZE_M = 30.0
GRID_CELLS = 50
RAINFALL_MM_HR = 50.0
INFILTRATION_MM_HR = 5.0
STORM_DURATION_S = 3600.0
OUTPUT_STEP_S = 60.0
MANNING_N = 0.05

TILE_PATH = RAW_DIR / "N26E094.hgt.gz"
TILE_URL = (
    "https://elevation-tiles-prod.s3.amazonaws.com/skadi/N26/N26E094.hgt.gz"
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

    columns = np.rint((longitudes - 94.0) * 3600.0).astype(np.int64)
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


def write_dem(dem: np.ndarray, xll: float, yll: float) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dem_path = OUTPUT_DIR / "majuli_dem.asc"
    with dem_path.open("w", encoding="ascii") as dem_file:
        dem_file.write(
            f"ncols {GRID_CELLS}\n"
            f"nrows {GRID_CELLS}\n"
            f"xllcorner {xll:.3f}\n"
            f"yllcorner {yll:.3f}\n"
            f"cellsize {CELL_SIZE_M:.1f}\n"
            "NODATA_value -9999\n"
        )
        np.savetxt(dem_file, dem, fmt="%.1f")
    return dem_path


def run_anuga(dem: np.ndarray, xll: float, yll: float) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    domain = anuga.rectangular_cross_domain(
        GRID_CELLS,
        GRID_CELLS,
        len1=GRID_CELLS * CELL_SIZE_M,
        len2=GRID_CELLS * CELL_SIZE_M,
        origin=(xll, yll),
    )

    def elevation_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        columns = np.floor((x - xll) / CELL_SIZE_M).astype(np.int64)
        rows_from_south = np.floor((y - yll) / CELL_SIZE_M).astype(np.int64)
        columns = np.clip(columns, 0, GRID_CELLS - 1)
        rows_from_south = np.clip(rows_from_south, 0, GRID_CELLS - 1)
        return dem[GRID_CELLS - 1 - rows_from_south, columns]

    domain.set_name("majuli_anuga")
    domain.set_datadir(str(OUTPUT_DIR / "anuga"))
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

    effective_rainfall_mm_hr = RAINFALL_MM_HR - INFILTRATION_MM_HR
    anuga.Rate_operator(domain, rate=effective_rainfall_mm_hr / 3_600_000.0)

    centroids = domain.get_centroid_coordinates(absolute=True)
    times: list[float] = []
    depths: list[np.ndarray] = []
    xmomenta: list[np.ndarray] = []
    ymomenta: list[np.ndarray] = []

    for model_time in domain.evolve(
        yieldstep=OUTPUT_STEP_S,
        finaltime=STORM_DURATION_S,
    ):
        stage = domain.quantities["stage"].centroid_values.copy()
        depth = np.maximum(
            stage - domain.quantities["elevation"].centroid_values,
            0.0,
        )
        times.append(float(model_time))
        depths.append(depth)
        xmomenta.append(domain.quantities["xmomentum"].centroid_values.copy())
        ymomenta.append(domain.quantities["ymomentum"].centroid_values.copy())

    depth_series = np.stack(depths)
    xmomentum_series = np.stack(xmomenta)
    ymomentum_series = np.stack(ymomenta)
    velocity_x = np.divide(
        xmomentum_series,
        depth_series,
        out=np.zeros_like(depth_series),
        where=depth_series > 1e-6,
    )
    velocity_y = np.divide(
        ymomentum_series,
        depth_series,
        out=np.zeros_like(depth_series),
        where=depth_series > 1e-6,
    )
    np.savez_compressed(
        OUTPUT_DIR / "anuga_training_data.npz",
        time_seconds=np.asarray(times),
        x_m=centroids[:, 0],
        y_m=centroids[:, 1],
        depth_m=depth_series,
        xmomentum_m2_s=xmomentum_series,
        ymomentum_m2_s=ymomentum_series,
        velocity_x_m_s=velocity_x,
        velocity_y_m_s=velocity_y,
        velocity_m_s=np.hypot(velocity_x, velocity_y),
    )

    final_volume_m3 = float(np.sum(depth_series[-1] * domain.areas))
    expected_volume_m3 = (
        effective_rainfall_mm_hr
        / 1000.0
        / 3600.0
        * STORM_DURATION_S
        * (GRID_CELLS * CELL_SIZE_M) ** 2
    )
    relative_mass_error = abs(final_volume_m3 - expected_volume_m3) / max(
        expected_volume_m3,
        1e-9,
    )
    if relative_mass_error > 0.01:
        raise RuntimeError(
            f"ANUGA water-volume error {relative_mass_error:.2%} exceeds 1%"
        )
    print(
        f"ANUGA: {len(times)} frames, {len(centroids)} triangles, "
        f"final volume {final_volume_m3:.1f} m3, "
        f"mass error {relative_mass_error:.2%}"
    )
    return np.asarray(times), depth_series, velocity_x, velocity_y


def export_cesium(
    dem: np.ndarray,
    xll: float,
    yll: float,
    times: np.ndarray,
    triangle_depth: np.ndarray,
    triangle_vx: np.ndarray,
    triangle_vy: np.ndarray,
) -> None:
    training = np.load(OUTPUT_DIR / "anuga_training_data.npz")
    x = training["x_m"]
    y = training["y_m"]
    columns = np.floor((x - xll) / CELL_SIZE_M).astype(np.int64)
    rows_from_south = np.floor((y - yll) / CELL_SIZE_M).astype(np.int64)
    rows = GRID_CELLS - 1 - rows_from_south
    cell_ids = rows * GRID_CELLS + columns
    cell_counts = np.bincount(cell_ids, minlength=GRID_CELLS**2)
    if not np.all(cell_counts == 4):
        raise ValueError("Could not map ANUGA triangle centroids to raster cells")

    raster_depth = np.empty((len(times), GRID_CELLS**2), dtype=np.float32)
    raster_vx = np.empty_like(raster_depth)
    raster_vy = np.empty_like(raster_depth)
    for frame in range(len(times)):
        for values, target in (
            (triangle_depth[frame], raster_depth),
            (triangle_vx[frame], raster_vx),
            (triangle_vy[frame], raster_vy),
        ):
            target[frame] = (
                np.bincount(
                    cell_ids,
                    weights=values,
                    minlength=GRID_CELLS**2,
                )
                / cell_counts
            )

    to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    cells: list[dict[str, object]] = []
    for row in range(GRID_CELLS):
        y_top = yll + (GRID_CELLS - row) * CELL_SIZE_M
        y_bottom = y_top - CELL_SIZE_M
        for column in range(GRID_CELLS):
            x_left = xll + column * CELL_SIZE_M
            x_right = x_left + CELL_SIZE_M
            corner_xy = [
                (x_left, y_bottom),
                (x_right, y_bottom),
                (x_right, y_top),
                (x_left, y_top),
            ]
            corners = [
                list(to_wgs84.transform(easting, northing))
                for easting, northing in corner_xy
            ]
            center = to_wgs84.transform(
                x_left + CELL_SIZE_M / 2.0,
                y_bottom + CELL_SIZE_M / 2.0,
            )
            cells.append(
                {
                    "corners_lonlat": corners,
                    "center_lonlat": list(center),
                    "elevation_m": float(dem[row, column]),
                }
            )

    frames = [
        {
            "time_s": float(times[index]),
            "depth_m": np.round(raster_depth[index], 5).tolist(),
            "vx_m_s": np.round(raster_vx[index], 5).tolist(),
            "vy_m_s": np.round(raster_vy[index], 5).tolist(),
        }
        for index in range(len(times))
    ]
    payload = {
        "schema_version": 1,
        "location": LOCATION,
        "center": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "source": "NASA/USGS SRTM 1 arc-second HGT via AWS Terrain Tiles",
        "model": "ANUGA 2D shallow-water model; finite-volume solver",
        "grid": {
            "rows": GRID_CELLS,
            "columns": GRID_CELLS,
            "cell_size_m": CELL_SIZE_M,
            "crs": CRS,
            "xll_m": xll,
            "yll_m": yll,
            "center_x_m": xll + GRID_CELLS * CELL_SIZE_M / 2.0,
            "center_y_m": yll + GRID_CELLS * CELL_SIZE_M / 2.0,
            "rainfall_mm_hr": RAINFALL_MM_HR,
            "infiltration_mm_hr": INFILTRATION_MM_HR,
            "effective_rainfall_mm_hr": (
                RAINFALL_MM_HR - INFILTRATION_MM_HR
            ),
            "storm_duration_s": STORM_DURATION_S,
            "output_interval_s": OUTPUT_STEP_S,
            "manning_n": MANNING_N,
            "boundary": "reflective on all sides",
        },
        "visualization": {
            "terrain_vertical_exaggeration": 5,
            "water_depth_exaggeration": 12,
        },
        "limitations": [
            "Synthetic uniform rainfall and constant infiltration; not observed or calibrated.",
            "SRTM at about 30 m does not resolve buildings, drains, or small channels.",
            "The 1.5 km square is a demonstration domain, not a delineated catchment.",
            "This is a numerical demonstration, not a flood warning or validated forecast.",
        ],
        "cells": cells,
        "frames": frames,
    }

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FRONTEND_DATA_DIR / "majuli_simulation.json"
    output_path.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata = {key: value for key, value in payload.items() if key != "frames"}
    metadata.pop("cells")
    metadata["frame_count"] = len(frames)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Cesium simulation data: {output_path}")


def main() -> None:
    dem, xll, yll = load_dem()
    write_dem(dem, xll, yll)
    times, depth, velocity_x, velocity_y = run_anuga(dem, xll, yll)
    export_cesium(dem, xll, yll, times, depth, velocity_x, velocity_y)


if __name__ == "__main__":
    main()

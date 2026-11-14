#!/usr/bin/env python3
"""Prepare open SRTM terrain and run ANUGA plus LISFLOOD-FP for Sotai."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import urllib.request

import anuga
import numpy as np
from pyproj import Transformer


BACKEND_DIR = Path(__file__).resolve().parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
OUTPUT_DIR = BACKEND_DIR / "data" / "processed" / "sotai_demo"
TILE_PATH = RAW_DIR / "N28E077.hgt.gz"
TILE_URL = (
    "https://elevation-tiles-prod.s3.amazonaws.com/skadi/N28/N28E077.hgt.gz"
)

VILLAGE = "Sotai, Ballabgarh, Faridabad, Haryana, India"
LATITUDE = 28.2927976
LONGITUDE = 77.3594597
CRS = "EPSG:32643"
CELL_SIZE_M = 30.0
GRID_CELLS = 50
RAINFALL_MM_HR = 50.0
INFILTRATION_MM_HR = 5.0
STORM_DURATION_S = 3600.0
OUTPUT_STEP_S = 300.0
MANNING_N = 0.05

LISFLOOD_DIR = (
    BACKEND_DIR / "simulators" / "lisflood-fp-bmi" / "lisflood-fp-bmi-v5.9"
)


def download_tile() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not TILE_PATH.exists():
        print(f"Downloading SRTM tile: {TILE_URL}")
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


def make_dem() -> tuple[Path, np.ndarray, float, float]:
    download_tile()
    with gzip.open(TILE_PATH, "rb") as tile_file:
        source = np.frombuffer(tile_file.read(), dtype=">i2").reshape((3601, 3601))

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

    columns = np.rint((longitudes - 77.0) * 3600.0).astype(np.int64)
    rows = np.rint((29.0 - latitudes) * 3600.0).astype(np.int64)
    if (
        columns.min() < 0
        or columns.max() > 3600
        or rows.min() < 0
        or rows.max() > 3600
    ):
        raise ValueError("Selected simulation grid falls outside the SRTM tile")

    elevation = source[rows, columns].astype(np.float64)
    if np.any(elevation == -32768):
        raise ValueError("SRTM tile contains voids inside the selected grid")
    if not np.all(np.isfinite(elevation)):
        raise ValueError("SRTM elevation grid contains non-finite values")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dem_path = OUTPUT_DIR / "sotai_dem.asc"
    with dem_path.open("w", encoding="ascii") as dem_file:
        dem_file.write(
            f"ncols {GRID_CELLS}\n"
            f"nrows {GRID_CELLS}\n"
            f"xllcorner {xll:.3f}\n"
            f"yllcorner {yll:.3f}\n"
            f"cellsize {CELL_SIZE_M:.1f}\n"
            "NODATA_value -9999\n"
        )
        np.savetxt(dem_file, elevation, fmt="%.1f")

    metadata = {
        "location": VILLAGE,
        "center": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "crs": CRS,
        "grid": {
            "rows": GRID_CELLS,
            "columns": GRID_CELLS,
            "cell_size_m": CELL_SIZE_M,
            "xllcorner": xll,
            "yllcorner": yll,
        },
        "elevation_source": TILE_URL,
        "elevation_product": "SRTM 1 arc-second HGT, approximately 30 m",
        "rainfall": {
            "intensity_mm_per_hour": RAINFALL_MM_HR,
            "duration_seconds": STORM_DURATION_S,
            "type": "synthetic uniform demo storm; not observed rainfall",
        },
        "infiltration_mm_per_hour": INFILTRATION_MM_HR,
        "manning_n": MANNING_N,
        "limitations": [
            "SRTM terrain is too coarse to represent individual buildings or drains.",
            "The uniform storm and constant infiltration are illustrative, not calibrated.",
            "The square domain is not a delineated watershed.",
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return dem_path, elevation, xll, yll


def read_dem(dem_path: Path) -> np.ndarray:
    return np.loadtxt(dem_path, skiprows=6, dtype=np.float64)


def run_anuga(dem_path: Path, xll: float, yll: float) -> None:
    dem = read_dem(dem_path)
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

    domain.set_name("sotai_anuga")
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
    anuga.Rate_operator(
        domain,
        rate=effective_rainfall_mm_hr / 3_600_000.0,
    )

    times: list[float] = []
    depths: list[np.ndarray] = []
    x_momenta: list[np.ndarray] = []
    y_momenta: list[np.ndarray] = []
    centroids = domain.get_centroid_coordinates(absolute=True)
    centroids_elevation = elevation_at(centroids[:, 0], centroids[:, 1])

    for model_time in domain.evolve(
        yieldstep=OUTPUT_STEP_S,
        finaltime=STORM_DURATION_S,
    ):
        stage = domain.quantities["stage"].centroid_values.copy()
        depth = np.maximum(
            stage - domain.quantities["elevation"].centroid_values,
            0.0,
        )
        xmomentum = domain.quantities["xmomentum"].centroid_values.copy()
        ymomentum = domain.quantities["ymomentum"].centroid_values.copy()
        velocity_x = np.divide(
            xmomentum,
            depth,
            out=np.zeros_like(depth),
            where=depth > 1e-6,
        )
        velocity_y = np.divide(
            ymomentum,
            depth,
            out=np.zeros_like(depth),
            where=depth > 1e-6,
        )
        times.append(float(model_time))
        depths.append(depth)
        x_momenta.append(xmomentum)
        y_momenta.append(ymomentum)

    depth_series = np.stack(depths)
    xmomentum_series = np.stack(x_momenta)
    ymomentum_series = np.stack(y_momenta)
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
        elevation_m=centroids_elevation,
        depth_m=depth_series,
        xmomentum_m2_s=xmomentum_series,
        ymomentum_m2_s=ymomentum_series,
        velocity_x_m_s=velocity_x,
        velocity_y_m_s=velocity_y,
        velocity_m_s=np.hypot(velocity_x, velocity_y),
        flow_direction_rad=np.arctan2(velocity_y, velocity_x),
    )

    final_volume_m3 = float(np.sum(depth_series[-1] * domain.areas))
    expected_runoff_m3 = (
        effective_rainfall_mm_hr
        / 1000.0
        / 3600.0
        * STORM_DURATION_S
        * (GRID_CELLS * CELL_SIZE_M) ** 2
    )
    relative_mass_error = abs(final_volume_m3 - expected_runoff_m3) / max(
        expected_runoff_m3, 1e-9
    )
    if relative_mass_error > 0.01:
        raise RuntimeError(
            f"ANUGA water-volume error {relative_mass_error:.2%} exceeds 1%"
        )
    print(
        f"ANUGA: {len(times)} frames, {len(centroids)} triangles, "
        f"final volume {final_volume_m3:.1f} m3, "
        f"rainfall-only target {expected_runoff_m3:.1f} m3 "
        f"(relative difference {relative_mass_error:.2%})"
    )


def write_lisflood_inputs(dem_path: Path) -> Path:
    if not (LISFLOOD_DIR / "lisflood").is_file():
        raise FileNotFoundError(
            f"LISFLOOD executable missing. Build it in {LISFLOOD_DIR} "
            "using the commands in backend/README.md."
        )

    run_dir = OUTPUT_DIR / "lisflood"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / dem_path.name).write_bytes(dem_path.read_bytes())
    (run_dir / "sotai.rain").write_text(
        "# intensity_mm_per_hour time_hours\n"
        "2 hours\n"
        f"{RAINFALL_MM_HR:.1f} 0.0\n"
        f"{RAINFALL_MM_HR:.1f} {STORM_DURATION_S / 3600.0:.1f}\n",
        encoding="ascii",
    )
    infiltration_m_s = INFILTRATION_MM_HR / 3_600_000.0
    par_path = run_dir / "sotai.par"
    par_path.write_text(
        f"DEMfile {dem_path.name}\n"
        "resroot lisflood_sotai\n"
        f"sim_time {STORM_DURATION_S:.1f}\n"
        f"saveint {OUTPUT_STEP_S:.1f}\n"
        f"massint {OUTPUT_STEP_S:.1f}\n"
        f"fpfric {MANNING_N:.3f}\n"
        f"infiltration {infiltration_m_s:.12g}\n"
        "rainfall sotai.rain\n"
        "voutput\n"
        "qoutput\n"
        "elevoff\n",
        encoding="ascii",
    )
    return par_path


def run_lisflood(dem_path: Path) -> None:
    par_path = write_lisflood_inputs(dem_path)
    environment = os.environ.copy()
    existing_library_path = environment.get("DYLD_LIBRARY_PATH", "")
    environment["DYLD_LIBRARY_PATH"] = os.pathsep.join(
        part
        for part in (str(LISFLOOD_DIR), existing_library_path)
        if part
    )
    subprocess.run(
        [str(LISFLOOD_DIR / "lisflood"), "-v", par_path.name],
        cwd=par_path.parent,
        env=environment,
        check=True,
    )
    depth_outputs = sorted(par_path.parent.glob("lisflood_sotai-*.wd"))
    if not depth_outputs:
        raise RuntimeError("LISFLOOD completed without producing depth grids")
    mass_path = par_path.parent / "lisflood_sotai.mass"
    mass_rows = []
    for line in mass_path.read_text(encoding="ascii").splitlines():
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) == 12:
            mass_rows.append(values)
    if not mass_rows:
        raise RuntimeError("LISFLOOD did not produce readable mass-balance output")
    volume_error = mass_rows[-1][10]
    if not np.isfinite(volume_error) or abs(volume_error) > 1e-5:
        raise RuntimeError(
            f"LISFLOOD volume-error diagnostic {volume_error} exceeds tolerance"
        )
    print(
        f"LISFLOOD: produced {len(depth_outputs)} depth frames; "
        f"final volume-error diagnostic {volume_error:.2e}"
    )


def compare_models(xll: float, yll: float) -> None:
    anuga_data = np.load(OUTPUT_DIR / "anuga_training_data.npz")
    depth_paths = sorted((OUTPUT_DIR / "lisflood").glob("lisflood_sotai-*.wd"))
    times = np.arange(len(depth_paths), dtype=np.float64) * OUTPUT_STEP_S
    if not np.array_equal(anuga_data["time_seconds"], times):
        raise ValueError("ANUGA and LISFLOOD output times do not match")

    x = anuga_data["x_m"]
    y = anuga_data["y_m"]
    columns = np.floor((x - xll) / CELL_SIZE_M).astype(np.int64)
    rows_from_south = np.floor((y - yll) / CELL_SIZE_M).astype(np.int64)
    rows = GRID_CELLS - 1 - rows_from_south
    cell_ids = rows * GRID_CELLS + columns
    cell_counts = np.bincount(cell_ids, minlength=GRID_CELLS**2)
    if not np.all(cell_counts == 4):
        raise ValueError("Could not map ANUGA triangle centroids to raster cells")

    anuga_depth_rasters: list[np.ndarray] = []
    lisflood_depths: list[np.ndarray] = []
    lisflood_vx: list[np.ndarray] = []
    lisflood_vy: list[np.ndarray] = []
    for frame, depth_path in enumerate(depth_paths):
        depth = anuga_data["depth_m"][frame]
        raster_sum = np.bincount(
            cell_ids,
            weights=depth,
            minlength=GRID_CELLS**2,
        )
        anuga_depth_rasters.append(
            (raster_sum / cell_counts).reshape(GRID_CELLS, GRID_CELLS)
        )

        model_depth = np.loadtxt(depth_path, skiprows=6, dtype=np.float64)
        model_vx = np.loadtxt(
            depth_path.with_suffix(".Vx"),
            skiprows=6,
            dtype=np.float64,
        )
        model_vy = np.loadtxt(
            depth_path.with_suffix(".Vy"),
            skiprows=6,
            dtype=np.float64,
        )
        if model_depth.shape != (GRID_CELLS, GRID_CELLS):
            raise ValueError(f"Unexpected LISFLOOD raster shape at {depth_path}")
        if model_vx.shape != (GRID_CELLS, GRID_CELLS + 1):
            raise ValueError(f"Unexpected LISFLOOD x-velocity shape at {depth_path}")
        if model_vy.shape != (GRID_CELLS + 1, GRID_CELLS):
            raise ValueError(f"Unexpected LISFLOOD y-velocity shape at {depth_path}")
        lisflood_depths.append(model_depth)
        lisflood_vx.append(0.5 * (model_vx[:, :-1] + model_vx[:, 1:]))
        lisflood_vy.append(0.5 * (model_vy[:-1, :] + model_vy[1:, :]))

    depth_stack = np.stack(lisflood_depths)
    vx_stack = np.stack(lisflood_vx)
    vy_stack = np.stack(lisflood_vy)
    velocity_x = np.where(depth_stack > 1e-6, vx_stack, 0.0)
    velocity_y = np.where(depth_stack > 1e-6, vy_stack, 0.0)
    np.savez_compressed(
        OUTPUT_DIR / "lisflood_testing_data.npz",
        time_seconds=times,
        depth_m=depth_stack,
        velocity_x_m_s=velocity_x,
        velocity_y_m_s=velocity_y,
        velocity_m_s=np.hypot(velocity_x, velocity_y),
        flow_direction_rad=np.arctan2(velocity_y, velocity_x),
        xmomentum_m2_s=depth_stack * velocity_x,
        ymomentum_m2_s=depth_stack * velocity_y,
    )

    frame_metrics = []
    for model_time, anuga_depth, lisflood_depth in zip(
        times,
        anuga_depth_rasters,
        depth_stack,
        strict=True,
    ):
        difference = anuga_depth - lisflood_depth
        frame_metrics.append(
            {
                "time_seconds": float(model_time),
                "depth_rmse_m": float(np.sqrt(np.mean(difference**2))),
                "depth_mae_m": float(np.mean(np.abs(difference))),
            }
        )
    metrics = {
        "comparison": (
            "ANUGA triangle-centroid depths averaged to raster cells versus "
            "LISFLOOD-FP depth rasters"
        ),
        "warning": "Diagnostic comparison only; this demo is not calibrated.",
        "frames": frame_metrics,
    }
    (OUTPUT_DIR / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Benchmark: final-frame depth RMSE "
        f"{frame_metrics[-1]['depth_rmse_m']:.3f} m, "
        f"MAE {frame_metrics[-1]['depth_mae_m']:.3f} m"
    )


def main() -> None:
    dem_path, _, xll, yll = make_dem()
    run_anuga(dem_path, xll, yll)
    run_lisflood(dem_path)
    compare_models(xll, yll)
    print(f"Demo data and model outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

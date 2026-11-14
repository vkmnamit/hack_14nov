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


def main() -> None:
    dem, xll, yll = load_dem()
    rainfall, power_header = load_hourly_rainfall()
    events = select_event_times(rainfall)
    scenario_records: list[dict[str, object]] = []
    states_list: list[np.ndarray] = []
    raw_rain_list: list[np.ndarray] = []
    effective_rain_list: list[np.ndarray] = []
    mass_errors_list: list[np.ndarray] = []

    total_scenarios = len(events) * len(PARAMETER_COMBINATIONS)
    for event_time, split in events:
        spinup_times = [
            event_time - timedelta(hours=offset)
            for offset in range(SPINUP_HOURS, 0, -1)
        ]
        rainfall_profile = [
            rainfall[timestamp]
            for timestamp in [*spinup_times, event_time]
        ]
        for manning_n, infiltration_loss in PARAMETER_COMBINATIONS:
            scenario_id = len(scenario_records)
            print(
                f"Scenario {scenario_id + 1:02d}/{total_scenarios}: "
                f"{event_time.isoformat()} {split}, "
                f"rain={rainfall_profile[-1]:.2f} mm/hr, "
                f"n={manning_n:.2f}, infiltration={infiltration_loss:.0f} mm/hr"
            )
            states, raw_rain, effective_rain, mass_errors = run_scenario(
                dem,
                xll,
                yll,
                rainfall_profile,
                manning_n,
                infiltration_loss,
            )
            states_list.append(states)
            raw_rain_list.append(raw_rain)
            effective_rain_list.append(effective_rain)
            mass_errors_list.append(mass_errors)
            speed = np.divide(
                np.hypot(states[:, 1], states[:, 2]),
                states[:, 0],
                out=np.zeros_like(states[:, 0]),
                where=states[:, 0] >= WET_THRESHOLD_M,
            )
            scenario_records.append(
                {
                    "scenario_id": scenario_id,
                    "name": f"power_{event_time:%Y%m%d%H}_n{manning_n:.2f}",
                    "event_time_lST": event_time.strftime("%Y-%m-%d %H:%M"),
                    "temporal_profile": "NASA_POWER_hourly",
                    "spatial_pattern": "uniform_downscaled",
                    "rainfall_mm_hr_by_hour": rainfall_profile,
                    "manning_n": manning_n,
                    "infiltration_loss_mm_hr": infiltration_loss,
                    "initial_condition": f"{SPINUP_HOURS}h_rain_spinup_from_dry",
                    "boundary": "reflective on all sides",
                    "split": split,
                    "max_depth_m": float(states[:, 0].max()),
                    "max_speed_m_s_where_depth_ge_1cm": float(speed.max()),
                    "max_mass_balance_relative_error": float(mass_errors.max()),
                    "final_mass_balance_relative_error": float(mass_errors[-1]),
                    "final_inundated_cells_at_1cm": int(
                        np.count_nonzero(states[-1, 0] >= WET_THRESHOLD_M)
                    ),
                }
            )

    states_array = np.stack(states_list)
    raw_rain_array = np.stack(raw_rain_list)
    effective_rain_array = np.stack(effective_rain_list)
    mass_error_array = np.stack(mass_errors_list)
    splits = np.asarray(
        [record["split"] for record in scenario_records],
        dtype="U10",
    )
    train_ids = np.flatnonzero(splits == "train")
    train_states = states_array[train_ids]
    state_mean = train_states.mean(axis=(0, 1, 3, 4), dtype=np.float64)
    state_std = train_states.std(axis=(0, 1, 3, 4), dtype=np.float64)
    train_rain = raw_rain_array[train_ids]
    infiltration_maps = np.stack(
        [
            np.full(dem.shape, record["infiltration_loss_mm_hr"], dtype=np.float32)
            for record in scenario_records
        ]
    )
    manning_maps = np.stack(
        [
            np.full(dem.shape, record["manning_n"], dtype=np.float32)
            for record in scenario_records
        ]
    )
    train_infiltration = infiltration_maps[train_ids]
    train_manning = manning_maps[train_ids]

    metadata = {
        "location": LOCATION,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "crs": CRS,
        "grid_cells": GRID_CELLS,
        "cell_size_m": CELL_SIZE_M,
        "times_standard": power_header["time_standard"],
        "rainfall_source": "NASA POWER hourly PRECTOTCORR, MERRA2",
        "rainfall_units": "mm/hour",
        "terrain_source": "SRTM 1 arc-second, AWS Terrain Tiles, N26E091",
        "rainfall_assumption": (
            "One hourly NASA POWER point value is applied uniformly over the "
            "50x50 local grid; the hourly value is held constant during that hour."
        ),
        "spinup_assumption": (
            f"Each event starts dry and is spun up for {SPINUP_HOURS} hours "
            "using preceding hourly precipitation."
        ),
        "infiltration_assumption": (
            "Constant loss rates of 0, 5, and 10 mm/hour; not soil-calibrated."
        ),
        "manning_assumption": (
            "Uniform Manning n values of 0.03, 0.05, and 0.08; "
            "not land-cover-derived."
        ),
        "boundary_condition": "Reflective on all sides.",
        "split_by_year": {
            str(year): split for year, split in SPLIT_BY_YEAR.items()
        },
        "scenarios": scenario_records,
    }
    archive_path = OUTPUT_DIR / "guwahati_anuga_convlstm.npz"
    normalization_path = MODEL_DIR / "guwahati_normalization.npz"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    times = np.arange(FORECAST_STEPS + 1, dtype=np.float32) * OUTPUT_STEP_S
    np.savez_compressed(
        archive_path,
        state_sequence=np.asarray(states_array, dtype=np.float32),
        initial_state=np.asarray(states_array[:, 0], dtype=np.float32),
        raw_rainfall_mm_hr=np.asarray(raw_rain_array, dtype=np.float32),
        rainfall_effective_m_s=np.asarray(effective_rain_array, dtype=np.float32),
        infiltration_loss_mm_hr=infiltration_maps,
        elevation_m=np.asarray(dem, dtype=np.float32),
        manning_n=manning_maps,
        wet_mask_1cm=np.asarray(
            states_array[:, :, 0] >= WET_THRESHOLD_M,
            dtype=np.bool_,
        ),
        times_seconds=times,
        scenario_ids=np.arange(total_scenarios, dtype=np.int16),
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
            [
                [record["rainfall_mm_hr_by_hour"][-1]] * 6
                for record in scenario_records
            ],
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
        normalization_raw_rainfall_mean_mm_hr=np.float32(
            train_rain.mean(dtype=np.float64)
        ),
        normalization_raw_rainfall_std_mm_hr=np.float32(
            max(train_rain.std(dtype=np.float64), 1e-8)
        ),
        normalization_infiltration_mean_mm_hr=np.float32(
            train_infiltration.mean(dtype=np.float64)
        ),
        normalization_infiltration_std_mm_hr=np.float32(
            max(train_infiltration.std(dtype=np.float64), 1e-8)
        ),
        normalization_elevation_mean_m=np.float32(dem.mean(dtype=np.float64)),
        normalization_elevation_std_m=np.float32(
            max(dem.std(dtype=np.float64), 1e-8)
        ),
        normalization_manning_mean=np.float32(
            train_manning.mean(dtype=np.float64)
        ),
        normalization_manning_std=np.float32(
            max(train_manning.std(dtype=np.float64), 1e-8)
        ),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    np.savez_compressed(
        normalization_path,
        state_mean=np.asarray(state_mean, dtype=np.float32),
        state_std=np.maximum(state_std, 1e-8).astype(np.float32),
        rain_mean=np.float32(train_rain.mean(dtype=np.float64)),
        rain_std=np.float32(max(train_rain.std(dtype=np.float64), 1e-8)),
        infil_mean=np.float32(train_infiltration.mean(dtype=np.float64)),
        infil_std=np.float32(
            max(train_infiltration.std(dtype=np.float64), 1e-8)
        ),
        elev_mean=np.float32(dem.mean(dtype=np.float64)),
        elev_std=np.float32(max(dem.std(dtype=np.float64), 1e-8)),
        mann_mean=np.float32(train_manning.mean(dtype=np.float64)),
        mann_std=np.float32(max(train_manning.std(dtype=np.float64), 1e-8)),
    )
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ANUGA dataset: {archive_path}")
    print(f"Normalization: {normalization_path}")
    print(
        f"Scenarios: {len(train_ids)} train, "
        f"{np.count_nonzero(splits == 'validation')} validation, "
        f"{np.count_nonzero(splits == 'test')} test"
    )
    print(
        "DEM elevation range: "
        f"{dem.min():.1f} to {dem.max():.1f} m"
    )


if __name__ == "__main__":
    main()

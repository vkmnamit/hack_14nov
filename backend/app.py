"""HTTP API for serving the generated flood simulation to the frontend."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .live_anuga import run as run_live_anuga


BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKEND_DIR / "data" / "processed" / "majuli_demo"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
DATA_PATH = OUTPUT_DIR / "anuga_training_data.npz"
DEM_PATH = OUTPUT_DIR / "majuli_dem.asc"
LOCATION_ID = "majuli-garamur"
LOCATION_NAME = "Garamur, Majuli, Assam, India"


class SimulationRequest(BaseModel):
    locationId: str = Field(min_length=1)
    rainfallIntensityMmHr: float = Field(ge=0, le=500)
    durationMin: int = Field(gt=0, le=240)
    pattern: str


class SimulationMetadata(BaseModel):
    id: str
    locationId: str
    locationName: str
    createdAt: str
    rainfallIntensityMmHr: float
    durationMin: int
    pattern: str
    mode: str = "connected"


class FloodFrame(BaseModel):
    t: float
    rainfallMmHr: float
    depths: list[float]
    velocityX: list[float]
    velocityY: list[float]


class TerrainGrid(BaseModel):
    size: int
    elevation: list[float]


class SimulationFrames(BaseModel):
    simulationId: str
    frames: list[FloodFrame]
    terrain: TerrainGrid


SIMULATIONS: dict[str, SimulationMetadata] = {}
LIVE_RESULTS: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _read_ascii_grid(path: Path) -> np.ndarray:
    with path.open(encoding="ascii") as dem_file:
        for _ in range(6):
            next(dem_file)
        values = np.loadtxt(dem_file, dtype=np.float64)
    if values.shape != (50, 50):
        raise RuntimeError(f"Unexpected DEM shape: {values.shape}")
    return values


@lru_cache(maxsize=1)
def _load_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not DATA_PATH.exists() or not DEM_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Generated Majuli output is missing. Run "
            "`cd backend && .venv/bin/python run_majuli_demo.py` first."
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    grid = metadata["grid"]
    with np.load(DATA_PATH) as data:
        times = np.asarray(data["time_seconds"], dtype=np.float64) / 60.0
        depth = np.asarray(data["depth_m"], dtype=np.float64)
        velocity_x = np.asarray(data["velocity_x_m_s"], dtype=np.float64)
        velocity_y = np.asarray(data["velocity_y_m_s"], dtype=np.float64)
        x = np.asarray(data["x_m"], dtype=np.float64)
        y = np.asarray(data["y_m"], dtype=np.float64)

    columns = np.floor((x - grid["xll_m"]) / grid["cell_size_m"]).astype(np.int64)
    rows = 50 - 1 - np.floor((y - grid["yll_m"]) / grid["cell_size_m"]).astype(np.int64)
    cell_ids = rows * 50 + columns
    counts = np.bincount(cell_ids, minlength=2500)
    if not np.all(counts == 4):
        raise RuntimeError("Generated ANUGA triangles cannot be mapped to the DEM grid.")

    def rasterize(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                (np.bincount(cell_ids, weights=frame, minlength=2500) / counts)
                for frame in values
            ]
        )

    return times, rasterize(depth), rasterize(velocity_x), rasterize(velocity_y), _read_ascii_grid(DEM_PATH)


def _rainfall(pattern: str, intensity: float, duration: float, time: float) -> float:
    position = min(time / duration, 1.0)
    if pattern == "increasing":
        return intensity * position
    if pattern == "peak":
        return intensity * (position * 2 if position < 0.5 else 2 - position * 2)
    return intensity if position < 1 else 0.0


def _frames(meta: SimulationMetadata) -> list[FloodFrame]:
    reference_times, depths, velocity_x, velocity_y, _ = _load_reference()
    target_times = np.linspace(0, meta.durationMin, len(reference_times))
    reference_rainfall = 45.0
    intensity_scale = meta.rainfallIntensityMmHr / reference_rainfall
    result: list[FloodFrame] = []
    for index, target_time in enumerate(target_times):
        source_time = target_time * reference_times[-1] / max(meta.durationMin, 1)
        source_index = int(np.clip(np.searchsorted(reference_times, source_time), 0, len(reference_times) - 1))
        result.append(
            FloodFrame(
                t=float(target_time),
                rainfallMmHr=_rainfall(meta.pattern, meta.rainfallIntensityMmHr, meta.durationMin, float(target_time)),
                depths=np.maximum(depths[source_index] * intensity_scale, 0).round(6).tolist(),
                velocityX=(velocity_x[source_index] * np.sqrt(max(intensity_scale, 0))).round(6).tolist(),
                velocityY=(velocity_y[source_index] * np.sqrt(max(intensity_scale, 0))).round(6).tolist(),
            )
        )
    return result


def _live_anuga_enabled() -> bool:
    return os.getenv("VARUNA_USE_LIVE_ANUGA", "true").lower() not in {"0", "false", "no"}


def _anuga_installed() -> bool:
    try:
        import anuga  # noqa: F401
    except ImportError:
        return False
    return True


def _live_frames(meta: SimulationMetadata) -> list[FloodFrame]:
    cached = LIVE_RESULTS.get(meta.id)
    if cached is None:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        grid = metadata["grid"]
        _, _, _, _, elevation = _load_reference()
        times, depths, velocity_x, velocity_y = run_live_anuga(
            elevation,
            float(grid["xll_m"]),
            float(grid["yll_m"]),
            meta.rainfallIntensityMmHr,
            meta.durationMin,
            meta.pattern,
        )
        cached = (times, depths, velocity_x, velocity_y)
        LIVE_RESULTS[meta.id] = cached

    times, depths, velocity_x, velocity_y = cached
    return [
        FloodFrame(
            t=float(times[index]),
            rainfallMmHr=_rainfall(meta.pattern, meta.rainfallIntensityMmHr, meta.durationMin, float(times[index])),
            depths=np.maximum(depths[index], 0).round(6).tolist(),
            velocityX=velocity_x[index].round(6).tolist(),
            velocityY=velocity_y[index].round(6).tolist(),
        )
        for index in range(len(times))
    ]


app = FastAPI(title="VARUNA Flood Simulation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    try:
        _load_reference()
    except RuntimeError:
        return {"status": "degraded", "version": "1.0.0", "solver": "ANUGA precomputed output unavailable"}
    if _live_anuga_enabled() and _anuga_installed():
        return {"status": "ok", "version": "1.0.0", "solver": "ANUGA live raster output"}
    return {"status": "ok", "version": "1.0.0", "solver": "ANUGA precomputed raster fallback"}


@app.post("/api/simulate", response_model=SimulationMetadata)
def simulate(request: SimulationRequest) -> SimulationMetadata:
    if request.pattern not in {"constant", "increasing", "peak"}:
        raise HTTPException(status_code=422, detail="pattern must be constant, increasing, or peak")
    if request.locationId != LOCATION_ID:
        raise HTTPException(status_code=404, detail=f"Unknown location: {request.locationId}")
    try:
        _load_reference()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    simulation = SimulationMetadata(
        id=f"sim-{uuid4().hex[:12]}",
        locationId=LOCATION_ID,
        locationName=LOCATION_NAME,
        createdAt=datetime.now(timezone.utc).isoformat(),
        rainfallIntensityMmHr=request.rainfallIntensityMmHr,
        durationMin=request.durationMin,
        pattern=request.pattern,
    )
    SIMULATIONS[simulation.id] = simulation
    return simulation


@app.get("/api/simulation/{simulation_id}/frames", response_model=SimulationFrames)
def simulation_frames(simulation_id: str) -> SimulationFrames:
    meta = SIMULATIONS.get(simulation_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    try:
        _, _, _, _, elevation = _load_reference()
        if _live_anuga_enabled() and _anuga_installed():
            frames = _live_frames(meta)
        else:
            frames = _frames(meta)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return SimulationFrames(
        simulationId=simulation_id,
        frames=frames,
        terrain=TerrainGrid(size=50, elevation=elevation.ravel().round(3).tolist()),
    )

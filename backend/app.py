"""HTTP API for real ANUGA and city-trained ConvLSTM flood rollouts."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .model_service import (
        CITY_MODELS,
        CityModelConfig,
        ModelAssetsError,
        city_config,
        load_city_assets,
        rainfall_rate,
        run_city_convlstm,
    )
except ImportError:
    from model_service import (
        CITY_MODELS,
        CityModelConfig,
        ModelAssetsError,
        city_config,
        load_city_assets,
        rainfall_rate,
        run_city_convlstm,
    )


MODEL_NAMES = {
    "physics": "ANUGA shallow-water solver",
    "convlstm": "City-trained ConvLSTM",
    "adaptive": "ConvLSTM with adaptive SWE fallback",
}


class SimulationRequest(BaseModel):
    locationId: str = Field(min_length=1)
    modelType: Literal["physics", "convlstm", "adaptive"] = "adaptive"
    rainfallIntensityMmHr: float = Field(ge=0, le=500)
    durationMin: int = Field(gt=0, le=60)
    pattern: Literal["constant", "increasing", "peak"] = "constant"


class SimulationMetadata(BaseModel):
    id: str
    locationId: str
    locationName: str
    createdAt: str
    rainfallIntensityMmHr: float
    durationMin: int
    pattern: str
    modelType: Literal["physics", "convlstm", "adaptive"]
    mode: Literal["connected"] = "connected"


class FloodFrame(BaseModel):
    t: float
    rainfallMmHr: float
    depths: list[float]
    velocityX: list[float]
    velocityY: list[float]


class TerrainGrid(BaseModel):
    size: int
    elevation: list[float]


class SimulationDiagnostics(BaseModel):
    solver: str
    runtimeSeconds: float
    physicsFallbackSteps: int
    dryShortcutSteps: int


class SimulationFrames(BaseModel):
    simulationId: str
    frames: list[FloodFrame]
    terrain: TerrainGrid
    diagnostics: SimulationDiagnostics


app = FastAPI(title="VARUNA Flood Simulation API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

SIMULATIONS: dict[str, SimulationMetadata] = {}
RESULTS: dict[str, SimulationFrames] = {}


def _anuga_installed() -> bool:
    return find_spec("anuga") is not None


def _run_isolated_anuga(
    elevation: np.ndarray,
    config: CityModelConfig,
    request: SimulationMetadata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = {
        "dem": elevation.tolist(),
        "xll": config.xll_m,
        "yll": config.yll_m,
        "rainfall_intensity_mm_hr": request.rainfallIntensityMmHr,
        "duration_min": request.durationMin,
        "pattern": request.pattern,
        "forecast_duration_min": 60,
    }
    try:
        with TemporaryDirectory(prefix="varuna-anuga-") as temp_dir:
            output_path = Path(temp_dir) / "result.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.anuga_worker",
                    str(output_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                input=json.dumps(payload),
                capture_output=True,
                check=True,
                text=True,
                timeout=300,
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "ANUGA worker exited without details."
        raise RuntimeError(f"ANUGA worker failed: {detail}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("ANUGA worker exceeded the 5-minute time limit.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("ANUGA worker did not produce valid output.") from error
    return (
        np.asarray(result["times"], dtype=np.float64),
        np.asarray(result["depths"], dtype=np.float32),
        np.asarray(result["velocity_x"], dtype=np.float32),
        np.asarray(result["velocity_y"], dtype=np.float32),
    )


def _get_city_grid(location_id: str) -> tuple[np.ndarray, CityModelConfig]:
    try:
        elevation, _ = load_city_assets(location_id)
        config = city_config(location_id)
    except (ValueError, ModelAssetsError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return elevation, config


def _make_frames(
    request: SimulationMetadata,
    states: np.ndarray,
) -> list[FloodFrame]:
    frames: list[FloodFrame] = []
    for minute, state in enumerate(states):
        depth = np.maximum(state[0], 0.0)
        velocity_x = np.divide(
            state[1],
            depth,
            out=np.zeros_like(depth),
            where=depth >= 1e-6,
        )
        velocity_y = np.divide(
            state[2],
            depth,
            out=np.zeros_like(depth),
            where=depth >= 1e-6,
        )
        frames.append(
            FloodFrame(
                t=float(minute),
                rainfallMmHr=rainfall_rate(
                    request.pattern,
                    request.rainfallIntensityMmHr,
                    request.durationMin,
                    minute,
                ),
                depths=depth.ravel().round(6).tolist(),
                velocityX=velocity_x.ravel().round(6).tolist(),
                velocityY=velocity_y.ravel().round(6).tolist(),
            )
        )
    return frames


@app.get("/")
def index() -> dict[str, str]:
    return {
        "service": "VARUNA Flood Simulation API",
        "status": "online",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    solver = (
        "ANUGA + ConvLSTM modes available"
        if _anuga_installed()
        else "ConvLSTM modes available; ANUGA is not installed"
    )
    return {"status": "ok", "version": "2.0.0", "solver": solver}


@app.get("/api/locations")
def get_locations() -> list[dict[str, object]]:
    return [
        {
            "id": config.location_id,
            "name": config.name,
            "region": config.region,
            "longitude": config.longitude,
            "latitude": config.latitude,
            "cameraHeight": 2600,
            "seed": config.seed,
        }
        for config in CITY_MODELS.values()
    ]


@app.get("/api/models")
def get_models() -> list[dict[str, str]]:
    return [
        {"id": model_type, "name": name}
        for model_type, name in MODEL_NAMES.items()
    ]


@app.post("/api/simulate", response_model=SimulationMetadata)
def simulate(request: SimulationRequest) -> SimulationMetadata:
    try:
        config = city_config(request.locationId)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    if request.modelType == "physics" and not _anuga_installed():
        raise HTTPException(
            status_code=503,
            detail="ANUGA is required for the physics-only model. Install backend/requirements.txt.",
        )
    _get_city_grid(request.locationId)

    simulation = SimulationMetadata(
        id=f"sim-{uuid4().hex[:12]}",
        locationId=config.location_id,
        locationName=config.name,
        createdAt=datetime.now(timezone.utc).isoformat(),
        rainfallIntensityMmHr=request.rainfallIntensityMmHr,
        durationMin=request.durationMin,
        pattern=request.pattern,
        modelType=request.modelType,
    )
    SIMULATIONS[simulation.id] = simulation
    return simulation


@app.get(
    "/api/simulation/{simulation_id}/frames",
    response_model=SimulationFrames,
)
def simulation_frames(simulation_id: str) -> SimulationFrames:
    if simulation_id in RESULTS:
        return RESULTS[simulation_id]
    request = SIMULATIONS.get(simulation_id)
    if request is None:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    elevation, config = _get_city_grid(request.locationId)
    try:
        if request.modelType == "physics":
            started = time.perf_counter()
            times, depths, velocity_x, velocity_y = _run_isolated_anuga(
                elevation,
                config,
                request,
            )
            elapsed = time.perf_counter() - started
            frames = [
                FloodFrame(
                    t=float(times[index]),
                    rainfallMmHr=rainfall_rate(
                        request.pattern,
                        request.rainfallIntensityMmHr,
                        request.durationMin,
                        int(times[index]),
                    ),
                    depths=np.maximum(depths[index], 0.0).ravel().round(6).tolist(),
                    velocityX=velocity_x[index].ravel().round(6).tolist(),
                    velocityY=velocity_y[index].ravel().round(6).tolist(),
                )
                for index in range(len(times))
            ]
            fallback_steps = 0
            dry_steps = 0
            solver = MODEL_NAMES["physics"]
        else:
            states, elapsed, fallback_steps, dry_steps = run_city_convlstm(
                request.locationId,
                request.pattern,
                request.rainfallIntensityMmHr,
                request.durationMin,
                request.modelType,
            )
            frames = _make_frames(request, states)
            solver = MODEL_NAMES[request.modelType]
    except (ModelAssetsError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    result = SimulationFrames(
        simulationId=simulation_id,
        frames=frames,
        terrain=TerrainGrid(
            size=int(elevation.shape[0]),
            elevation=elevation.ravel().round(3).tolist(),
        ),
        diagnostics=SimulationDiagnostics(
            solver=solver,
            runtimeSeconds=float(elapsed),
            physicsFallbackSteps=int(fallback_steps),
            dryShortcutSteps=int(dry_steps),
        ),
    )
    RESULTS[simulation_id] = result
    return result

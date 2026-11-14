"""HTTP API for real ANUGA and city-trained ConvLSTM flood rollouts."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import find_spec
import json
import io
import base64
import html
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .comparison_report import (
        OllamaReportError,
        compare_final_frames,
        generate_ollama_verdict,
        verdict_facts,
    )
    from .exposure import (
        EXPOSURE_THRESHOLD_M,
        ExposureDataError,
        count_exposure,
        grid_bounds,
        load_osm_inventory,
    )
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
    from comparison_report import (
        OllamaReportError,
        compare_final_frames,
        generate_ollama_verdict,
        verdict_facts,
    )
    from exposure import (
        EXPOSURE_THRESHOLD_M,
        ExposureDataError,
        count_exposure,
        grid_bounds,
        load_osm_inventory,
    )
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
    momentumX: list[float]
    momentumY: list[float]
    affectedBuildings: int | None = None
    affectedRoadWays: int | None = None
    buildingsInCell: list[int] | None = None
    roadWaysInCell: list[int] | None = None


class GeographicBounds(BaseModel):
    west: float
    south: float
    east: float
    north: float


class TerrainGrid(BaseModel):
    size: int
    elevation: list[float]
    cellSizeM: float
    bounds: GeographicBounds


class ExposureSummary(BaseModel):
    status: Literal["available", "unavailable"]
    source: str
    message: str | None = None
    totalBuildings: int | None = None
    totalRoadWays: int | None = None
    thresholdM: float = EXPOSURE_THRESHOLD_M
    method: str = (
        "Building-node/way centroid or road-way intersection with cells at "
        "depth >= 1 cm; OSM coverage may be incomplete."
    )


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
    exposure: ExposureSummary


app = FastAPI(title="VARUNA Flood Simulation API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "https://hack-14nov.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

SIMULATIONS: dict[str, SimulationMetadata] = {}
RESULTS: dict[str, SimulationFrames] = {}


@app.get("/api/simulation/{simulation_id}/report")
def simulation_report(simulation_id: str) -> StreamingResponse:
    """Return a compact spatial risk report with a Seaborn depth heat map."""
    result = RESULTS.get(simulation_id)
    request = SIMULATIONS.get(simulation_id)
    if result is None or request is None:
        raise HTTPException(status_code=404, detail="Simulation frames are not available yet.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as error:
        raise HTTPException(status_code=503, detail="Install backend/requirements.txt to generate Seaborn reports.") from error

    frame = result.frames[-1]
    size = result.terrain.size
    depth = np.asarray(frame.depths, dtype=np.float32).reshape(size, size)
    vx = np.asarray(frame.velocityX, dtype=np.float32).reshape(size, size)
    vy = np.asarray(frame.velocityY, dtype=np.float32).reshape(size, size)
    speed = np.hypot(vx, vy)
    risk = np.select(
        [(depth >= 0.5) | (speed >= 1.5), (depth >= 0.3) | (speed >= 0.8), depth >= 0.01],
        [3, 2, 1], default=0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sns.heatmap(depth * 100, ax=axes[0], cmap="YlGnBu", cbar_kws={"label": "Depth (cm)"}, xticklabels=False, yticklabels=False)
    axes[0].set_title(f"Water depth • T+{frame.t:g} min")
    sns.heatmap(risk, ax=axes[1], cmap=sns.color_palette(["#17212b", "#46b9d6", "#f4b942", "#ed554f"], as_cmap=True), vmin=0, vmax=3, cbar=False, xticklabels=False, yticklabels=False)
    axes[1].set_title("Risk classification")
    axes[1].text(0, size + 1.2, "Dry  •  Low  •  Moderate  •  High", fontsize=9)
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=160, facecolor="#0b1118")
    plt.close(fig)
    image.seek(0)
    return StreamingResponse(image, media_type="image/png", headers={"Content-Disposition": f'attachment; filename="{simulation_id}-risk-report.png"'})


@app.get(
    "/api/simulation/{simulation_id}/comparison-report",
    response_class=HTMLResponse,
)
def simulation_comparison_report(simulation_id: str) -> HTMLResponse:
    """Compare all three forecast modes against an ANUGA reference map."""
    request = SIMULATIONS.get(simulation_id)
    if request is None:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    reference_run = RESULTS.get(simulation_id)
    if reference_run is None:
        reference_run = simulation_frames(simulation_id)
    elevation, config = _get_city_grid(request.locationId)

    model_runs: dict[str, tuple[list[FloodFrame], SimulationDiagnostics, ExposureSummary]] = {}
    for model_type in ("physics", "convlstm", "adaptive"):
        if model_type == request.modelType:
            frames = reference_run.frames
            diagnostics = reference_run.diagnostics
            exposure = reference_run.exposure
        else:
            comparison_request = request.model_copy(update={"modelType": model_type})
            frames, diagnostics = _simulate_mode(
                comparison_request,
                elevation,
                config,
            )
            frames, exposure, _ = _with_exposure(
                frames,
                config,
                int(elevation.shape[0]),
            )
        model_runs[model_type] = (frames, diagnostics, exposure)

    reference_frames = model_runs["physics"][0]
    reference_frame = min(reference_frames, key=lambda frame: abs(frame.t - 60.0))
    comparison: dict[str, object] = {
        "location": request.locationName,
        "rainfall_intensity_mm_hr": request.rainfallIntensityMmHr,
        "rainfall_duration_min": request.durationMin,
        "rainfall_pattern": request.pattern,
        "reference": "ANUGA shallow-water simulation (synthetic scenario)",
        "horizon_min": reference_frame.t,
        "metrics": {},
        "limitations": [
            "ANUGA is a simulation reference, not observed flood ground truth.",
            "The interactive scenario uses uniform rain, 5 mm/hour infiltration, and Manning n=0.05.",
            "The model comparison is specific to this one requested scenario.",
        ],
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise HTTPException(
            status_code=503,
            detail="Install backend/requirements.txt to generate model comparison maps.",
        ) from error

    final_frames: dict[str, FloodFrame] = {}
    report_rows: list[str] = []
    summary_metrics: dict[str, object] = {}
    vmax = 0.0
    for model_type, (frames, diagnostics, _) in model_runs.items():
        frame = min(frames, key=lambda item: abs(item.t - reference_frame.t))
        final_frames[model_type] = frame
        vmax = max(vmax, max(frame.depths, default=0.0))
        if model_type == "physics":
            metric_values: dict[str, float | None] = {}
            depth_mae_cell = depth_rmse_cell = csi_cell = "Reference"
            velocity_cell = volume_cell = "Reference"
        else:
            metric_values = compare_final_frames(reference_frame, frame)
            depth_mae_cell = f"{metric_values['depth_mae_m']:.4f} m"
            depth_rmse_cell = f"{metric_values['depth_rmse_m']:.4f} m"
            csi_cell = f"{metric_values['wet_area_csi_1cm']:.3f}"
            velocity_mae = metric_values[
                "velocity_vector_mae_m_s_on_reference_wet_cells"
            ]
            volume_error = metric_values["water_volume_relative_error_pct"]
            velocity_cell = (
                f"{velocity_mae:.4f} m/s"
                if velocity_mae is not None
                else "N/A (no ANUGA-wet cells)"
            )
            volume_cell = (
                f"{volume_error:.2f}%"
                if volume_error is not None
                else "N/A (zero ANUGA volume)"
            )
        summary_metrics[model_type] = {
            "solver": diagnostics.solver,
            "runtime_seconds": diagnostics.runtimeSeconds,
            **metric_values,
        }
        report_rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_NAMES[model_type])}</td>"
            f"<td>{diagnostics.runtimeSeconds:.3f} s</td>"
            f"<td>{depth_mae_cell}</td>"
            f"<td>{depth_rmse_cell}</td>"
            f"<td>{csi_cell}</td>"
            f"<td>{html.escape(velocity_cell)}</td>"
            f"<td>{html.escape(volume_cell)}</td>"
            "</tr>"
        )

    comparison["metrics"] = summary_metrics
    comparison["verdict_facts"] = verdict_facts(comparison)
    try:
        ollama_model, verdict = generate_ollama_verdict(comparison)
    except OllamaReportError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    max_depth = max(vmax, 0.01)
    for axis, model_type in zip(axes, ("physics", "convlstm", "adaptive")):
        frame = final_frames[model_type]
        grid = np.asarray(frame.depths, dtype=np.float32).reshape(
            int(elevation.shape[0]),
            int(elevation.shape[0]),
        )
        image = axis.imshow(grid, cmap="YlGnBu", vmin=0.0, vmax=max_depth)
        axis.set_title(f"{MODEL_NAMES[model_type]}\nDepth at T+{frame.t:g} min")
        axis.set_xlabel("Grid column")
        axis.set_ylabel("Grid row")
    fig.colorbar(image, ax=axes, label="Water depth (m)", shrink=0.82)
    map_image = io.BytesIO()
    fig.savefig(map_image, format="png", dpi=150, facecolor="#0b1118")
    plt.close(fig)
    image_data = base64.b64encode(map_image.getvalue()).decode("ascii")

    exposure = model_runs["physics"][2]
    if exposure.status == "available":
        exposure_text = (
            f"{exposure.totalBuildings} OSM buildings and "
            f"{exposure.totalRoadWays} OSM road ways inventoried. "
            f"At T+{reference_frame.t:g}, "
            f"{reference_frame.affectedBuildings} building centroids and "
            f"{reference_frame.affectedRoadWays} road ways intersect cells with "
            f"depth >= {exposure.thresholdM:.2f} m."
        )
    else:
        exposure_text = exposure.message or "OpenStreetMap inventory unavailable."

    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VARUNA model comparison — {html.escape(simulation_id)}</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;background:#081015;color:#e5edf2;margin:0;padding:2rem}}
main{{max-width:1200px;margin:auto}}h1,h2{{color:#b8ff24}}.note{{color:#aeb9c2}}.card{{background:#111c23;border:1px solid #263640;border-radius:8px;padding:1rem;margin:1rem 0}}
img{{width:100%;height:auto;border-radius:6px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:.7rem;border-bottom:1px solid #263640}}
.verdict{{white-space:pre-wrap}}small{{color:#9baab3}}
</style></head><body><main>
<h1>VARUNA · All-model scenario comparison</h1>
<p>{html.escape(request.locationName)} · {request.rainfallIntensityMmHr:g} mm/hr · {request.durationMin} min rain · {html.escape(request.pattern)} pattern · T+{reference_frame.t:g} min</p>
<p class="note">ANUGA is the reference for this synthetic scenario. This is not a comparison against observed flood measurements.</p>
<section class="card"><h2>Depth maps</h2><img alt="Side-by-side depth maps for ANUGA, ConvLSTM, and adaptive forecasts" src="data:image/png;base64,{image_data}"></section>
<section class="card"><h2>Runtime and agreement with ANUGA</h2>
<table><thead><tr><th>Model</th><th>Runtime</th><th>Depth MAE</th><th>Depth RMSE</th><th>CSI (1 cm)</th><th>Velocity vector MAE</th><th>Water-volume error</th></tr></thead><tbody>{''.join(report_rows)}</tbody></table>
<p class="note">CSI uses a 1 cm wet threshold. Depth MAE/RMSE are in metres. Velocity vector MAE is evaluated only on ANUGA-wet cells; water-volume relative error is undefined when ANUGA volume is zero. ANUGA's reference row is not an independent accuracy score.</p></section>
<section class="card"><h2>Buildings and roads</h2><p>{html.escape(exposure_text)}</p>
<small>Inventory source: OpenStreetMap Map API. Buildings are counted by centroid; roads are counted by OSM way intersection. OSM coverage and tagging can be incomplete. © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> (ODbL).</small></section>
<section class="card"><h2>Ollama verdict ({html.escape(ollama_model)})</h2><p class="verdict">{html.escape(verdict)}</p>
<small>Generated locally from the metrics above; it is a summary, not a safety assessment.</small></section>
<section class="card"><h2>Reproducibility and limitations</h2><p>Simulation ID: {html.escape(simulation_id)}. Inputs: {request.rainfallIntensityMmHr:g} mm/hr, {request.durationMin} min, {html.escape(request.pattern)} rainfall. All three models use the same city grid and requested storm inputs; interactive initial water is dry. The synthetic training and ANUGA reference scenarios are not observed flood measurements.</p></section>
</main></body></html>"""
    return HTMLResponse(
        report,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{simulation_id}-model-comparison.html"'
            )
        },
    )


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
                momentumX=state[1].ravel().round(6).tolist(),
                momentumY=state[2].ravel().round(6).tolist(),
            )
        )
    return frames


def _simulate_mode(
    request: SimulationMetadata,
    elevation: np.ndarray,
    config: CityModelConfig,
) -> tuple[list[FloodFrame], SimulationDiagnostics]:
    try:
        if request.modelType == "physics":
            if not _anuga_installed():
                raise RuntimeError("ANUGA is required for the physics-only model.")
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
                    momentumX=(depths[index] * velocity_x[index]).ravel().round(6).tolist(),
                    momentumY=(depths[index] * velocity_y[index]).ravel().round(6).tolist(),
                )
                for index in range(len(times))
            ]
            return frames, SimulationDiagnostics(
                solver=MODEL_NAMES["physics"],
                runtimeSeconds=float(elapsed),
                physicsFallbackSteps=0,
                dryShortcutSteps=0,
            )

        states, elapsed, fallback_steps, dry_steps = run_city_convlstm(
            request.locationId,
            request.pattern,
            request.rainfallIntensityMmHr,
            request.durationMin,
            request.modelType,
        )
        return _make_frames(request, states), SimulationDiagnostics(
            solver=MODEL_NAMES[request.modelType],
            runtimeSeconds=float(elapsed),
            physicsFallbackSteps=int(fallback_steps),
            dryShortcutSteps=int(dry_steps),
        )
    except (ModelAssetsError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _with_exposure(
    frames: list[FloodFrame],
    config: CityModelConfig,
    size: int,
) -> tuple[list[FloodFrame], ExposureSummary, GeographicBounds]:
    bounds = grid_bounds(config.longitude, config.latitude, size, 30.0)
    geographic_bounds = GeographicBounds(
        west=bounds.west,
        south=bounds.south,
        east=bounds.east,
        north=bounds.north,
    )
    try:
        inventory = load_osm_inventory(
            bounds.south,
            bounds.west,
            bounds.north,
            bounds.east,
        )
    except ExposureDataError as error:
        return frames, ExposureSummary(
            status="unavailable",
            source="OpenStreetMap Map API",
            message=str(error),
        ), geographic_bounds

    enriched_frames: list[FloodFrame] = []
    for frame in frames:
        counts = count_exposure(
            frame.depths,
            size,
            bounds,
            inventory,
        )
        enriched_frames.append(
            frame.model_copy(
                update={
                    "affectedBuildings": counts.buildings,
                    "affectedRoadWays": counts.road_ways,
                    "buildingsInCell": list(counts.buildings_by_cell),
                    "roadWaysInCell": list(counts.road_ways_by_cell),
                }
            )
        )
    return enriched_frames, ExposureSummary(
        status="available",
        source="OpenStreetMap Map API",
        totalBuildings=len(inventory.buildings),
        totalRoadWays=len(inventory.roads),
    ), geographic_bounds


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
    frames, diagnostics = _simulate_mode(request, elevation, config)
    frames, exposure, bounds = _with_exposure(
        frames,
        config,
        int(elevation.shape[0]),
    )

    result = SimulationFrames(
        simulationId=simulation_id,
        frames=frames,
        terrain=TerrainGrid(
            size=int(elevation.shape[0]),
            elevation=elevation.ravel().round(3).tolist(),
            cellSizeM=30.0,
            bounds=bounds,
        ),
        diagnostics=diagnostics,
        exposure=exposure,
    )
    RESULTS[simulation_id] = result
    return result

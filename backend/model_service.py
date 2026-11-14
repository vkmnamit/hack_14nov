"""City-specific ConvLSTM forecast execution for the frontend API."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from .evaluate_convlstm import ConvLSTMSurrogate, make_model, run_rollout
except ImportError:
    from evaluate_convlstm import ConvLSTMSurrogate, make_model, run_rollout


BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
MODEL_DIR = REPO_DIR / "model"
GRID_CELLS = 50
TIMESTEPS = 60
CELL_SIZE_M = 30.0
DEFAULT_INFILTRATION_MM_HR = 5.0
DEFAULT_MANNING_N = 0.05


@dataclass(frozen=True)
class CityModelConfig:
    location_id: str
    name: str
    region: str
    latitude: float
    longitude: float
    seed: int
    xll_m: float
    yll_m: float
    elevation_path: Path
    weights_path: Path
    normalization_path: Path
    carry_hidden_state: bool
    fallback_volume_error_pct: float


CITY_MODELS = {
    "majuli-garamur": CityModelConfig(
        location_id="majuli-garamur",
        name="Garamur, Majuli",
        region="Majuli, Assam, India",
        latitude=26.980306,
        longitude=94.1574452,
        seed=1337,
        xll_m=614110.7709637863,
        yll_m=2984030.3490885,
        elevation_path=MODEL_DIR / "majuli_elevation_m.npy",
        weights_path=MODEL_DIR / "convlstm_surrogate.pt",
        normalization_path=MODEL_DIR / "normalization_config.npz",
        carry_hidden_state=False,
        fallback_volume_error_pct=100.0,
    ),
    "guwahati": CityModelConfig(
        location_id="guwahati",
        name="Guwahati",
        region="Guwahati, Assam, India",
        latitude=26.1445,
        longitude=91.7362,
        seed=2618,
        xll_m=372920.66804708773,
        yll_m=2891550.4440719853,
        elevation_path=MODEL_DIR / "guwahati_elevation_m.npy",
        weights_path=MODEL_DIR / "guwahati_convlstm.pt",
        normalization_path=MODEL_DIR / "guwahati_normalization.npz",
        carry_hidden_state=True,
        fallback_volume_error_pct=75.0,
    ),
}

ModelMode = Literal["convlstm", "adaptive"]

_NORMALIZATION_KEYS = {
    "state_mean": "normalization_state_mean",
    "state_std": "normalization_state_std",
    "rain_mean": "normalization_raw_rainfall_mean_mm_hr",
    "rain_std": "normalization_raw_rainfall_std_mm_hr",
    "infil_mean": "normalization_infiltration_mean_mm_hr",
    "infil_std": "normalization_infiltration_std_mm_hr",
    "elev_mean": "normalization_elevation_mean_m",
    "elev_std": "normalization_elevation_std_m",
    "mann_mean": "normalization_manning_mean",
    "mann_std": "normalization_manning_std",
}


class ModelAssetsError(RuntimeError):
    """Raised when a city's local training assets are unavailable or invalid."""


def city_config(location_id: str) -> CityModelConfig:
    try:
        return CITY_MODELS[location_id]
    except KeyError as error:
        raise ValueError(f"No trained model is configured for {location_id!r}.") from error


@lru_cache(maxsize=2)
def load_city_assets(
    location_id: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    config = city_config(location_id)
    if not config.elevation_path.is_file():
        raise ModelAssetsError(
            f"{config.name} terrain grid is missing: {config.elevation_path}"
        )
    if not config.normalization_path.is_file():
        raise ModelAssetsError(
            f"Normalization file is missing: {config.normalization_path}"
        )

    elevation = np.asarray(
        np.load(config.elevation_path, allow_pickle=False),
        dtype=np.float32,
    )
    if elevation.shape != (GRID_CELLS, GRID_CELLS):
        raise ModelAssetsError(
            f"Expected a {GRID_CELLS} x {GRID_CELLS} grid for {config.name}."
        )
    with np.load(
        config.normalization_path,
        allow_pickle=False,
    ) as normalization_archive:
        normalization = {
            key: np.asarray(normalization_archive[key]).copy()
            for key in _NORMALIZATION_KEYS
        }
    return elevation, normalization


@lru_cache(maxsize=2)
def _load_model(location_id: str) -> ConvLSTMSurrogate:
    config = city_config(location_id)
    if not config.weights_path.is_file():
        raise ModelAssetsError(f"Model checkpoint is missing: {config.weights_path}")
    return make_model(config.weights_path)


def rainfall_rate(
    pattern: str,
    intensity_mm_hr: float,
    duration_min: int,
    minute: int,
) -> float:
    if minute >= duration_min:
        return 0.0
    fraction = min(minute / max(duration_min, 1), 1.0)
    if pattern == "increasing":
        return intensity_mm_hr * fraction
    if pattern == "peak":
        factor = fraction * 2.0 if fraction < 0.5 else (1.0 - fraction) * 2.0
        return intensity_mm_hr * factor
    return intensity_mm_hr


def make_rollout_data(
    elevation_m: np.ndarray,
    pattern: str,
    rainfall_intensity_mm_hr: float,
    duration_min: int,
) -> dict[str, np.ndarray]:
    rain_rates = np.asarray(
        [
            rainfall_rate(pattern, rainfall_intensity_mm_hr, duration_min, minute)
            for minute in range(TIMESTEPS)
        ],
        dtype=np.float32,
    )
    rainfall = np.broadcast_to(
        rain_rates[None, :, None, None],
        (1, TIMESTEPS, GRID_CELLS, GRID_CELLS),
    ).copy()
    infiltration = np.full(
        (1, GRID_CELLS, GRID_CELLS),
        DEFAULT_INFILTRATION_MM_HR,
        dtype=np.float32,
    )
    effective_rainfall = (
        np.maximum(rainfall - infiltration[:, None], 0.0) / 3_600_000.0
    )
    return {
        "state_sequence": np.zeros(
            (1, TIMESTEPS + 1, 3, GRID_CELLS, GRID_CELLS),
            dtype=np.float32,
        ),
        "raw_rainfall_mm_hr": rainfall,
        "rainfall_effective_m_s": effective_rainfall.astype(np.float32),
        "infiltration_loss_mm_hr": infiltration,
        "manning_n": np.full_like(infiltration, DEFAULT_MANNING_N),
        "elevation_m": np.asarray(elevation_m, dtype=np.float32),
    }


def run_city_convlstm(
    location_id: str,
    pattern: str,
    rainfall_intensity_mm_hr: float,
    duration_min: int,
    model_mode: ModelMode,
) -> tuple[np.ndarray, float, int, int]:
    config = city_config(location_id)
    elevation, normalization = load_city_assets(location_id)
    model = _load_model(location_id)
    data = make_rollout_data(
        elevation,
        pattern,
        rainfall_intensity_mm_hr,
        duration_min,
    )
    return run_rollout(
        model,
        data,
        normalization,
        scenario_index=0,
        carry_hidden_state=config.carry_hidden_state,
        fallback_volume_error_pct=(
            config.fallback_volume_error_pct if model_mode == "adaptive" else None
        ),
    )

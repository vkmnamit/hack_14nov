"""Grounded metrics and local Ollama verdicts for model comparisons."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


class OllamaReportError(RuntimeError):
    """Raised when the local Ollama service cannot produce a verdict."""


def compare_final_frames(
    reference: Any,
    prediction: Any,
) -> dict[str, float | None]:
    reference_depth = np.asarray(reference.depths, dtype=np.float64)
    predicted_depth = np.asarray(prediction.depths, dtype=np.float64)
    if reference_depth.shape != predicted_depth.shape:
        raise ValueError("Reference and prediction depth grids must have matching shapes.")

    reference_wet = reference_depth >= 0.01
    predicted_wet = predicted_depth >= 0.01
    union = np.logical_or(reference_wet, predicted_wet).sum()
    intersection = np.logical_and(reference_wet, predicted_wet).sum()
    csi = float(intersection / union) if union else 1.0
    difference = predicted_depth - reference_depth

    reference_vx = np.asarray(reference.velocityX, dtype=np.float64)
    reference_vy = np.asarray(reference.velocityY, dtype=np.float64)
    predicted_vx = np.asarray(prediction.velocityX, dtype=np.float64)
    predicted_vy = np.asarray(prediction.velocityY, dtype=np.float64)
    velocity_mask = reference_wet
    velocity_error = np.hypot(
        predicted_vx[velocity_mask] - reference_vx[velocity_mask],
        predicted_vy[velocity_mask] - reference_vy[velocity_mask],
    )
    velocity_mae = float(velocity_error.mean()) if velocity_error.size else None

    reference_volume = float(reference_depth.sum())
    prediction_volume = float(predicted_depth.sum())
    volume_error_pct = (
        abs(prediction_volume - reference_volume) / reference_volume * 100.0
        if reference_volume > 1e-8
        else None
    )
    return {
        "depth_mae_m": float(np.abs(difference).mean()),
        "depth_rmse_m": float(np.sqrt(np.square(difference).mean())),
        "wet_area_csi_1cm": csi,
        "velocity_vector_mae_m_s_on_reference_wet_cells": velocity_mae,
        "water_volume_relative_error_pct": volume_error_pct,
    }


def verdict_facts(comparison: dict[str, Any]) -> dict[str, Any]:
    metrics = comparison["metrics"]
    neural_models = {
        model_name: model_metrics
        for model_name, model_metrics in metrics.items()
        if model_name in {"convlstm", "adaptive"}
    }
    if not neural_models:
        raise ValueError("Comparison report is missing neural-model metrics.")

    lowest_mae_model = min(
        neural_models,
        key=lambda name: neural_models[name]["depth_mae_m"],
    )
    highest_csi_model = max(
        neural_models,
        key=lambda name: neural_models[name]["wet_area_csi_1cm"],
    )
    fastest_model = min(
        metrics,
        key=lambda name: metrics[name]["runtime_seconds"],
    )
    return {
        "best_neural_depth_agreement": {
            "model": lowest_mae_model,
            "depth_mae_m": neural_models[lowest_mae_model]["depth_mae_m"],
        },
        "best_neural_wet_area_agreement": {
            "model": highest_csi_model,
            "csi": neural_models[highest_csi_model]["wet_area_csi_1cm"],
        },
        "fastest_runtime_model": {
            "model": fastest_model,
            "runtime_seconds": metrics[fastest_model]["runtime_seconds"],
        },
        "reference_warning": (
            "ANUGA is the simulation reference; do not compare a neural model's "
            "accuracy against ANUGA as if ANUGA were ground truth."
        ),
        "scope_warning": (
            "This is one synthetic scenario, not a validation against observed "
            "flood measurements."
        ),
    }


def generate_ollama_verdict(comparison: dict[str, Any]) -> tuple[str, str]:
    model = os.environ.get("VARUNA_OLLAMA_MODEL", "qwen2.5:7b")
    endpoint = os.environ.get(
        "VARUNA_OLLAMA_GENERATE_URL",
        "http://127.0.0.1:11434/api/generate",
    )
    payload = {
        "model": model,
        "stream": False,
        "system": (
            "You are writing a short technical verdict for a flood-model "
            "comparison. Paraphrase only the exact findings under "
            "verdict_facts. ANUGA is a reference and is not an accuracy "
            "competitor; never say a neural model outperformed, beat, or is "
            "more accurate than ANUGA. State that the comparison is one "
            "synthetic scenario, not observed flood truth. Do not add "
            "measurements, causes, rankings, or operational safety advice."
        ),
        "prompt": (
            "Write 3-4 concise sentences using the supplied verdict facts: "
            "identify depth agreement, wet-area agreement, and runtime; mention "
            "the reference limitation. Do not make a deployment recommendation. "
            "Use only these facts:\n"
            + json.dumps(verdict_facts(comparison), allow_nan=False)
        ),
        "options": {"temperature": 0.1},
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise OllamaReportError(f"Local Ollama verdict generation failed: {error}") from error

    verdict = result.get("response")
    if not isinstance(verdict, str) or not verdict.strip():
        raise OllamaReportError("Local Ollama returned an empty verdict.")
    normalized_verdict = verdict.strip()
    forbidden_claims = (
        "outperformed anuga",
        "outperforms anuga",
        "beats anuga",
        "better than anuga",
        "more accurate than anuga",
        "outperformed the physics",
        "safer to deploy",
        "real-world accuracy",
    )
    lowered_verdict = normalized_verdict.casefold()
    if any(claim in lowered_verdict for claim in forbidden_claims):
        raise OllamaReportError(
            "Ollama produced an unsupported accuracy claim; regenerate the report."
        )
    return model, normalized_verdict

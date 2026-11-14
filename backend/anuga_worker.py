"""Isolated ANUGA process entry point for the HTTP API."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from .live_anuga import run


def main() -> None:
    payload = json.load(sys.stdin)
    times, depths, velocity_x, velocity_y = run(
        np.asarray(payload["dem"], dtype=np.float32),
        float(payload["xll"]),
        float(payload["yll"]),
        float(payload["rainfall_intensity_mm_hr"]),
        int(payload["duration_min"]),
        str(payload["pattern"]),
        forecast_duration_min=int(payload["forecast_duration_min"]),
    )
    result = json.dumps(
        {
            "times": times.tolist(),
            "depths": depths.tolist(),
            "velocity_x": velocity_x.tolist(),
            "velocity_y": velocity_y.tolist(),
        },
        allow_nan=False,
    )
    Path(sys.argv[1]).write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()

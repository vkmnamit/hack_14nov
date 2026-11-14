# Majuli ANUGA training dataset

This prototype uses ANUGA to generate shallow-water sequences over public SRTM
terrain near Garamur, Majuli, Assam. The older Sotai/LISFLOOD-FP setup remains
available as a separate comparison baseline.

## Installed models

- ANUGA 4.2.0 is installed in `backend/.venv`.
- LISFLOOD-FP 5.9 with BMI is built from the GPL-3.0
  [OpenEarth repository](https://github.com/openearth/lisflood-fp-bmi).
  This is an older, openly available release used here only as a benchmark.

Install ANUGA and build LISFLOOD-FP (OpenMP is disabled for this Apple
Silicon build):

```sh
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
```

If the LISFLOOD source checkout is not already present:

```sh
git clone --depth 1 https://github.com/openearth/lisflood-fp-bmi.git \
  backend/simulators/lisflood-fp-bmi
```

Then compile it:

```sh
make -C backend/simulators/lisflood-fp-bmi/lisflood-fp-bmi-v5.9 \
  -j2 omp=0 CC=clang++ LD=clang++
```

## Generate the Majuli dataset

Generate a multi-scenario dataset for surrogate-model experiments:

```sh
cd backend
.venv/bin/python generate_majuli_dataset.py
```

The script downloads the public 1-arc-second SRTM `N26E094` tile when it is
not already present. The 1.5 km square domain is centered on Garamur, Majuli,
Assam (26.980306 N, 94.1574452 E) and discretized into 50 x 50 cells at 30 m
spacing in UTM zone 46N. It runs 24 one-hour simulations. The synthetic
storms combine six temporal rainfall profiles with four spatial patterns.
The constant 5 mm/hour infiltration loss and Manning's n of 0.05 are
prototype assumptions; all
domain boundaries are reflective.

Outputs are under `backend/data/processed/majuli_training/`:

- `majuli_anuga_sequences.npz`: `state_sequence` has shape
  `[scenario, time, channel, row, column]`, with channels `[h, hu, hv]`;
  `rainfall_effective_m_s` has shape
  `[scenario, time, row, column]`. The archive also includes the 1 cm
  `wet_mask_1cm`, DEM, Manning roughness, times, scenario-level
  train/validation/test labels, and frame-wise mass-balance errors.
- `manifest.json`: array meanings and units, scenario definitions, split
  counts, and normalization statistics computed from training scenarios only.

For one-step supervised learning, use `state_sequence[:, :-1]` and matching
rainfall/DEM/roughness as inputs, with `state_sequence[:, 1:]` as targets.
For ConvLSTM experiments, retain complete sequences and split by the provided
scenario labels; do not randomly split adjacent frames from the same storm.
Derive velocity as `(hu/h, hv/h)` only when depth is at least 0.01 m; mask
velocity or momentum loss in near-dry cells using `wet_mask_1cm`.

### Expanded parameterized dataset (v2)

To add independent rainfall, infiltration, roughness, and initial-water inputs
without overwriting the starter data, run:

```sh
cd backend
.venv/bin/python generate_majuli_dataset_v2.py
```

This generates 72 one-hour ANUGA runs: six synthetic temporal rainfall
profiles, four spatial rainfall patterns, and three deterministic parameter
replicates per rainfall pattern. It varies Manning's `n` across 0.03, 0.05,
and 0.08; applies a scenario-constant infiltration loss of 0, 5, or
10 mm/hour; and uses dry, uniformly 2 cm wet, or lowland-puddle initial
states. Scenarios are split by whole temporal/spatial rainfall-pattern group
(36 train, 18 validation, 18 test), so related frames and parameter
replicates do not cross splits.

Outputs are under `backend/data/processed/majuli_training_v2/`:

- `majuli_anuga_convlstm_v2.npz`: `state_sequence` has shape
  `[72, 61, 3, 50, 50]`, with `[h, hu, hv]`; `initial_state` is
  `[72, 3, 50, 50]` and is the ANUGA state at `t=0`. `raw_rainfall_mm_hr` and
  `rainfall_effective_m_s` have shape `[72, 61, 50, 50]`.
  `infiltration_loss_mm_hr` and `manning_n` are scenario-specific static
  grids, and `elevation_m` is the shared DEM. The archive also includes
  scenario labels, frame times, wet masks, and frame-wise mass-balance errors.
- `manifest.json`: units, assumptions, scenario definitions, splits, array
  shapes, and normalization statistics computed from training scenarios.

For ConvLSTM training, use recent `[h, hu, hv]` states and the matching raw
rainfall, infiltration-loss, elevation, and Manning grids as inputs; predict
the next `[h, hu, hv]` state. Effective rainfall is the raw rainfall less the
scenario's constant loss, floored at zero. Flow velocity is derived from
momentum only where water depth is at least 1 cm. This simplified loss is not
a Green-Ampt infiltration model.

The v2 runs still have reflective boundaries and no river inflow, river
stages, drainage network, or observed flood labels. The varied parameter
values and initial states are synthetic experiments, not local measurements.
Mass conservation verifies the numerical water budget; it does not validate
flood accuracy for Majuli or another real location.

## Evaluate the ConvLSTM surrogate

With the checkpoint and normalization archive in the repository's `model/`
directory, run:

```sh
cd backend
.venv/bin/python evaluate_convlstm.py
```

The evaluator uses the saved ANUGA reference trajectories and every held-out
validation scenario. It reports depth, momentum, wet-area, velocity, and
water-volume metrics at 1, 5, 15, 30, and 60 minutes, with summaries grouped
by rainfall profile, rainfall pattern, initial condition, infiltration, and
Manning roughness. It also writes per-scenario CSV metrics, spatial prediction
and error fields, and maps under
`backend/data/processed/majuli_training_v2/convlstm_evaluation/`.

The saved trajectories are ANUGA references, not observed flood measurements.
The notebook's scenario index 36 is a training scenario in this dataset; use
the validation report for held-out performance.


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

## Fine-tune the ConvLSTM for recursive rollouts

The original notebook trains on one-frame inputs and resets the recurrent
state at each prediction. To train temporal memory for multi-step forecasts,
run the local fine-tuning script:

```sh
cd backend
.venv/bin/python train_convlstm.py
```

By default, it fine-tunes from the original checkpoint on the 36 training
scenarios, carries hidden state across 10-frame truncated-backpropagation
chunks, decays teacher forcing over 12 epochs, and selects the checkpoint with
the best validation score across the 18 ANUGA validation scenarios. The score
equally combines depth-MAE ratio, water-volume-error ratio, and wet-area CSI
miss ratio against the original checkpoint. A validation-only sweep over
mass-loss weights selected `1e-8`; pass `--mass-loss-weight` to override it.
The original weights are kept;
the selected model and epoch history are saved as
`model/convlstm_multistep.pt` and `model/convlstm_multistep_training.json`.

Evaluate that checkpoint with recurrent state carried between rollout steps:

```sh
.venv/bin/python evaluate_convlstm.py \
  --weights ../model/convlstm_multistep.pt \
  --carry-hidden-state \
  --output-dir data/processed/majuli_training_v2/convlstm_multistep_evaluation
```

The validation set is used to select the checkpoint. After selection, run the
separate test split once for a final assessment by adding `--split test` and
using a different `--output-dir`. All three splits are synthetic ANUGA
scenarios; they do not establish accuracy against observed floods.

## Build a Guwahati city-specific surrogate

Generate the city dataset and normalization archive:

```sh
cd backend
.venv/bin/python generate_guwahati_dataset.py
```

The script downloads the public SRTM `N26E091` tile and NASA POWER hourly
`PRECTOTCORR` data for 2020–2024. The 50 x 50 grid is centered at
26.1445 N, 91.7362 E in UTM zone 46N at 30 m spacing. Rainfall windows are
split by year (2020–2022 train, 2023 validation, 2024 test), with a six-hour
ANUGA spin-up before each 60-minute forecast. NASA POWER is coarse MERRA-2
precipitation, not a local gauge; its point value is applied uniformly across
the grid. Manning roughness, infiltration, and reflective boundaries are
simulation assumptions, not locally calibrated parameters.
The three roughness/infiltration pairs are `(0.03, 0)`, `(0.05, 5)`, and
`(0.08, 10)`; their grouped comparisons are exploratory because the parameters
are paired, not independently varied.
The city-specific validation sweep selected a mass-loss weight of `5e-8`;
selection uses only the 2023 validation year, not the 2024 test year.

Fine-tune a separate checkpoint and evaluate both held-out splits:

```sh
.venv/bin/python train_convlstm.py \
  --data data/processed/guwahati/guwahati_anuga_convlstm.npz \
  --normalization ../model/guwahati_normalization.npz \
  --output-weights ../model/guwahati_convlstm.pt \
  --history ../model/guwahati_training.json \
  --mass-loss-weight 5e-8

.venv/bin/python evaluate_convlstm.py \
  --data data/processed/guwahati/guwahati_anuga_convlstm.npz \
  --normalization ../model/guwahati_normalization.npz \
  --weights ../model/guwahati_convlstm.pt \
  --carry-hidden-state \
  --split validation \
  --output-dir data/processed/guwahati/evaluation_validation

.venv/bin/python evaluate_convlstm.py \
  --data data/processed/guwahati/guwahati_anuga_convlstm.npz \
  --normalization ../model/guwahati_normalization.npz \
  --weights ../model/guwahati_convlstm.pt \
  --carry-hidden-state \
  --split test \
  --output-dir data/processed/guwahati/evaluation_test
```

These are comparisons against ANUGA-generated scenarios, not observed Guwahati
flood measurements or a calibrated operational forecast.

## Run the Sotai/LISFLOOD-FP comparison

```sh
cd backend
.venv/bin/python run_sotai_demo.py
```

The script downloads the public 1-arc-second SRTM `N28E077` HGT tile from the
[AWS terrain tile dataset](https://registry.opendata.aws/terrain-tiles/) when
it is not already present. The test location is the OpenStreetMap-mapped
village of [Sotai](https://www.openstreetmap.org/?mlat=28.292798&mlon=77.359460#map=14/28.292798/77.359460),
Ballabgarh, Faridabad, Haryana (28.2927976 N, 77.3594597 E).

The script samples a 50 x 50 grid at 30 m spacing in UTM zone 43N and runs a
one-hour, uniform 50 mm/hour **synthetic** storm with a constant 5 mm/hour
infiltration loss and Manning's n of 0.05. These are demo assumptions, not
observed weather or calibrated soil/land-cover parameters. ANUGA uses the
effective rainfall rate (45 mm/hour); LISFLOOD-FP applies the same rainfall
and a uniform infiltration rate. Both models use reflective boundaries on
this square test domain.

## Outputs

`backend/data/processed/sotai_demo/` contains the projected ASCII DEM and
metadata. ANUGA writes its native `.sww` file and `anuga_training_data.npz`,
with time, centroid coordinates, water depth, momentum, velocity components,
speed, and flow direction (radians). LISFLOOD-FP writes depth, velocity, and
flux rasters at each output interval in its subdirectory; these are also
packed into `lisflood_testing_data.npz`. `comparison_metrics.json` reports
frame-by-frame depth RMSE and MAE after mapping ANUGA triangles to the shared
50 x 50 raster. The run checks ANUGA rainfall-volume error and LISFLOOD-FP's
volume-error diagnostic before reporting success.

## Limitations

Across the generated runs, mass-balance error checks numerical conservation,
not real-world flood prediction accuracy. No observed Majuli flood depths or
extents were used, so observational accuracy cannot yet be reported. The
dataset is a small prototype, not sufficient by itself to claim generalization
or production forecasting. SRTM's roughly 30 m terrain does not resolve
buildings, drains, or small channels; the square domain is not a delineated
catchment. Rainfall, infiltration, and roughness assumptions are synthetic
and uncalibrated. Obtain a higher-resolution local DEM, observed rainfall,
and validated land-cover/soil/drainage and flood-observation data before
using results for research or decisions. Retain the AWS SRTM attribution when
redistributing elevation data.

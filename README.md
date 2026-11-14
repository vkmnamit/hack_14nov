# VARUNA Flood Simulation

A full-stack flood-simulation prototype for 3D geospatial visualization. VARUNA offers live ANUGA shallow-water forecasts, city-trained ConvLSTM rollouts, and a ConvLSTM mode with adaptive shallow-water fallback.

---

## Architecture Overview

```
├── model/                     # Model checkpoints, normalization, and city terrain grids
│   ├── convlstm_surrogate.pt  # Majuli ConvLSTM checkpoint
│   ├── guwahati_convlstm.pt   # Guwahati ConvLSTM checkpoint
│   ├── normalization_*.npz    # Checkpoint normalization statistics
│   └── *_elevation_m.npy      # Compact 50 x 50 runtime terrain grids
│
├── backend/                   # FastAPI backend & numerical solvers
│   ├── app.py                 # FastAPI server providing Connected Mode REST API
│   ├── model_service.py       # City-specific ConvLSTM loading and rollouts
│   ├── live_anuga.py          # Parameterized ANUGA solver
│   ├── anuga_worker.py        # Isolated ANUGA process entry point
│   ├── evaluate_convlstm.py   # Autoregressive rollout evaluation engine
│   ├── train_convlstm.py      # Autoregressive fine-tuning pipeline
│   ├── shallow_water.py       # Physics-informed shallow-water equations solver
│   └── requirements.txt       # Python backend dependencies
│
└── frontend/                  # React + TypeScript + Cesium 3D Dashboard
    ├── src/
    │   ├── components/map/    # 3D Cesium tile and water-depth layer renderer
    │   ├── components/dashboard/ # Analytics, hyetograph, timeline controls
    │   ├── services/api.ts    # Connected Mode & fallback demo simulation API client
    │   └── config/appConfig.ts# Environment variable configurations
    └── .env.example           # Frontend environment template
```

---

## Quickstart

Run both backend and frontend concurrently with the provided starter script:

```bash
chmod +x ./run_dev.sh
./run_dev.sh
```

- **Frontend Application**: [http://localhost:5173](http://localhost:5173)
- **Backend API Server**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Backend Setup & Execution

### 1. Requirements & Virtual Environment
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Application
From the repository root:
```bash
./run_dev.sh
```
The development launcher starts FastAPI and Vite together. The ConvLSTM checkpoint is loaded on demand, and ANUGA runs in an isolated worker process.

### 3. API Contract
- `GET /api/health` - Reports backend and solver availability.
- `GET /api/models` - Lists ANUGA, ConvLSTM, and adaptive model modes.
- `GET /api/locations` - Lists the supported Majuli and Guwahati grids.
- `POST /api/simulate` - Accepts city, model, rainfall, and duration; returns a simulation ID.
- `GET /api/simulation/{id}/frames` - Runs the selected model and returns terrain, 61 depth/velocity frames, and measured runtime diagnostics.

---

## Frontend Setup & Execution

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Configure Environment
Copy the example environment configuration:
```bash
cp .env.example .env
```

Key environment variables:
- `VITE_API_BASE_URL`: URL to the FastAPI backend (default: `http://localhost:8000`).
- `VITE_DEMO_MODE`: Set to `true` only to use synthetic demo output; the default is connected model mode.
- `VITE_CESIUM_ION_TOKEN`: (Optional) Free token from [Cesium Ion](https://ion.cesium.com/tokens) for high-resolution 3D world terrain & photogrammetry buildings.

### 3. Run Development Server
```bash
npm run dev
```

### 4. Build for Production
```bash
npm run build
```

---

## Model Details

- **Architecture**: Stateful ConvLSTM recurrent neural network with 7 input channels:
  1. Water Depth $h$ ($m$)
  2. Momentum $hu$ ($m^2/s$)
  3. Momentum $hv$ ($m^2/s$)
  4. Raw Rainfall ($mm/hr$)
  5. Infiltration Loss ($mm/hr$)
  6. Digital Elevation Model DEM ($m$)
  7. Manning's Roughness Coefficient $n$
- **Loss Function**: Composite Loss combining MSE state reconstruction and physical mass-conservation penalty.
- **Inference modes**: ANUGA physics-only, city-trained ConvLSTM-only, and ConvLSTM with a mass-budget-triggered SWE fallback.
- **Limitations**: Forecast scenarios use synthetic assumptions and ANUGA-generated training data, not observed flood measurements. Runtime diagnostics do not imply observational accuracy.

import { createContext, useContext, useMemo, useReducer, type ReactNode } from 'react';
import {
  DEMO_LOCATIONS,
  getDatasetForLocation,
  postSimulate,
  getSimulationFrames,
  type DemoLocation,
  type ModelType,
  type SimulationMetadata,
} from '@/services/api';
import { DEMO_MODE_DEFAULT } from '@/config/appConfig';
import type { DemoDataset, RainfallPattern, SimulationStatus } from '@/lib/demoData';

interface SimulationState {
  location: DemoLocation;
  locations: DemoLocation[];
  rainfallIntensity: number;
  durationMin: number;
  pattern: RainfallPattern;
  modelType: ModelType;
  status: SimulationStatus;
  progress: number;
  dataset: DemoDataset | null;
  currentFrameIndex: number;
  playing: boolean;
  playbackSpeed: 1 | 2 | 4;
  simulationMeta: SimulationMetadata | null;
  isDemoMode: boolean;
  simulatedOnce: boolean;
  errorMessage: string | null;
  rainEnabled: boolean;
  flowEnabled: boolean;
  flowSpeedMultiplier: number;
  windDirection: number;
  windSpeed: number;
  radarEnabled: boolean;
  radarOpacity: number;
}

type Action =
  | { type: 'SET_LOCATION'; location: DemoLocation }
  | { type: 'SET_INTENSITY'; value: number }
  | { type: 'SET_DURATION'; value: number }
  | { type: 'SET_PATTERN'; value: RainfallPattern }
  | { type: 'SET_MODEL'; value: ModelType }
  | { type: 'RUN_START' }
  | { type: 'RUN_PROGRESS'; value: number }
  | { type: 'RUN_COMPLETE'; dataset: DemoDataset; meta: SimulationMetadata }
  | { type: 'RUN_ERROR'; message: string }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'RESET' }
  | { type: 'SET_FRAME'; index: number }
  | { type: 'SET_PLAYING'; value: boolean }
  | { type: 'SET_SPEED'; value: 1 | 2 | 4 }
  | { type: 'SET_RAIN_ENABLED'; value: boolean }
  | { type: 'SET_FLOW_ENABLED'; value: boolean }
  | { type: 'SET_FLOW_SPEED'; value: number }
  | { type: 'SET_WIND_DIRECTION'; value: number }
  | { type: 'SET_WIND_SPEED'; value: number }
  | { type: 'SET_RADAR_ENABLED'; value: boolean }
  | { type: 'SET_RADAR_OPACITY'; value: number };

const initialState: SimulationState = {
  location: DEMO_LOCATIONS[0],
  locations: DEMO_LOCATIONS,
  rainfallIntensity: 100,
  durationMin: 60,
  pattern: 'constant',
  modelType: 'adaptive',
  status: 'ready',
  progress: 0,
  // Terrain is available immediately so the 3D view shows the DEM tile
  // before any simulation has run.
  dataset: DEMO_MODE_DEFAULT ? getDatasetForLocation(DEMO_LOCATIONS[0]) : null,
  currentFrameIndex: 0,
  playing: false,
  playbackSpeed: 1,
  simulationMeta: null,
  isDemoMode: DEMO_MODE_DEFAULT,
  simulatedOnce: false,
  errorMessage: null,
  rainEnabled: true,
  flowEnabled: true,
  flowSpeedMultiplier: 1,
  windDirection: 180,
  windSpeed: 4,
  radarEnabled: false,
  radarOpacity: 0.55,
};

function reducer(state: SimulationState, action: Action): SimulationState {
  switch (action.type) {
    case 'SET_LOCATION':
      return {
        ...state,
        location: action.location,
        dataset: state.isDemoMode ? getDatasetForLocation(action.location) : null,
        status: 'ready',
        progress: 0,
        currentFrameIndex: 0,
        playing: false,
        simulatedOnce: false,
        errorMessage: null,
      };
    case 'SET_INTENSITY':
      return { ...state, rainfallIntensity: action.value };
    case 'SET_DURATION':
      return { ...state, durationMin: action.value };
    case 'SET_PATTERN':
      return { ...state, pattern: action.value };
    case 'SET_MODEL':
      return { ...state, modelType: action.value };
    case 'RUN_START':
      return {
        ...state,
        status: 'running',
        progress: 0,
        playing: false,
        currentFrameIndex: 0,
        errorMessage: null,
      };
    case 'RUN_PROGRESS':
      return { ...state, progress: action.value };
    case 'RUN_COMPLETE':
      return {
        ...state,
        status: 'completed',
        progress: 100,
        dataset: action.dataset,
        simulationMeta: action.meta,
        currentFrameIndex: 0,
        playing: true,
        simulatedOnce: true,
        errorMessage: null,
      };
    case 'RUN_ERROR':
      return {
        ...state,
        status: 'error',
        playing: false,
        errorMessage: action.message,
      };
    case 'PAUSE':
      return state.status === 'running' || state.playing
        ? {
            ...state,
            status: state.simulatedOnce ? 'completed' : 'paused',
            playing: false,
          }
        : state;
    case 'RESUME':
      return state.simulatedOnce ? { ...state, status: 'completed', playing: true } : state;
    case 'RESET':
      return {
        ...state,
        status: 'ready',
        progress: 0,
        currentFrameIndex: 0,
        playing: false,
        simulatedOnce: false,
        simulationMeta: null,
        errorMessage: null,
        dataset: state.dataset ? { ...state.dataset, frames: [] } : null,
      };
    case 'SET_FRAME':
      return { ...state, currentFrameIndex: action.index };
    case 'SET_PLAYING':
      return { ...state, playing: action.value };
    case 'SET_SPEED':
      return { ...state, playbackSpeed: action.value };
    case 'SET_RAIN_ENABLED':
      return { ...state, rainEnabled: action.value };
    case 'SET_FLOW_ENABLED':
      return { ...state, flowEnabled: action.value };
    case 'SET_FLOW_SPEED':
      return { ...state, flowSpeedMultiplier: action.value };
    case 'SET_WIND_DIRECTION':
      return { ...state, windDirection: action.value };
    case 'SET_WIND_SPEED':
      return { ...state, windSpeed: action.value };
    case 'SET_RADAR_ENABLED':
      return { ...state, radarEnabled: action.value };
    case 'SET_RADAR_OPACITY':
      return { ...state, radarOpacity: action.value };
    default:
      return state;
  }
}

interface SimulationContextValue {
  state: SimulationState;
  setLocation: (l: DemoLocation) => void;
  setIntensity: (v: number) => void;
  setDuration: (v: number) => void;
  setPattern: (v: RainfallPattern) => void;
  setModelType: (v: ModelType) => void;
  runSimulation: () => Promise<void>;
  pause: () => void;
  resume: () => void;
  reset: () => void;
  setFrame: (i: number) => void;
  setPlaying: (v: boolean) => void;
  setSpeed: (v: 1 | 2 | 4) => void;
  setRainEnabled: (v: boolean) => void;
  setFlowEnabled: (v: boolean) => void;
  setFlowSpeedMultiplier: (v: number) => void;
  setWindDirection: (v: number) => void;
  setWindSpeed: (v: number) => void;
  setRadarEnabled: (v: boolean) => void;
  setRadarOpacity: (v: number) => void;
}

const SimulationContext = createContext<SimulationContextValue | null>(null);

export function SimulationProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const value = useMemo<SimulationContextValue>(
    () => ({
      state,
      setLocation: (location) => dispatch({ type: 'SET_LOCATION', location }),
      setIntensity: (value) => dispatch({ type: 'SET_INTENSITY', value }),
      setDuration: (value) => dispatch({ type: 'SET_DURATION', value }),
      setPattern: (value) => dispatch({ type: 'SET_PATTERN', value }),
      setModelType: (value) => dispatch({ type: 'SET_MODEL', value }),
      runSimulation: async () => {
        dispatch({ type: 'RUN_START' });
        const req = {
          locationId: state.location.id,
          modelType: state.modelType,
          rainfallIntensityMmHr: state.rainfallIntensity,
          durationMin: state.durationMin,
          pattern: state.pattern,
        };
        try {
          if (state.isDemoMode) {
            for (const p of [25, 55, 80]) {
              await new Promise((r) => setTimeout(r, 350));
              dispatch({ type: 'RUN_PROGRESS', value: p });
            }
          } else {
            dispatch({ type: 'RUN_PROGRESS', value: 20 });
          }
          const meta = await postSimulate(req);
          dispatch({ type: 'RUN_PROGRESS', value: state.isDemoMode ? 80 : 60 });
          const { simulationId, frames, terrain, diagnostics, exposure } = await getSimulationFrames(req, meta);
          const datasetTerrain = terrain ?? (state.isDemoMode ? getDatasetForLocation(state.location).terrain : null);
          if (!datasetTerrain) {
            throw new Error('Backend response did not include terrain metadata for the flood frames.');
          }
          dispatch({
            type: 'RUN_COMPLETE',
            meta: { ...meta, id: simulationId, diagnostics },
            dataset: {
              terrain: datasetTerrain,
              frames,
              exposure,
              meta: {
                gridSize: datasetTerrain.size,
                rainfallIntensityMmHr: state.rainfallIntensity,
                durationMin: state.durationMin,
                pattern: state.pattern,
                seed: state.location.seed,
              },
            },
          });
        } catch (error) {
          dispatch({
            type: 'RUN_ERROR',
            message: error instanceof Error ? error.message : 'Simulation failed for an unknown reason.',
          });
        }
      },
      pause: () => dispatch({ type: 'PAUSE' }),
      resume: () => dispatch({ type: 'RESUME' }),
      reset: () => dispatch({ type: 'RESET' }),
      setFrame: (index) => dispatch({ type: 'SET_FRAME', index }),
      setPlaying: (value) => dispatch({ type: 'SET_PLAYING', value }),
      setSpeed: (value) => dispatch({ type: 'SET_SPEED', value }),
      setRainEnabled: (value) => dispatch({ type: 'SET_RAIN_ENABLED', value }),
      setFlowEnabled: (value) => dispatch({ type: 'SET_FLOW_ENABLED', value }),
      setFlowSpeedMultiplier: (value) => dispatch({ type: 'SET_FLOW_SPEED', value }),
      setWindDirection: (value) => dispatch({ type: 'SET_WIND_DIRECTION', value }),
      setWindSpeed: (value) => dispatch({ type: 'SET_WIND_SPEED', value }),
      setRadarEnabled: (value) => dispatch({ type: 'SET_RADAR_ENABLED', value }),
      setRadarOpacity: (value) => dispatch({ type: 'SET_RADAR_OPACITY', value }),
    }),
    [state]
  );

  return <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider>;
}

export function useSimulation(): SimulationContextValue {
  const ctx = useContext(SimulationContext);
  if (!ctx) throw new Error('useSimulation must be used within SimulationProvider');
  return ctx;
}

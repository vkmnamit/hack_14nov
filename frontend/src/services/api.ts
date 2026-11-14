/**
 * API service layer.
 *
 * Demo mode is isolated here. Connected mode uses the documented FastAPI
 * routes and fails explicitly when the backend or response is unavailable.
 */

import { API_BASE_URL, DEMO_MODE_DEFAULT } from '@/config/appConfig';
import {
  generateFloodFrames,
  generateTerrain,
  type DemoDataset,
  type ExposureSummary,
  type FloodFrame,
  type RainfallPattern,
  type TerrainGrid,
} from '@/lib/demoData';

export interface SimulationRequest {
  locationId: string;
  modelType: ModelType;
  rainfallIntensityMmHr: number;
  durationMin: number;
  pattern: RainfallPattern;
}

export type ModelType = 'physics' | 'convlstm' | 'adaptive';

export interface SimulationDiagnostics {
  solver: string;
  runtimeSeconds: number;
  physicsFallbackSteps: number;
  dryShortcutSteps: number;
}

export interface SimulationMetadata {
  id: string;
  locationId: string;
  locationName: string;
  createdAt: string;
  rainfallIntensityMmHr: number;
  durationMin: number;
  pattern: RainfallPattern;
  modelType: ModelType;
  mode: 'demo' | 'connected';
  diagnostics?: SimulationDiagnostics;
}

export interface SimulationFrames {
  simulationId: string;
  frames: FloodFrame[];
  terrain?: TerrainGrid;
  diagnostics?: SimulationDiagnostics;
  exposure?: ExposureSummary;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  version: string;
  solver: string;
}

export interface DemoLocation {
  id: string;
  name: string;
  region: string;
  longitude: number;
  latitude: number;
  /** Camera height (m) for an initial framing shot. */
  cameraHeight: number;
  seed: number;
  defaultZoom?: number;
}

export const DEMO_LOCATIONS: DemoLocation[] = [
  {
    id: 'majuli-garamur',
    name: 'Garamur, Majuli',
    region: 'Majuli, Assam, India',
    longitude: 94.1574452,
    latitude: 26.980306,
    cameraHeight: 2600,
    seed: 1337,
  },
  {
    id: 'guwahati',
    name: 'Guwahati',
    region: 'Guwahati, Assam, India',
    longitude: 91.7362,
    latitude: 26.1445,
    cameraHeight: 2600,
    seed: 2618,
  },
];

class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(`Unable to reach the backend at ${API_BASE_URL}.`);
  }
  if (!response.ok) {
    let detail = '';
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === 'string') detail = payload.detail;
    } catch {
      // Keep the HTTP status when the server does not return a JSON error body.
    }
    throw new ApiError(
      detail || `Backend request failed (${response.status} ${response.statusText}).`,
      response.status
    );
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(`Backend returned invalid JSON for ${path}.`);
  }
}

function isFloodFrame(value: unknown): value is FloodFrame {
  if (!value || typeof value !== 'object') return false;
  const frame = value as Partial<FloodFrame>;
  const depths = frame.depths;
  const velocityX = frame.velocityX;
  const velocityY = frame.velocityY;
  const momentumX = frame.momentumX;
  const momentumY = frame.momentumY;
  return (
    typeof frame.t === 'number' &&
    (frame.rainfallMmHr === undefined || Number.isFinite(frame.rainfallMmHr)) &&
    Array.isArray(depths) &&
    depths.every((depth) => Number.isFinite(depth)) &&
    ((velocityX === undefined && velocityY === undefined) ||
      (Array.isArray(velocityX) &&
        Array.isArray(velocityY) &&
        velocityX.length === depths.length &&
        velocityY.length === depths.length &&
        velocityX.every((velocity) => Number.isFinite(velocity)) &&
        velocityY.every((velocity) => Number.isFinite(velocity)))) &&
    ((momentumX === undefined && momentumY === undefined) ||
      (Array.isArray(momentumX) &&
        Array.isArray(momentumY) &&
        momentumX.length === depths.length &&
        momentumY.length === depths.length &&
        momentumX.every((momentum) => Number.isFinite(momentum)) &&
        momentumY.every((momentum) => Number.isFinite(momentum)))) &&
    (frame.affectedBuildings === undefined ||
      (Number.isInteger(frame.affectedBuildings) && frame.affectedBuildings >= 0)) &&
    (frame.affectedRoadWays === undefined ||
      (Number.isInteger(frame.affectedRoadWays) && frame.affectedRoadWays >= 0))
    &&
    ((frame.buildingsInCell === undefined && frame.roadWaysInCell === undefined) ||
      (Array.isArray(frame.buildingsInCell) &&
        Array.isArray(frame.roadWaysInCell) &&
        frame.buildingsInCell.length === depths.length &&
        frame.roadWaysInCell.length === depths.length &&
        frame.buildingsInCell.every((count) => Number.isInteger(count) && count >= 0) &&
        frame.roadWaysInCell.every((count) => Number.isInteger(count) && count >= 0)))
  );
}

function isTerrainGrid(value: unknown): value is TerrainGrid {
  if (!value || typeof value !== 'object') return false;
  const terrain = value as Partial<TerrainGrid>;
  const size = terrain.size;
  return (
    typeof size === 'number' &&
    Number.isInteger(size) &&
    size > 1 &&
    Array.isArray(terrain.elevation) &&
    terrain.elevation.length === size * size &&
    terrain.elevation.every((elevation) => Number.isFinite(elevation)) &&
    (terrain.cellSizeM === undefined ||
      (typeof terrain.cellSizeM === 'number' && terrain.cellSizeM > 0)) &&
    (terrain.bounds === undefined ||
      (typeof terrain.bounds.west === 'number' &&
        typeof terrain.bounds.south === 'number' &&
        typeof terrain.bounds.east === 'number' &&
        typeof terrain.bounds.north === 'number' &&
        terrain.bounds.west < terrain.bounds.east &&
        terrain.bounds.south < terrain.bounds.north))
  );
}

function isExposureSummary(value: unknown): value is ExposureSummary {
  if (!value || typeof value !== 'object') return false;
  const exposure = value as Partial<ExposureSummary>;
  return (
    (exposure.status === 'available' || exposure.status === 'unavailable') &&
    typeof exposure.source === 'string' &&
    typeof exposure.thresholdM === 'number' &&
    Number.isFinite(exposure.thresholdM) &&
    typeof exposure.method === 'string' &&
    (exposure.message === undefined ||
      exposure.message === null ||
      typeof exposure.message === 'string') &&
    (exposure.totalBuildings === undefined ||
      exposure.totalBuildings === null ||
      (Number.isInteger(exposure.totalBuildings) && exposure.totalBuildings >= 0)) &&
    (exposure.totalRoadWays === undefined ||
      exposure.totalRoadWays === null ||
      (Number.isInteger(exposure.totalRoadWays) && exposure.totalRoadWays >= 0))
  );
}

function isSimulationDiagnostics(value: unknown): value is SimulationDiagnostics {
  if (!value || typeof value !== 'object') return false;
  const diagnostics = value as Partial<SimulationDiagnostics>;
  return (
    typeof diagnostics.solver === 'string' &&
    typeof diagnostics.runtimeSeconds === 'number' &&
    Number.isFinite(diagnostics.runtimeSeconds) &&
    Number.isInteger(diagnostics.physicsFallbackSteps) &&
    Number.isInteger(diagnostics.dryShortcutSteps)
  );
}

function validateHealth(value: HealthResponse): HealthResponse {
  if (
    !value ||
    !['ok', 'degraded', 'down'].includes(value.status) ||
    typeof value.version !== 'string' ||
    typeof value.solver !== 'string'
  ) {
    throw new ApiError('Backend returned an invalid health payload.');
  }
  return value;
}

function validateSimulationMetadata(value: SimulationMetadata): SimulationMetadata {
  if (
    !value ||
    typeof value.id !== 'string' ||
    typeof value.locationId !== 'string' ||
    typeof value.locationName !== 'string' ||
    typeof value.createdAt !== 'string' ||
    !Number.isFinite(value.rainfallIntensityMmHr) ||
    !Number.isFinite(value.durationMin) ||
    !['constant', 'increasing', 'peak'].includes(value.pattern) ||
    !['physics', 'convlstm', 'adaptive'].includes(value.modelType) ||
    !['demo', 'connected'].includes(value.mode) ||
    (value.diagnostics !== undefined && !isSimulationDiagnostics(value.diagnostics))
  ) {
    throw new ApiError('Backend returned invalid simulation metadata.');
  }
  return value;
}

function validateFrames(value: SimulationFrames): SimulationFrames {
  if (
    !value ||
    typeof value.simulationId !== 'string' ||
    !Array.isArray(value.frames) ||
    !value.frames.every(isFloodFrame) ||
    (value.terrain !== undefined && !isTerrainGrid(value.terrain)) ||
    (value.diagnostics !== undefined && !isSimulationDiagnostics(value.diagnostics)) ||
    (value.exposure !== undefined && !isExposureSummary(value.exposure))
  ) {
    throw new ApiError('Backend returned an invalid simulation frame payload.');
  }
  if (value.terrain) {
    const expectedLength = value.terrain.size * value.terrain.size;
    if (value.frames.some((frame) => frame.depths.length !== expectedLength)) {
      throw new ApiError('Backend frame dimensions do not match the terrain grid.');
    }
  }
  return value;
}

export async function checkHealth(): Promise<HealthResponse> {
  if (DEMO_MODE_DEFAULT) {
    return { status: 'down', version: 'demo', solver: 'unavailable (demo mode)' };
  }
  return validateHealth(await fetchJson<HealthResponse>('/api/health'));
}

export async function postSimulate(req: SimulationRequest): Promise<SimulationMetadata> {
  if (!DEMO_MODE_DEFAULT) {
    const response = await fetchJson<SimulationMetadata>('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return validateSimulationMetadata(response);
  }
  const location = DEMO_LOCATIONS.find((l) => l.id === req.locationId) ?? DEMO_LOCATIONS[0];
  await new Promise((r) => setTimeout(r, 200));
  return {
    id: `sim-${Date.now().toString(36)}`,
    ...req,
    locationId: location.id,
    locationName: location.name,
    createdAt: new Date().toISOString(),
    mode: 'demo',
  };
}

export async function getSimulationFrames(
  req: SimulationRequest,
  meta: SimulationMetadata
): Promise<SimulationFrames> {
  if (!DEMO_MODE_DEFAULT) {
    const response = await fetchJson<SimulationFrames>(
      `/api/simulation/${encodeURIComponent(meta.id)}/frames`
    );
    return validateFrames(response);
  }
  const location = DEMO_LOCATIONS.find((l) => l.id === req.locationId) ?? DEMO_LOCATIONS[0];
  const terrain = generateTerrain(64, location.seed);
  const frames = generateFloodFrames(
    terrain,
    req.rainfallIntensityMmHr,
    req.durationMin,
    req.pattern
  );
  await new Promise((r) => setTimeout(r, 400));
  return {
    simulationId: meta.id,
    frames,
    terrain,
    diagnostics: {
      solver: 'Synthetic demo generator (not a flood model)',
      runtimeSeconds: 0,
      physicsFallbackSteps: 0,
      dryShortcutSteps: 0,
    },
  };
}

export function getDatasetForLocation(location: DemoLocation): DemoDataset {
  const terrain = generateTerrain(64, location.seed);
  return { terrain, frames: [], meta: { gridSize: 64, rainfallIntensityMmHr: 0, durationMin: 0, pattern: 'constant', seed: location.seed } };
}

export const apiConfig = { baseUrl: API_BASE_URL };

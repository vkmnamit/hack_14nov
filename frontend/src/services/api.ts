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
  type FloodFrame,
  type RainfallPattern,
  type TerrainGrid,
} from '@/lib/demoData';

export interface SimulationRequest {
  locationId: string;
  rainfallIntensityMmHr: number;
  durationMin: number;
  pattern: RainfallPattern;
}

export interface SimulationMetadata {
  id: string;
  locationId: string;
  locationName: string;
  createdAt: string;
  rainfallIntensityMmHr: number;
  durationMin: number;
  pattern: RainfallPattern;
  mode: 'demo' | 'connected';
}

export interface SimulationFrames {
  simulationId: string;
  frames: FloodFrame[];
  terrain?: TerrainGrid;
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
    id: 'bellandur-lake',
    name: 'Bellandur Lake Corridor',
    region: 'Devarabeesanahalli, Bengaluru',
    longitude: 77.6762,
    latitude: 12.9261,
    cameraHeight: 2400,
    seed: 1337,
  },
  {
    id: 'devarabeesanahalli',
    name: 'Devarabeesanahalli Tech District',
    region: 'Outer Ring Road, Bengaluru',
    longitude: 77.681,
    latitude: 12.935,
    cameraHeight: 3100,
    seed: 4242,
  },
  {
    id: 'hsr-layout',
    name: 'HSR Layout Catchment',
    region: 'Southern Bengaluru, Karnataka',
    longitude: 77.6387,
    latitude: 12.9116,
    cameraHeight: 2800,
    seed: 90210,
  },
  {
    id: 'koramangala-valley',
    name: 'Koramangala Valley',
    region: 'Urban drainage basin, Bengaluru',
    longitude: 77.6228,
    latitude: 12.9352,
    cameraHeight: 3400,
    seed: 777,
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
    throw new ApiError(`Backend request failed (${response.status} ${response.statusText}).`, response.status);
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
        velocityY.every((velocity) => Number.isFinite(velocity))))
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
    terrain.elevation.every((elevation) => Number.isFinite(elevation))
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
    !['demo', 'connected'].includes(value.mode)
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
    (value.terrain !== undefined && !isTerrainGrid(value.terrain))
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
  return { simulationId: meta.id, frames, terrain };
}

export function getDatasetForLocation(location: DemoLocation): DemoDataset {
  const terrain = generateTerrain(64, location.seed);
  return { terrain, frames: [], meta: { gridSize: 64, rainfallIntensityMmHr: 0, durationMin: 0, pattern: 'constant', seed: location.seed } };
}

export const apiConfig = { baseUrl: API_BASE_URL };

/**
 * Synthetic demo dataset generator.
 *
 * Produces deterministic (seeded) terrain elevation and water-depth frames for a
 * 64x64 grid, structured to mirror the future FastAPI/NumPy output:
 *   { elevation: number[] (row-major, metres), depthFrames: { t, depths[] }[] }
 *
 * The hydrology is a heavily simplified physically-plausible heuristic: water
 * accumulates in topographic lows, drains along the elevation gradient, and the
 * total volume grows with the rainfall hyetograph. It is illustrative only —
 * NOT a real hydraulic model.
 */

export interface TerrainGrid {
  size: number;
  elevation: number[]; // row-major, metres
  cellSizeM?: number;
  bounds?: {
    west: number;
    south: number;
    east: number;
    north: number;
  };
}

export interface FloodFrame {
  /** Minutes since simulation start. */
  t: number;
  /** Synthetic or backend-provided rainfall rate for this frame (mm/hr). */
  rainfallMmHr?: number;
  /** Row-major water depths in metres. */
  depths: number[];
  /** Eastward and northward flow velocities in metres per second. */
  velocityX?: number[];
  velocityY?: number[];
  /** Conserved momenta hu and hv in m²/s. */
  momentumX?: number[];
  momentumY?: number[];
  /** OSM exposure counts at this frame; absent when inventory is unavailable. */
  affectedBuildings?: number;
  affectedRoadWays?: number;
  buildingsInCell?: number[];
  roadWaysInCell?: number[];
}

export interface ExposureSummary {
  status: 'available' | 'unavailable';
  source: string;
  message?: string | null;
  totalBuildings?: number | null;
  totalRoadWays?: number | null;
  thresholdM: number;
  method: string;
}

export interface DemoDataset {
  terrain: TerrainGrid;
  frames: FloodFrame[];
  exposure?: ExposureSummary;
  meta: {
    gridSize: number;
    rainfallIntensityMmHr: number;
    durationMin: number;
    pattern: RainfallPattern;
    seed: number;
  };
}

export type RainfallPattern = 'constant' | 'increasing' | 'peak';
export type SimulationStatus = 'ready' | 'running' | 'paused' | 'completed' | 'error';
export type PhysicsCheckState = 'pass' | 'warning' | 'unavailable';

export interface PhysicsCheck {
  id: string;
  label: string;
  description: string;
  state: PhysicsCheckState;
  detail: string;
}

export interface FloodStats {
  maxDepth: number;
  meanDepth: number;
  floodedAreaKm2: number;
  floodedFraction: number;
  affectedBuildings: number;
  affectedRoadSegments: number;
  severityCounts: { band: string; count: number }[];
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (mulberry32) so the demo is reproducible
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Terrain generation: smooth value noise -> plausible urban elevation
// ---------------------------------------------------------------------------

function valueNoise2D(size: number, rand: () => number, cell: number, amp: number): number[] {
  const n = Math.ceil(size / cell) + 2;
  const lattice: number[][] = [];
  for (let j = 0; j < n; j++) {
    lattice.push(Array.from({ length: n }, () => rand()));
  }
  const smooth = (t: number) => t * t * (3 - 2 * t);
  const out: number[] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const gx = x / cell;
      const gy = y / cell;
      const x0 = Math.floor(gx);
      const y0 = Math.floor(gy);
      const tx = smooth(gx - x0);
      const ty = smooth(gy - y0);
      const a = lattice[y0][x0];
      const b = lattice[y0][x0 + 1];
      const c = lattice[y0 + 1][x0];
      const d = lattice[y0 + 1][x0 + 1];
      out.push(
        amp * ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty)
      );
    }
  }
  return out;
}

export function generateTerrain(size: number, seed: number): TerrainGrid {
  const rand = mulberry32(seed);
  const broad = valueNoise2D(size, rand, 22, 14);
  const medium = valueNoise2D(size, rand, 9, 4.5);
  const fine = valueNoise2D(size, rand, 4, 1.2);

  const elevation: number[] = new Array(size * size);
  for (let i = 0; i < size * size; i++) {
    const x = i % size;
    const y = Math.floor(i / size);
    // Gentle valley running diagonally so water has somewhere to collect
    const distToValley = Math.abs(x - y * 0.7 - 14) / size;
    const valley = -7 * Math.exp(-(distToValley * distToValley) * 18);
    elevation[i] = 22 + broad[i] + medium[i] + fine[i] + valley;
  }
  return { size, elevation, cellSizeM: 25 };
}

// ---------------------------------------------------------------------------
// Rainfall hyetograph
// ---------------------------------------------------------------------------

export function hyetograph(
  pattern: RainfallPattern,
  intensity: number,
  durationMin: number,
  t: number
): number {
  /** Instantaneous rainfall rate (mm/hr) at minute t. */
  const u = durationMin > 0 ? Math.min(t / durationMin, 1) : 0;
  switch (pattern) {
    case 'increasing':
      return intensity * u;
    case 'peak':
      // Triangular pulse peaking at mid-storm
      return intensity * (u < 0.5 ? u * 2 : 2 - u * 2);
    case 'constant':
    default:
      return u >= 1 ? 0 : intensity;
  }
}

// ---------------------------------------------------------------------------
// Demo simulation: surface-accumulation heuristic over the grid
// ---------------------------------------------------------------------------

const CELL_SIZE_M = 25; // Demo grid resolution; backend grids expose their own.
const INFILTRATION_PER_MIN = 0.0006; // m/min soak-away, illustrative

export function generateFloodFrames(
  terrain: TerrainGrid,
  intensityMmHr: number,
  durationMin: number,
  pattern: RainfallPattern,
  frameIntervalMin = 5
): FloodFrame[] {
  const n = terrain.size;
  const elev = terrain.elevation;
  const frames: FloodFrame[] = [];
  const totalMin = Math.max(durationMin, 1);
  const durationWithRecession = totalMin + 30; // show 30 min of drain-down

  const depth = new Array(n * n).fill(0);
  const normalize = (v: number, lo: number, hi: number) => Math.min(1, Math.max(0, (v - lo) / (hi - lo)));

  for (let t = 0; t <= durationWithRecession; t += frameIntervalMin) {
    const rate = hyetograph(pattern, intensityMmHr, totalMin, t) / 1000 / 60; // m/min
    const raining = t < totalMin;

    // Rain-on-grid input
    if (raining) {
      for (let i = 0; i < depth.length; i++) depth[i] += rate * frameIntervalMin;
    }
    // Infiltration whenever there is water
    for (let i = 0; i < depth.length; i++) {
      depth[i] = Math.max(0, depth[i] - INFILTRATION_PER_MIN * frameIntervalMin);
    }

    // Gravity: move a share of each cell's water to its lowest neighbour
    const flow = new Array(n * n).fill(0);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const i = y * n + x;
        if (depth[i] <= 0.001) continue;
        let lowest = i;
        let lowestH = elev[i];
        const neighbours = [
          x > 0 ? i - 1 : -1,
          x < n - 1 ? i + 1 : -1,
          y > 0 ? i - n : -1,
          y < n - 1 ? i + n : -1,
        ];
        for (const j of neighbours) {
          if (j >= 0 && elev[j] + depth[j] < lowestH) {
            lowestH = elev[j] + depth[j];
            lowest = j;
          }
        }
        if (lowest !== i) {
          // Steeper slopes move more water
          const slope = normalize(elev[i] - elev[lowest], 0, 6);
          flow[i] += depth[i] * (0.18 + 0.3 * slope);
        }
      }
    }
    for (let i = 0; i < depth.length; i++) {
      const give = Math.min(depth[i], flow[i]);
      depth[i] -= give;
      if (give > 0) {
        let lowest = i;
        let lowestH = elev[i];
        const x = i % n;
        const y = Math.floor(i / n);
        const neighbours = [
          x > 0 ? i - 1 : -1,
          x < n - 1 ? i + 1 : -1,
          y > 0 ? i - n : -1,
          y < n - 1 ? i + n : -1,
        ];
        for (const j of neighbours) {
          if (j >= 0 && elev[j] + depth[j] < lowestH) {
            lowestH = elev[j] + depth[j];
            lowest = j;
          }
        }
        depth[lowest] += give;
      }
    }

    // Rainfall intensity metadata: heavier rain converges toward larger depths
    const intensityScale = 0.35 + 0.65 * normalize(intensityMmHr, 0, 150);
    for (let i = 0; i < depth.length; i++) {
      depth[i] *= raining || t < totalMin ? 1 : 0.985; // slow recession after storm
      depth[i] = Math.min(depth[i] * (raining ? intensityScale : 1), 2.4);
    }

    const velocityX = new Array(n * n).fill(0);
    const velocityY = new Array(n * n).fill(0);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const i = y * n + x;
        if (depth[i] < 0.001) continue;
        const left = y * n + Math.max(0, x - 1);
        const right = y * n + Math.min(n - 1, x + 1);
        const north = Math.max(0, y - 1) * n + x;
        const south = Math.min(n - 1, y + 1) * n + x;
        const surface = (index: number) => elev[index] + depth[index];
        const slopeX = (surface(right) - surface(left)) / (2 * CELL_SIZE_M);
        const slopeY = (surface(south) - surface(north)) / (2 * CELL_SIZE_M);
        velocityX[i] = Math.max(-2, Math.min(2, -slopeX * 0.8));
        velocityY[i] = Math.max(-2, Math.min(2, slopeY * 0.8));
      }
    }

    frames.push({
      t,
      rainfallMmHr: raining ? hyetograph(pattern, intensityMmHr, totalMin, t) : 0,
      depths: [...depth.map((d) => Math.max(0, d))],
      velocityX,
      velocityY,
    });
  }
  return frames;
}

// ---------------------------------------------------------------------------
// Analytics derived from a frame
// ---------------------------------------------------------------------------

export const DEPTH_BANDS = [
  { label: '0–5 cm', min: 0, max: 0.05, color: '#9BD8F5' },
  { label: '5–15 cm', min: 0.05, max: 0.15, color: '#00D4FF' },
  { label: '15–30 cm', min: 0.15, max: 0.3, color: '#1560D4' },
  { label: '30–50 cm', min: 0.3, max: 0.5, color: '#FFB547' },
  { label: '>50 cm', min: 0.5, max: Infinity, color: '#FF4545' },
] as const;

export function colorForDepth(depth: number): string {
  if (depth < 0.01) return 'transparent';
  for (const band of DEPTH_BANDS) {
    if (depth < band.max) return band.color;
  }
  return DEPTH_BANDS[DEPTH_BANDS.length - 1].color;
}

export function computeStats(
  frame: FloodFrame,
  terrain: TerrainGrid,
  totalBuildings: number,
  totalRoadSegments: number
): FloodStats {
  const cellSizeM = terrain.cellSizeM ?? CELL_SIZE_M;
  const cellAreaKm2 = (cellSizeM * cellSizeM) / 1e6;
  let floodedCells = 0;
  let maxDepth = 0;
  let sum = 0;
  const bands = DEPTH_BANDS.map((b) => ({ band: b.label, count: 0 }));

  for (let i = 0; i < frame.depths.length; i++) {
    const d = frame.depths[i];
    sum += d;
    if (d > maxDepth) maxDepth = d;
    if (d >= 0.01) {
      floodedCells++;
      for (let b = 0; b < bands.length; b++) {
        if (d < DEPTH_BANDS[b].max) {
          bands[b].count++;
          break;
        }
      }
    }
  }

  // Buildings: assume denser coverage on higher ground, flooded proportional to
  // the share of inundated area within each half of the grid — deterministic.
  const floodedFraction = floodedCells / frame.depths.length;
  const meanDepth = floodedCells > 0 ? sum / frame.depths.length : 0;
  const affectedBuildings = Math.round(totalBuildings * floodedFraction * 1.15);
  const affectedRoadSegments = Math.round(totalRoadSegments * floodedFraction * 0.9);

  return {
    maxDepth,
    meanDepth,
    floodedAreaKm2: floodedCells * cellAreaKm2,
    floodedFraction,
    affectedBuildings: Math.max(0, affectedBuildings),
    affectedRoadSegments: Math.max(0, affectedRoadSegments),
    severityCounts: bands,
  };
}

export function totalSimulationMinutes(frames: FloodFrame[]): number {
  return frames.length > 0 ? frames[frames.length - 1].t : 0;
}

/** Illustrative physics diagnostics for demo frames. */
export function demoPhysicsChecks(frame: FloodFrame, prevFrame: FloodFrame | null): PhysicsCheck[] {
  let nonNegative = true;
  let maxJump = 0;
  for (const d of frame.depths) {
    if (d < 0) {
      nonNegative = false;
      break;
    }
  }
  if (prevFrame) {
    for (let i = 0; i < frame.depths.length; i++) {
      const jump = Math.abs(frame.depths[i] - prevFrame.depths[i]);
      if (jump > maxJump) maxJump = jump;
    }
  }
  return [
    {
      id: 'mass',
      label: 'Mass Conservation',
      description: 'Total water volume change equals rainfall input minus infiltration.',
      state: 'warning',
      detail: 'Illustrative demo check — not computed by a real solver.',
    },
    {
      id: 'nonneg',
      label: 'Non-Negative Water Depth',
      description: 'No grid cell has negative water depth at any time step.',
      state: nonNegative ? 'pass' : 'warning',
      detail: nonNegative
        ? 'All sampled cells >= 0 m (demo frame).'
        : 'Negative depth detected in demo frame.',
    },
    {
      id: 'stability',
      label: 'Temporal Stability',
      description: 'No cell changes depth faster than the CFL-style stability bound.',
      state: maxJump <= 0.45 || !prevFrame ? 'pass' : 'warning',
      detail: prevFrame
        ? `Max frame-to-frame change: ${maxJump.toFixed(3)} m (demo).`
        : 'Awaiting second frame.',
    },
    {
      id: 'solver',
      label: 'Numerical Solver Agreement',
      description: 'Cross-comparison against an independent reference solver.',
      state: 'unavailable',
      detail: 'Requires the FastAPI solver backend.',
    },
  ];
}

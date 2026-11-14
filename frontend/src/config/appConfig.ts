/**
 * VARUNA application configuration.
 *
 * Cesium Ion token:
 *   Cesium's global terrain (World Terrain) and OSM Buildings imagery require a
 *   free Cesium Ion access token. Obtain one at https://ion.cesium.com/tokens and
 *   either (a) place it here, (b) set VITE_CESIUM_ION_TOKEN in an .env file, or
 *   (c) paste it at runtime via the key icon in the app's top bar.
 *   Without a token the map falls back to a keyless open-imagery viewer with
 *   procedurally generated terrain — everything else keeps working.
 */

const env = (import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {};

export const CESIUM_ION_TOKEN: string =
  env.VITE_CESIUM_ION_TOKEN ?? '';

/** Base URL of the future FastAPI backend (no trailing slash). */
export const API_BASE_URL: string = env.VITE_API_BASE_URL ?? 'http://localhost:8000';

/** Demo mode is an explicit fallback and can be enabled with VITE_DEMO_MODE=true. */
export const DEMO_MODE_DEFAULT = env.VITE_DEMO_MODE === 'true';

/** Optional backend weather/radar integrations. Empty values keep them disabled. */
export const WEATHER_PROVIDER = env.VITE_WEATHER_PROVIDER ?? '';
export const WEATHER_API_URL = env.VITE_WEATHER_API_URL ?? '';
export const WEATHER_API_KEY = env.VITE_WEATHER_API_KEY ?? '';
export const RAIN_RADAR_URL = env.VITE_RAIN_RADAR_URL ?? '';
export const RAIN_RADAR_TIMES = (env.VITE_RAIN_RADAR_TIMES ?? '')
  .split(',')
  .map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value));

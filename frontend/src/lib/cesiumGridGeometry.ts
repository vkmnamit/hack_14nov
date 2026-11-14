/**
 * Custom Cesium geometry helpers.
 *
 * Cesium's stock geometries cannot express a surface that follows an
 * arbitrary height field, so we build one by hand: a 64x64 vertex grid where
 * every vertex sits at (terrain elevation + water depth) above the ellipsoid.
 * Normals are derived from the height field so lighting reveals the terrain
 * shape, and the flood layer is emitted as one geometry per depth band so the
 * legend colours stay exact.
 */

import {
  BoundingSphere,
  Cartesian3,
  ComponentDatatype,
  Geometry,
  GeometryAttribute,
  GeometryAttributes,
  Math as CesiumMath,
  PrimitiveType,
} from 'cesium';

import type { TerrainGrid } from '@/lib/demoData';

export interface GridRect {
  west: number; // degrees
  south: number;
  east: number;
  north: number;
}

export interface GridGeometryOptions {
  /** Extrude walls from the tile edge down to the ellipsoid (solid block look). */
  skirt?: boolean;
  /** Constant lift (metres) applied to every vertex — used to keep the flood
   *  surface from z-fighting with the terrain at shallow depths. */
  heightOffset?: number;
}

/** ENU unit vectors at a geodetic position, for rotating local normals to ECEF. */
function enuAt(lonDeg: number, latDeg: number) {
  const lon = CesiumMath.toRadians(lonDeg);
  const lat = CesiumMath.toRadians(latDeg);
  const sl = Math.sin(lon);
  const cl = Math.cos(lon);
  const sp = Math.sin(lat);
  const cp = Math.cos(lat);
  return {
    east: new Cartesian3(-sl, cl, 0),
    north: new Cartesian3(-sp * cl, -sp * sl, cp),
    up: new Cartesian3(cp * cl, cp * sl, sp),
  };
}

function cell(lon: number, lat: number, height: number): Cartesian3 {
  return Cartesian3.fromDegrees(lon, lat, Math.max(0, height));
}

/**
 * Builds a triangle mesh over the terrain grid.
 *
 * @param terrain DEM grid (row-major, row 0 = north)
 * @param depths  Optional water depths (metres); when given, vertices are
 *                lifted to the water surface and only wet cells are emitted.
 * @param bandOf  Optional callback mapping depth -> band index; cells whose
 *                band differs from the target are skipped.
 */
export function buildGridGeometry(
  terrain: TerrainGrid,
  rect: GridRect,
  depths: number[] | null,
  bandOf?: (depth: number) => number,
  targetBand?: number,
  options: GridGeometryOptions = {}
): Geometry | null {
  const n = terrain.size;
  const elev = terrain.elevation;
  const heightOffset = options.heightOffset ?? 0;

  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  const dLon = (rect.east - rect.west) / n;
  const dLat = (rect.north - rect.south) / n;

  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const idx = row * n + col;
      const lon = rect.west + (col + 0.5) * dLon;
      const lat = rect.north - (row + 0.5) * dLat;
      const height = elev[idx] + (depths ? depths[idx] : 0) + heightOffset;

      const surface = cell(lon, lat, height);
      positions.push(surface.x, surface.y, surface.z);

      // Height-field normal in local ENU frame, rotated to ECEF.
      const heightAt = (sample: number) => elev[sample] + (depths ? depths[sample] : 0);
      const hL = heightAt(row * n + Math.max(0, col - 1));
      const hR = heightAt(row * n + Math.min(n - 1, col + 1));
      const hU = heightAt(Math.max(0, row - 1) * n + col);
      const hD = heightAt(Math.min(n - 1, row + 1) * n + col);
      const cellM = (dLon * 111320 * Math.cos(CesiumMath.toRadians(lat))) || 1;
      const dx = (hR - hL) / (2 * cellM);
      const dy = (hD - hU) / (2 * cellM);
      const enu = enuAt(lon, lat);
      const len = Math.sqrt(dx * dx + dy * dy + 1);
      const nx = -dx / len;
      const ny = -dy / len;
      const nz = 1 / len;
      const world = new Cartesian3(
        enu.east.x * nx + enu.north.x * ny + enu.up.x * nz,
        enu.east.y * nx + enu.north.y * ny + enu.up.y * nz,
        enu.east.z * nx + enu.north.z * ny + enu.up.z * nz
      );
      Cartesian3.normalize(world, world);
      normals.push(world.x, world.y, world.z);
    }
  }

  // Surface triangles: two per cell, filtered by band membership when the
  // flood layer is being generated.
  for (let row = 0; row < n - 1; row++) {
    for (let col = 0; col < n - 1; col++) {
      const i00 = row * n + col; // north-west
      const i01 = row * n + col + 1; // north-east
      const i10 = (row + 1) * n + col; // south-west
      const i11 = (row + 1) * n + col + 1; // south-east

      if (depths && bandOf && targetBand !== undefined) {
        const dAvg = (depths[i00] + depths[i01] + depths[i10] + depths[i11]) / 4;
        if (dAvg < 0.005) continue; // dry
        if (bandOf(dAvg) !== targetBand) continue;
      }
      indices.push(i00, i10, i11, i00, i11, i01);
    }
  }

  // Skirt: duplicate the boundary ring with bottom vertices at height 0 and
  // bridge them into vertical quads, so the DEM reads as a solid block.
  if (options.skirt) {
    const boundary: number[] = [];
    for (let col = 0; col < n; col++) boundary.push(col); // north edge, W->E
    for (let row = 1; row < n; row++) boundary.push(row * n + (n - 1)); // east edge
    for (let col = n - 2; col >= 0; col--) boundary.push((n - 1) * n + col); // south edge
    for (let row = n - 2; row >= 1; row--) boundary.push(row * n); // west edge

    const topStart = positions.length / 3;
    const bottomStart = topStart + boundary.length;
    for (let i = 0; i < boundary.length; i++) {
      const idx = boundary[i];
      const row = Math.floor(idx / n);
      const col = idx % n;
      const lon = rect.west + (col + 0.5) * dLon;
      const lat = rect.north - (row + 0.5) * dLat;

      const top = cell(lon, lat, elev[idx] + heightOffset);
      positions.push(top.x, top.y, top.z);
      normals.push(0, 0, 1); // walls are unlit-looking; up normal keeps them uniform

      const bottom = cell(lon, lat, 0);
      positions.push(bottom.x, bottom.y, bottom.z);
      normals.push(0, 0, 1);
    }
    for (let i = 0; i < boundary.length; i++) {
      const a = topStart + i;
      const b = topStart + ((i + 1) % boundary.length);
      indices.push(a, b, bottomStart + ((i + 1) % boundary.length));
      indices.push(a, bottomStart + ((i + 1) % boundary.length), bottomStart + i);
    }
  }

  if (indices.length === 0) return null;

  // Bounding sphere sampled across all vertices (cheap enough at 64x64+skirt)
  const allPoints: Cartesian3[] = [];
  for (let i = 0; i < positions.length; i += 3 * 4) {
    allPoints.push(new Cartesian3(positions[i], positions[i + 1], positions[i + 2]));
  }

  return new Geometry({
    attributes: {
      position: new GeometryAttribute({
        componentDatatype: ComponentDatatype.DOUBLE,
        componentsPerAttribute: 3,
        values: new Float64Array(positions),
      }),
      normal: new GeometryAttribute({
        componentDatatype: ComponentDatatype.FLOAT,
        componentsPerAttribute: 3,
        values: new Float32Array(normals),
      }),
    } as GeometryAttributes,
    indices: new Uint32Array(indices),
    primitiveType: PrimitiveType.TRIANGLES,
    boundingSphere: BoundingSphere.fromPoints(allPoints),
  });
}

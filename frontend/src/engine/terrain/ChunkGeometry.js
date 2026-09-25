/**
 * DepthWizard Geospatial Engine — ChunkGeometry
 *
 * Constructs indexed BufferGeometry for a single terrain chunk tile with:
 *   - Real-world metric dimensions (meters)
 *   - Skirt geometry around tile perimeter to completely seal inter-LOD cracks/seams
 *   - Metric analytic normals computed from horizontal distance in meters
 *   - Precise bounding boxes for frustum culling
 */

import * as THREE from 'three';

/**
 * Builds a chunk's BufferGeometry in meters.
 *
 * @param {Object} options
 * @param {number} options.worldMinX - West boundary in meters
 * @param {number} options.worldMaxX - East boundary in meters
 * @param {number} options.worldMinZ - North boundary in meters
 * @param {number} options.worldMaxZ - South boundary in meters
 * @param {number} options.uMin - UV west [0, 1]
 * @param {number} options.uMax - UV east [0, 1]
 * @param {number} options.vMin - UV north [0, 1]
 * @param {number} options.vMax - UV south [0, 1]
 * @param {number} options.segments - Number of quad segments along each axis (e.g. 32 or 64)
 * @param {Function} options.sampleHeight - (u, v) => elevation in meters
 * @param {boolean} [options.addSkirt=true] - Whether to generate border skirt geometry
 * @param {number} [options.skirtDepth=15.0] - Downward extrusion depth in meters
 * @returns {THREE.BufferGeometry}
 */
export function buildChunkGeometry(options) {
  const {
    worldMinX, worldMaxX,
    worldMinZ, worldMaxZ,
    uMin, uMax,
    vMin, vMax,
    segments,
    sampleHeight,
    addSkirt = true,
    skirtDepth = 15.0,
  } = options;

  const wSegs = Math.max(2, segments);
  const hSegs = Math.max(2, segments);
  const gridVertsX = wSegs + 1;
  const gridVertsZ = hSegs + 1;
  const surfaceVertCount = gridVertsX * gridVertsZ;
  const actualSkirtDepth = addSkirt ? Math.max(0.5, Number(skirtDepth) || 1.2) : 0;

  // Skirt parameters: 4 borders of the grid
  // North (wSegs+1), South (wSegs+1), West (hSegs+1), East (hSegs+1) -> total perimeter vertices
  const perimeterCount = 2 * (wSegs + hSegs);
  const totalVertCount = addSkirt ? surfaceVertCount + perimeterCount : surfaceVertCount;

  // Triangles: surface quads = wSegs * hSegs * 2 triangles (6 indices)
  // Skirt quads = perimeterCount quads = perimeterCount * 2 triangles (6 indices)
  const surfaceIndices = wSegs * hSegs * 6;
  const skirtIndices = addSkirt ? perimeterCount * 6 : 0;
  const totalIndices = surfaceIndices + skirtIndices;

  const positions = new Float32Array(totalVertCount * 3);
  const normals = new Float32Array(totalVertCount * 3);
  const uvs = new Float32Array(totalVertCount * 2);
  const indices = totalIndices > 65536 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices);

  const stepX = (worldMaxX - worldMinX) / wSegs;
  const stepZ = (worldMaxZ - worldMinZ) / hSegs;
  const stepU = (uMax - uMin) / wSegs;
  const stepV = (vMax - vMin) / hSegs;

  let minElev = Infinity;
  let maxElev = -Infinity;

  // ── 1. Populate Surface Vertices ──
  let vertIdx = 0;
  for (let iz = 0; iz <= hSegs; iz++) {
    const v = vMin + iz * stepV;
    const z = worldMinZ + iz * stepZ;

    for (let ix = 0; ix <= wSegs; ix++, vertIdx++) {
      const u = uMin + ix * stepU;
      const x = worldMinX + ix * stepX;
      const elev = sampleHeight(u, v);

      if (elev < minElev) minElev = elev;
      if (elev > maxElev) maxElev = elev;

      const p3 = vertIdx * 3;
      positions[p3] = x;
      positions[p3 + 1] = elev;
      positions[p3 + 2] = z;

      const p2 = vertIdx * 2;
      // Tile-LOCAL UVs [0, 1] within this chunk (v flipped for WebGL). Each
      // chunk's material maps these back onto the global raster frame via
      // uUvOffset = (uMin, 1 - vMax) and uUvScale = (uSpan, vSpan) — this is
      // what allows every tile to carry its own streamed texture (spec §17)
      // while semantic overlays/colormaps still sample global UVs.
      uvs[p2] = ix / wSegs;
      uvs[p2 + 1] = 1.0 - iz / hSegs;

      // Metric Normal derivation using central differences in meters
      const du = Math.max(stepU * 0.5, 1e-5);
      const dv = Math.max(stepV * 0.5, 1e-5);
      const hL = sampleHeight(u - du, v);
      const hR = sampleHeight(u + du, v);
      const hU = sampleHeight(u, v - dv);
      const hD = sampleHeight(u, v + dv);

      const dhdx = (hR - hL) / Math.max(2.0 * stepX, 1e-4);
      const dhdz = (hD - hU) / Math.max(2.0 * stepZ, 1e-4);

      const len = Math.hypot(-dhdx, 1.0, -dhdz) || 1.0;
      normals[p3] = -dhdx / len;
      normals[p3 + 1] = 1.0 / len;
      normals[p3 + 2] = -dhdz / len;
    }
  }

  // ── 2. Populate Surface Triangles ──
  let indexOffset = 0;
  for (let iz = 0; iz < hSegs; iz++) {
    for (let ix = 0; ix < wSegs; ix++) {
      const a = ix + iz * gridVertsX;
      const b = ix + (iz + 1) * gridVertsX;
      const c = ix + (iz + 1) * gridVertsX + 1;
      const d = ix + iz * gridVertsX + 1;

      indices[indexOffset++] = a;
      indices[indexOffset++] = b;
      indices[indexOffset++] = d;

      indices[indexOffset++] = b;
      indices[indexOffset++] = c;
      indices[indexOffset++] = d;
    }
  }

  // ── 3. Populate Skirt Geometry (Perimeter Curtain) ──
  if (addSkirt) {
    // Gather perimeter grid indices in clockwise order
    // Top (North): (0, 0) -> (wSegs, 0)
    // Right (East): (wSegs, 0) -> (wSegs, hSegs)
    // Bottom (South): (wSegs, hSegs) -> (0, hSegs)
    // Left (West): (0, hSegs) -> (0, 0)
    const perimeterIndices = [];

    // North edge
    for (let ix = 0; ix < wSegs; ix++) perimeterIndices.push(ix);
    // East edge
    for (let iz = 0; iz < hSegs; iz++) perimeterIndices.push(wSegs + iz * gridVertsX);
    // South edge
    for (let ix = wSegs; ix > 0; ix--) perimeterIndices.push(ix + hSegs * gridVertsX);
    // West edge
    for (let iz = hSegs; iz > 0; iz--) perimeterIndices.push(iz * gridVertsX);
    const skirtVertStart = surfaceVertCount;

    for (let p = 0; p < perimeterCount; p++) {
      const topIdx = perimeterIndices[p];
      const botIdx = skirtVertStart + p;

      const px = positions[topIdx * 3];
      const py = positions[topIdx * 3 + 1];
      const pz = positions[topIdx * 3 + 2];

      const b3 = botIdx * 3;
      positions[b3] = px;
      positions[b3 + 1] = py - actualSkirtDepth; // Subtle 1.2m downward seam seal
      positions[b3 + 2] = pz;

      normals[b3] = normals[topIdx * 3];
      normals[b3 + 1] = normals[topIdx * 3 + 1];
      normals[b3 + 2] = normals[topIdx * 3 + 2];

      const b2 = botIdx * 2;
      uvs[b2] = uvs[topIdx * 2];
      uvs[b2 + 1] = uvs[topIdx * 2 + 1];

      // Connect top and bottom skirt vertices into a quad
      const nextP = (p + 1) % perimeterCount;
      const topNext = perimeterIndices[nextP];
      const botNext = skirtVertStart + nextP;

      indices[indexOffset++] = topIdx;
      indices[indexOffset++] = botIdx;
      indices[indexOffset++] = topNext;

      indices[indexOffset++] = topNext;
      indices[indexOffset++] = botIdx;
      indices[indexOffset++] = botNext;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  geometry.boundingBox = new THREE.Box3(
    new THREE.Vector3(worldMinX, minElev - (addSkirt ? actualSkirtDepth : 0), worldMinZ),
    new THREE.Vector3(worldMaxX, maxElev, worldMaxZ)
  );

  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  const radius = geometry.boundingBox.min.distanceTo(geometry.boundingBox.max) * 0.5;
  geometry.boundingSphere = new THREE.Sphere(center, radius);

  // Attach elevation range for fast queries
  geometry.userData = {
    minElevation: minElev,
    maxElevation: maxElev,
    worldMinX, worldMaxX,
    worldMinZ, worldMaxZ,
  };

  return geometry;
}

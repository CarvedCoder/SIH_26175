/**
 * DepthWizard Geospatial Engine — Automated Test Suite
 *
 * Verifies:
 *   1. Metric scale invariants (GSD 1m, 2m, 0.5m)
 *   2. Coordinate transformations & origin rebasing
 *   3. Chunk geometry generation & perimeter skirt sealing
 *   4. Bilinear elevation sampling & metric slope derivation
 *   5. Terrain collision & ground clamping
 *   6. SpatialModel 3D measurements
 *   7. Quadtree spatial hierarchy subdivision
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import { GeoReference } from '../geo/GeoReference.js';
import { buildChunkGeometry } from '../terrain/ChunkGeometry.js';
import { TerrainDataset } from '../terrain/TerrainDataset.js';
import { TerrainCollision } from '../camera/TerrainCollision.js';
import { SpatialModel } from '../spatial/SpatialModel.js';
import { QuadtreeNode } from '../tiles/QuadtreeNode.js';

test('1. Metric Scale Invariants (Section 42 test requirement)', () => {
  // GSD = 1.0m/px, 1024px -> 1024m terrain
  const geo1 = new GeoReference({
    rasterWidth: 1024,
    rasterHeight: 1024,
    gsdX: 1.0,
    gsdY: 1.0,
    isGeoreferenced: true,
  });
  assert.equal(geo1.worldWidth, 1024.0);
  assert.equal(geo1.worldDepth, 1024.0);

  // GSD = 2.0m/px, 1024px -> 2048m terrain
  const geo2 = new GeoReference({
    rasterWidth: 1024,
    rasterHeight: 1024,
    gsdX: 2.0,
    gsdY: 2.0,
    isGeoreferenced: true,
  });
  assert.equal(geo2.worldWidth, 2048.0);
  assert.equal(geo2.worldDepth, 2048.0);

  // GSD = 0.5m/px, 1024px -> 512m terrain
  const geo3 = new GeoReference({
    rasterWidth: 1024,
    rasterHeight: 1024,
    gsdX: 0.5,
    gsdY: 0.5,
    isGeoreferenced: true,
  });
  assert.equal(geo3.worldWidth, 512.0);
  assert.equal(geo3.worldDepth, 512.0);
});

test('2. Coordinate Transformations & Roundtrip Invariants', () => {
  const geo = new GeoReference({
    rasterWidth: 2000,
    rasterHeight: 1000,
    gsdX: 2.0,
    gsdY: 2.0,
    isGeoreferenced: true,
  });

  assert.equal(geo.worldWidth, 4000.0);
  assert.equal(geo.worldDepth, 2000.0);

  // Center (u=0.5, v=0.5) must be local (0, 0)
  const center = geo.uvToLocal(0.5, 0.5, 50.0);
  assert.equal(center.x, 0.0);
  assert.equal(center.z, 0.0);
  assert.equal(center.y, 50.0);

  // Roundtrip local -> UV -> local
  const testX = 1234.5;
  const testZ = -432.1;
  const uv = geo.localToUv(testX, testZ);
  const back = geo.uvToLocal(uv.u, uv.v);
  assert.ok(Math.abs(back.x - testX) < 1e-6);
  assert.ok(Math.abs(back.z - testZ) < 1e-6);

  // Roundtrip pixel -> local -> pixel
  const px = 500;
  const py = 750;
  const loc = geo.pixelToLocal(px, py);
  const backPix = geo.localToPixel(loc.x, loc.z);
  assert.equal(backPix.px, px);
  assert.equal(backPix.py, py);
});

test('3. Origin Rebasing (Floating-Origin precision protection)', () => {
  // Global UTM coordinates in millions of meters
  const globalOrigin = [500000.0, 4000000.0, 100.0];
  const geo = new GeoReference({
    rasterWidth: 1024,
    rasterHeight: 1024,
    gsdX: 1.0,
    gsdY: 1.0,
    localOrigin: globalOrigin,
    isGeoreferenced: true,
  });

  // A point 100m East and 50m North of the origin
  const globalPtX = 500100.0;
  const globalPtY = 4000050.0;
  const globalElev = 125.0;

  const local = geo.projectedToLocal(globalPtX, globalPtY, globalElev);
  assert.equal(local.x, 100.0);
  assert.equal(local.z, -50.0); // Northing is -Z in Three.js
  assert.equal(local.y, 25.0);

  // Roundtrip back to global
  const backGlobal = geo.localToProjected(local.x, local.z, local.y);
  assert.equal(backGlobal.globalX, globalPtX);
  assert.equal(backGlobal.globalY, globalPtY);
  assert.equal(backGlobal.elevation, globalElev);
});

test('4. ChunkGeometry Generation & Perimeter Skirt Sealing', () => {
  const segs = 4;
  const sampleHeight = (u, v) => (u + v) * 50.0; // Known linear plane [0 to 100m]

  const geoWithSkirt = buildChunkGeometry({
    worldMinX: -100, worldMaxX: 100,
    worldMinZ: -100, worldMaxZ: 100,
    uMin: 0, uMax: 1,
    vMin: 0, vMax: 1,
    segments: segs,
    sampleHeight,
    addSkirt: true,
    skirtDepth: 20.0,
  });

  const surfaceVerts = (segs + 1) * (segs + 1); // 25
  const perimeterVerts = 2 * (segs + segs);     // 16
  const totalVerts = surfaceVerts + perimeterVerts; // 41

  const posAttr = geoWithSkirt.getAttribute('position');
  assert.equal(posAttr.count, totalVerts);

  // Check bounding box in meters
  const bbox = geoWithSkirt.boundingBox;
  assert.equal(bbox.min.x, -100);
  assert.equal(bbox.max.x, 100);
  assert.equal(bbox.min.z, -100);
  assert.equal(bbox.max.z, 100);
  assert.equal(bbox.max.y, 100); // peak height
  assert.ok(bbox.min.y <= 0 - 20.0); // skirt dropped down

  // Verify normals attribute exists and are unit length
  const normAttr = geoWithSkirt.getAttribute('normal');
  for (let i = 0; i < 5; i++) {
    const nx = normAttr.getX(i);
    const ny = normAttr.getY(i);
    const nz = normAttr.getZ(i);
    const len = Math.hypot(nx, ny, nz);
    assert.ok(Math.abs(len - 1.0) < 1e-4);
  }
});

test('5. TerrainDataset Bilinear Sampling & Metric Slope', () => {
  const geoRef = new GeoReference({
    rasterWidth: 2,
    rasterHeight: 2,
    worldWidth: 100.0, // 100m wide
    worldDepth: 100.0, // 100m deep
    minElevation: 0.0,
    maxElevation: 100.0,
    isGeoreferenced: true,
  });

  // 2x2 grid: (0,0)=0, (1,0)=1.0, (0,1)=0, (1,1)=1.0 (slopes from West to East)
  const heightData = new Float32Array([
    0.0, 1.0,
    0.0, 1.0,
  ]);

  const dataset = new TerrainDataset({
    heightData,
    width: 2,
    height: 2,
    minElevation: 0.0,
    maxElevation: 100.0,
    isNormalized: true,
    geoRef,
  });

  // Midpoint (u=0.5, v=0.5) must be 50.0m
  const midH = dataset.sampleElevation(0.5, 0.5);
  assert.ok(Math.abs(midH - 50.0) < 1e-5);

  // West edge (u=0) must be 0m, East edge (u=1) must be 100m
  assert.ok(Math.abs(dataset.sampleElevation(0.0, 0.5) - 0.0) < 1e-5);
  assert.ok(Math.abs(dataset.sampleElevation(1.0, 0.5) - 100.0) < 1e-5);

  // Slope: 100m rise over 100m run = 45 degrees
  const slope = dataset.sampleSlope(0.5, 0.5);
  assert.ok(Math.abs(slope - 45.0) < 0.5);
});

test('6. TerrainCollision & Ground Clamping in Meters', () => {
  const geoRef = new GeoReference({
    rasterWidth: 10,
    rasterHeight: 10,
    worldWidth: 1000.0,
    worldDepth: 1000.0,
    minElevation: 100.0,
    maxElevation: 200.0,
    isGeoreferenced: true,
  });

  const dataset = {
    sampleElevation: () => 150.0, // uniform 150m terrain
  };

  const collision = new TerrainCollision({
    geoRef,
    dataset,
    eyeHeight: 1.7,
    minClearance: 2.0,
  });

  // Walking mode: camera altitude must be clamped to ground + eyeHeight (150 + 1.7 = 151.7m)
  const pos = new THREE.Vector3(0, 50, 0); // camera attempted to go under terrain (50m < 150m)
  collision.clampPosition(pos, { mode: 'walk' });
  assert.equal(pos.y, 151.7);

  // Flythrough mode: camera descending below minimum clearance (150 + 2.0 = 152.0m)
  const flyPos = new THREE.Vector3(0, 100, 0);
  collision.clampPosition(flyPos, { mode: 'fly' });
  assert.equal(flyPos.y, 152.0);

  // Flythrough mode above ground: remains unchanged
  const highPos = new THREE.Vector3(0, 300, 0);
  collision.clampPosition(highPos, { mode: 'fly' });
  assert.equal(highPos.y, 300.0);
});

test('7. SpatialModel 3D Metric Measurements', () => {
  const geoRef = new GeoReference({
    rasterWidth: 100,
    rasterHeight: 100,
    worldWidth: 1000.0,
    worldDepth: 1000.0,
    minElevation: 0.0,
    maxElevation: 100.0,
  });

  const dataset = {
    sampleElevation: (u) => (u === 0 ? 0.0 : 100.0),
    sampleSlope: () => 10.0,
    sampleNormal: () => new THREE.Vector3(0, 1, 0),
  };

  const spatial = new SpatialModel({
    geoRef,
    dataset,
    collision: null,
  });

  // Point A at (0, 0) with elevation 0m
  // Point B at (300, 400) with elevation 0m
  // Horizontal distance = sqrt(300^2 + 400^2) = 500m
  const measure = spatial.measureDistance({ x: 0, z: 0, elevation: 0 }, { x: 300, z: 400, elevation: 0 });
  assert.equal(measure.horizontalDistance_m, 500.0);
  assert.equal(measure.distance3D_m, 500.0);
  assert.equal(measure.elevationDelta_m, 0.0);

  // With 100m elevation delta: 3D distance = sqrt(500^2 + 100^2) ≈ 509.90m
  const measure3D = spatial.measureDistance({ x: 0, z: 0, elevation: 0 }, { x: 300, z: 400, elevation: 100 });
  assert.equal(measure3D.elevationDelta_m, 100.0);
  assert.ok(Math.abs(measure3D.distance3D_m - Math.hypot(500, 100)) < 1e-4);
});

test('8. Quadtree Spatial Hierarchy & LOD Subdivision', () => {
  const root = new QuadtreeNode({
    id: 'root',
    level: 0,
    maxLod: 2,
    worldMinX: -1000, worldMaxX: 1000,
    worldMinZ: -1000, worldMaxZ: 1000,
    uMin: 0, uMax: 1,
    vMin: 0, vMax: 1,
    segments: 16,
    sampleHeight: () => 0.0,
    material: new THREE.MeshBasicMaterial(),
  });

  // Split root into 4 children
  root.split();
  assert.equal(root.isSplit, true);
  assert.equal(root.children.length, 4);

  // Check child bounds
  const nw = root.children[0];
  assert.equal(nw.worldMinX, -1000);
  assert.equal(nw.worldMaxX, 0);
  assert.equal(nw.worldMinZ, -1000);
  assert.equal(nw.worldMaxZ, 0);

  // Merge children back
  root.merge();
  assert.equal(root.isSplit, false);
  assert.equal(root.children, null);
});

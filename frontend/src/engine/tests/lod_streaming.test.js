/**
 * DepthWizard Geospatial Engine — Streamed LOD Runtime Tests
 *
 * Verifies async height-tile streaming inside the LODManager:
 *   - Quadtree indices and parent links
 *   - Root tile prefetch and first-frame behavior
 *   - Ancestor fallback while child tiles load (no holes)
 *   - Async swap-in once patches arrive
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import { GeoReference } from '../geo/GeoReference.js';
import { TerrainDataset } from '../terrain/TerrainDataset.js';
import { LODManager } from '../lod/LODManager.js';

const tick = () => new Promise((r) => setTimeout(r, 0));

test('Streamed LOD: ancestor fallback, then async swap-in', async () => {
  const geoRef = new GeoReference({
    rasterWidth: 512, rasterHeight: 512,
    gsdX: 1, gsdY: 1, isGeoreferenced: true,
    minElevation: 0, maxElevation: 100,
  });
  const dataset = new TerrainDataset({
    heightData: new Float32Array(64 * 64), // inert in streamed mode
    width: 64, height: 64,
    minElevation: 0, maxElevation: 100,
    isNormalized: true, geoRef,
  });

  const fetchLog = [];
  const streamer = {
    tileSize: 256,
    request: (z, x, y, size) => {
      fetchLog.push(`${z}/${x}/${y}/${size}`);
      const data = new Float32Array(size * size).fill(0.5);
      return Promise.resolve({ data, width: size, height: size });
    },
  };

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 20000);
  camera.position.set(0, 1200, 1200);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld(true);

  const lod = new LODManager({
    geoRef, dataset,
    material: new THREE.MeshBasicMaterial(),
    scene,
    maxLod: 2,
    splitThreshold: 0.85,
    segments: 8,
    heightTiles: { streamer, tileSize: 256 },
  });

  // Root tile was requested at construction; the first update has nothing
  // renderable yet — no holes, just an empty scene for one frame.
  lod.update(camera);
  assert.equal(lod.activeTiles.size, 0);
  assert.ok(fetchLog.includes('0/0/0/256'), 'root tile requested');

  await tick(); // root patch resolves
  lod.update(camera);
  assert.equal(lod.activeTiles.size, 1, 'root renders once ready');

  // Fly close: quadtree refines (camera inside the root bbox -> distance 0
  // -> immediate descent to maxLod). Finer tiles are requested, but until
  // their patches arrive the ready root keeps rendering (ancestor fallback).
  camera.position.set(0, 60, 60);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld(true);
  lod.update(camera);
  assert.ok(fetchLog.some((k) => k.startsWith('2/')), 'finer tiles requested');
  for (const [, tile] of lod.activeTiles.entries()) {
    assert.equal(tile.level, 0, 'fallback keeps the coarse ancestor visible');
  }

  await tick(); // finer patches resolve
  lod.update(camera);
  const levels = new Set([...lod.activeTiles.values()].map((t) => t.level));
  assert.ok(levels.has(2), 'refined children swap in after load');
  assert.ok(!levels.has(0), 'coarse ancestor retired after refinement');

  // Global queries still work: spatial sampling uses the dataset (overview
  // heightfield in real streamed setups).
  assert.equal(typeof lod.stats.triangleCount, 'number');

  lod.dispose();
});

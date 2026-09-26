/**
 * DepthWizard Geospatial Engine — Tile Streaming Test Suite
 *
 * Verifies:
 *   1. TileStreamer LRU eviction, de-duplication, priority dispatch, retries
 *   2. Quadtree window arithmetic (mirrors backend _quadtree_window)
 *   3. PatchHeightfield global-UV sampling in meters
 *   4. Overview conversion for non-square rasters
 *   5. Tile-local UV geometry + material offset/scale roundtrip
 *   6. Quadtree parent links and tile indices
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import { TileStreamer } from '../streaming/TileStreamer.js';
import {
  PatchHeightfield,
  quadtreeWindow,
  overviewToHeightfield,
} from '../streaming/PatchHeightfield.js';
import { buildChunkGeometry } from '../terrain/ChunkGeometry.js';
import { QuadtreeNode } from '../tiles/QuadtreeNode.js';

const tick = () => new Promise((r) => setTimeout(r, 0));

test('1. TileStreamer: cache hits, dedup, and bounded LRU', async () => {
  let fetches = 0;
  const streamer = new TileStreamer({
    fetchTile: async () => {
      fetches++;
      return { n: fetches };
    },
    maxCacheEntries: 3,
    maxConcurrent: 4,
  });

  const a = await streamer.request(0, 0, 0, 256);
  const b = await streamer.request(0, 0, 0, 256); // cache hit
  assert.equal(a, b);
  assert.equal(fetches, 1);
  assert.equal(streamer.stats.cacheHits, 1);

  // Same tile requested twice concurrently dedupes into one fetch
  const [p1, p2] = await Promise.all([
    streamer.request(1, 0, 0, 256),
    streamer.request(1, 0, 0, 256),
  ]);
  assert.equal(p1, p2);
  assert.equal(fetches, 2);
  assert.equal(streamer.stats.deduped, 1);

  // LRU bound: fill beyond capacity, oldest entries evicted
  for (let i = 0; i < 5; i++) {
    await streamer.request(2, i, 0, 256);
  }
  assert.ok(streamer._cache.size <= 3, `cache size ${streamer._cache.size} > 3`);
  assert.equal(streamer.stats.evictions > 0, true);
  streamer.dispose();
  await tick();
});

test('2. TileStreamer: priority dispatch with concurrency 1', async () => {
  const order = [];
  let release;
  const gate = new Promise((r) => { release = r; });

  const streamer = new TileStreamer({
    fetchTile: async ({ x }) => {
      if (x === 0) await gate; // hold the first (low-priority) fetch open
      order.push(x);
      return { x };
    },
    maxConcurrent: 1,
  });

  const p0 = streamer.request(0, 0, 0, 256, { priority: 0 });
  const p1 = streamer.request(0, 1, 0, 256, { priority: 5 });
  const p2 = streamer.request(0, 2, 0, 256, { priority: 1 });
  await tick();
  release();
  await Promise.all([p0, p1, p2]);

  // Tile 1 (priority 5) must dispatch before tile 2 (priority 1)
  assert.deepEqual(order, [0, 1, 2]);
  streamer.dispose();
  await tick();
});

test('3. TileStreamer: retries a failing tile then succeeds', async () => {
  let attempts = 0;
  const streamer = new TileStreamer({
    fetchTile: async () => {
      attempts++;
      if (attempts < 2) throw new Error('transient');
      return { ok: true };
    },
    maxRetries: 2,
    retryDelayMs: 1,
  });
  const result = await streamer.request(0, 0, 0, 128);
  assert.deepEqual(result, { ok: true });
  assert.equal(attempts, 2);
  streamer.dispose();
});

test('4. Quadtree window arithmetic matches the backend', () => {
  // Adjacent tiles share the boundary pixel
  const a = quadtreeWindow(1024, 0, 0, 1);
  const b = quadtreeWindow(1024, 1, 0, 1);
  assert.equal(a[1], b[0]);

  // Non-power-of-two axis tiles fully without gaps/overlaps
  let covered = 0;
  for (let i = 0; i < 4; i++) {
    const [x0, x1] = quadtreeWindow(1000, i, 0, 2);
    assert.equal(x0, covered);
    covered = x1;
  }
  assert.equal(covered, 1000);

  // Level 0 = whole axis
  assert.deepEqual(quadtreeWindow(512, 0, 0, 0), [0, 512, 0, 512]);
});

test('5. PatchHeightfield samples meters at global UV', () => {
  // 2x2 patch spanning the whole raster: west edge 0, east edge 1
  const data = new Float32Array([0.0, 1.0, 0.0, 1.0]);
  const pf = new PatchHeightfield({
    data,
    size: 2,
    z: 0,
    tx: 0,
    ty: 0,
    rasterWidth: 100,
    rasterHeight: 100,
    minElevation: 10,
    maxElevation: 110,
  });

  assert.ok(Math.abs(pf.sampleGlobal(0, 0.5) - 10.0) < 1e-6, 'west edge = min elevation');
  assert.ok(Math.abs(pf.sampleGlobal(1, 0.5) - 110.0) < 1e-6, 'east edge = max elevation');
  assert.ok(Math.abs(pf.sampleGlobal(0.5, 0.5) - 60.0) < 1e-6, 'midpoint = mid elevation');
});

test('6. Overview conversion produces a rectangular heightfield', () => {
  // 2000x1000 raster, overview patch 512x512 (only the top half rows carry
  // real data when padded by the backend — conversion must not stretch it).
  const size = 512;
  const data = new Float32Array(size * size).fill(0.5);
  const { data: out, width, height } = overviewToHeightfield({
    data,
    size,
    rasterWidth: 2000,
    rasterHeight: 1000,
    maxEdge: 512,
  });
  assert.equal(width, 512);
  assert.equal(height, 256);
  assert.equal(out.length, width * height);
  assert.ok(Math.abs(out[0] - 0.5) < 1e-6);
});

test('7. Chunk UVs are GLOBAL raster UVs (single shared drape texture)', () => {
  const uMin = 0.25, uMax = 0.75, vMin = 0.5, vMax = 1.0;
  const geo = buildChunkGeometry({
    worldMinX: -100, worldMaxX: 100,
    worldMinZ: 0, worldMaxZ: 100,
    uMin, uMax, vMin, vMax,
    segments: 4,
    sampleHeight: () => 5.0,
  });

  const uv = geo.getAttribute('uv');
  const pos = geo.getAttribute('position');
  const wSegs = 4;

  // Vertex UVs are already GLOBAL raster UVs (v flipped for WebGL) — the
  // shared material samples the drape texture directly, no offset/scale.
    for (let iz = 0; iz <= wSegs; iz++) {
      for (let ix = 0; ix <= wSegs; ix++) {
        const vi = iz * (wSegs + 1) + ix;
        const u = uv.getX(vi);
        const v = uv.getY(vi);

        // Geometry maps tile world bounds onto this tile's global UV window;
        // reconstruct the expected global UV from the vertex position. vUv is
        // texture-space (v flipped): vUv = 1 - v_global.
        const uLocalExpect = (pos.getX(vi) + 100) / 200; // worldMinX=-100, width 200
        const vLocalExpect = (pos.getZ(vi) - 0) / 100;   // worldMinZ=0,  depth 100
        const expectU = uMin + uLocalExpect * (uMax - uMin);
        const expectV = 1 - (vMin + vLocalExpect * (vMax - vMin));
        assert.ok(Math.abs(u - expectU) < 1e-6, `u mismatch at ${ix},${iz}`);
        assert.ok(Math.abs(v - expectV) < 1e-6, `v mismatch at ${ix},${iz}`);
      }
    }
});

test('8. Quadtree children carry indices, parents, and halved bounds', () => {
  const root = new QuadtreeNode({
    id: 'root', level: 0, maxLod: 2,
    worldMinX: -100, worldMaxX: 100,
    worldMinZ: -100, worldMaxZ: 100,
    uMin: 0, uMax: 1, vMin: 0, vMax: 1,
    segments: 8,
    sampleHeight: () => 0,
    material: new THREE.MeshBasicMaterial(),
  });

  root.split();
  assert.equal(root.children.length, 4);
  const [nw, ne, sw, se] = root.children;
  assert.deepEqual([nw.tx, nw.ty], [0, 0]);
  assert.deepEqual([ne.tx, ne.ty], [1, 0]);
  assert.deepEqual([sw.tx, sw.ty], [0, 1]);
  assert.deepEqual([se.tx, se.ty], [1, 1]);
  assert.equal(nw.parent, root);
  assert.equal(nw.tile.parentNode, root, 'tile points at its parent node');
  assert.equal(root.tile.parentNode, null, 'root tile has no parent');
  assert.equal(nw.tile.tx, 0);
  root.merge();
  assert.equal(root.children, null);
});

/**
 * DepthWizard Geospatial Engine — heightDecode despike test suite
 *
 * Verifies despikeHeightfield (the rendering-quality safety net applied to
 * every decoded heightmap and streamed height tile):
 *   1. Isolated 1-px needles/pits above the threshold are replaced by the
 *      neighbourhood median
 *   2. Genuine structures survive: roof interiors, building walls, ridges
 *   3. Flat fields are bit-identical
 *   4. minDelta scaling and degenerate sizes
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { despikeHeightfield } from '../streaming/heightDecode.js';

test('despike: isolated 1-px needle replaced by neighbour median', () => {
  const w = 9, h = 9;
  const data = new Float32Array(w * h).fill(0.2);
  data[4 * w + 4] = 0.9; // 0.7 above every neighbour
  despikeHeightfield(data, w, h);
  assert.ok(Math.abs(data[4 * w + 4] - 0.2) < 1e-6, `needle not flattened: ${data[4 * w + 4]}`);
});

test('despike: isolated pit raised to neighbour median', () => {
  const w = 9, h = 9;
  const data = new Float32Array(w * h).fill(0.5);
  data[4 * w + 4] = 0.05;
  despikeHeightfield(data, w, h);
  assert.ok(Math.abs(data[4 * w + 4] - 0.5) < 1e-9, `pit not filled: ${data[4 * w + 4]}`);
});

test('despike: plateau interior and step edges survive', () => {
  const w = 9, h = 9;
  const data = new Float32Array(w * h).fill(0.2);
  // building plateau: 3x3 block at 0.8 — interior neighbours equal, edges
  // have neighbours on BOTH sides (wall) -> nothing is an isolated needle
  for (let y = 3; y <= 5; y++) {
    for (let x = 3; x <= 5; x++) data[y * w + x] = 0.8;
  }
  const before = Float32Array.from(data);
  despikeHeightfield(data, w, h);
  assert.deepEqual(Array.from(data), Array.from(before), 'plateau was modified');
});

test('despike: flat field is bit-identical', () => {
  const data = new Float32Array(16 * 16).fill(0.42);
  const before = Float32Array.from(data);
  despikeHeightfield(data, 16, 16);
  assert.deepEqual(Array.from(data), Array.from(before));
});

test('despike: minDelta gate — sub-threshold bump survives', () => {
  const w = 9, h = 9;
  const data = new Float32Array(w * h).fill(0.2);
  data[4 * w + 4] = 0.21; // only 0.01 above neighbours (below 0.02 default)
  const before = Float32Array.from(data);
  despikeHeightfield(data, w, h);
  assert.deepEqual(Array.from(data), Array.from(before), 'sub-threshold pixel was modified');
});

test('despike: degenerate sizes are no-ops', () => {
  for (const [w, h] of [[1, 1], [2, 5], [5, 2]]) {
    const data = new Float32Array(w * h).fill(0.3);
    const before = Float32Array.from(data);
    despikeHeightfield(data, w, h);
    assert.deepEqual(Array.from(data), Array.from(before));
  }
});

test('despike: returns the same (mutated) array for chaining', () => {
  const data = new Float32Array(9 * 9).fill(0.1);
  data[40] = 0.9;
  assert.equal(despikeHeightfield(data, 9, 9), data);
});

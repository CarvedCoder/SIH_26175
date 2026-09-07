/** @module api/terrain */
import { apiFetch } from './client.js';

/** GET /scenes/{id}/terrain */
export const getTerrain = (id) => apiFetch(`/scenes/${id}/terrain`);

/** GET /scenes/{id}/terrain/tiles */
export const getTerrainTiles = (id) => apiFetch(`/scenes/${id}/terrain/tiles`);

/** GET /scenes/{id}/minimap */
export const getMinimap = (id) => apiFetch(`/scenes/${id}/minimap`);

/**
 * Point elevation probe (throttled — prefer client-side heightmap sampling).
 * GET /scenes/{id}/elevation?x={x}&y={y}
 */
export const getElevation = (id, x, y) => apiFetch(`/scenes/${id}/elevation?x=${x}&y=${y}`);

/** POST /scenes/{id}/measure/height */
export const measureHeight = (id, ground, top) =>
  apiFetch(`/scenes/${id}/measure/height`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ground, top }),
  });

/** POST /scenes/{id}/measure/slope */
export const measureSlope = (id, pointA, pointB) =>
  apiFetch(`/scenes/${id}/measure/slope`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ point_a: pointA, point_b: pointB }),
  });

/** POST /scenes/{id}/refine */
export const refineArea = (id, bbox, resolution = 'high') =>
  apiFetch(`/scenes/${id}/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bbox, resolution }),
  });

/** @module api/route */
import { apiFetch } from './client.js';

/**
 * Route Assist — jury round 1: vehicle passability + path recommendation.
 * POST /scenes/{id}/route/assess
 * @param {string} sceneId
 * @param {{x: number, y: number}} start - source-raster pixel coords
 * @param {{x: number, y: number}} end
 * @param {string[]} vehicles - fire_truck | ambulance | rescue_atv | suv_4x4
 * @returns {Promise<object>} RouteAssessResponse
 */
export const assessRoute = (sceneId, start, end, vehicles) =>
  apiFetch(`/scenes/${sceneId}/route/assess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ start, end, vehicles }),
  });

/**
 * Mini heat map (jury round 2): passability PNG URL + risk stats.
 * GET /scenes/{id}/route/heatmap?vehicle=...
 */
export const getRouteHeatmap = (sceneId, vehicle = 'fire_truck') =>
  apiFetch(`/scenes/${sceneId}/route/heatmap?vehicle=${vehicle}`);

export const VEHICLES = [
  { key: 'fire_truck', label: 'Fire Truck' },
  { key: 'ambulance', label: 'Ambulance' },
  { key: 'rescue_atv', label: 'Rescue ATV' },
  { key: 'suv_4x4', label: '4×4 SUV' },
];

export const VERDICT_META = {
  CAN_GO: { label: 'CAN GO', color: '#2ecc71', glyph: '✓' },
  CAUTION: { label: 'CAUTION', color: '#f1c40f', glyph: '⚠' },
  CANNOT_GO: { label: 'CANNOT GO', color: '#e74c3c', glyph: '✗' },
};

/** @module api/semantic */
import { apiFetch, getBaseUrl } from './client.js';

/**
 * Fetch semantic segmentation metadata and availability for a scene.
 * GET /scenes/{scene_id}/semantic
 * @param {string} sceneId
 * @returns {Promise<object>} SemanticMetaResponse
 */
export const getSemanticMeta = (sceneId) =>
  apiFetch(`/scenes/${sceneId}/semantic`);

/**
 * Returns the URL for the semantic colored preview PNG.
 * @param {string} sceneId
 * @returns {string}
 */
export const getSemanticMapUrl = (sceneId) =>
  `${getBaseUrl()}/scenes/${sceneId}/results/semantic`;

/**
 * Returns the URL for downloading the raw semantic label numpy array (.npy).
 * @param {string} sceneId
 * @returns {string}
 */
export const getSemanticLabelsUrl = (sceneId) =>
  `${getBaseUrl()}/scenes/${sceneId}/results/semantic-labels`;

/**
 * Returns the URL for the combined route-risk heat map PNG.
 * @param {string} sceneId
 * @param {string} vehicle
 * @returns {string}
 */
export const getRouteRiskUrl = (sceneId, vehicle = 'fire_truck') =>
  `${getBaseUrl()}/scenes/${sceneId}/results/route-risk?vehicle=${encodeURIComponent(vehicle)}`;

export const SEMANTIC_CLASSES = [
  'building',
  'vegetation',
  'road',
  'water',
  'ground',
  'other',
];

export const SEMANTIC_CLASS_LABELS = {
  building: 'Building',
  vegetation: 'Vegetation / Canopy',
  road: 'Road',
  water: 'Water Body',
  ground: 'Ground / Open Terrain',
  other: 'Other / Unknown',
};

// Normalized float RGBA (0.0 - 1.0) for WebGL shaders
export const SEMANTIC_COLORS = {
  building:   [0.906, 0.298, 0.235, 1.0],  // #E74C3C
  vegetation: [0.180, 0.800, 0.443, 1.0],  // #2ECC71
  road:       [0.608, 0.608, 0.608, 1.0],  // #9B9B9B
  water:      [0.204, 0.596, 0.859, 1.0],  // #3498DB
  ground:     [0.824, 0.706, 0.549, 1.0],  // #D2B48C
  other:      [0.584, 0.510, 0.659, 1.0],  // #9582A8
};

// Hex strings for DOM elements / CSS
export const SEMANTIC_COLORS_HEX = {
  building:   '#E74C3C',
  vegetation: '#2ECC71',
  road:       '#9B9B9B',
  water:      '#3498DB',
  ground:     '#D2B48C',
  other:      '#9582A8',
};

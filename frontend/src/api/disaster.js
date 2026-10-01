/** @module api/disaster
 *
 * API client for building detection and damage assessment endpoints.
 * These are completely separate from the 6-class semantic segmentation.
 */
import { apiFetch, getBaseUrl } from './client.js';

/**
 * Fetch building detection metadata and availability.
 * GET /scenes/{scene_id}/buildings
 * @param {string} sceneId
 * @returns {Promise<object>}
 */
export const getBuildingsMeta = (sceneId) =>
  apiFetch(`/scenes/${sceneId}/buildings`);

/**
 * Fetch damage assessment metadata and availability.
 * GET /scenes/{scene_id}/damage
 * @param {string} sceneId
 * @returns {Promise<object>}
 */
export const getDamageMeta = (sceneId) =>
  apiFetch(`/scenes/${sceneId}/damage`);

/**
 * URL for buildings GeoJSON.
 * @param {string} sceneId
 * @returns {string}
 */
export const getBuildingsGeoJsonUrl = (sceneId) =>
  `${getBaseUrl()}/scenes/${sceneId}/results/buildings-geojson`;

/**
 * URL for buildings preview PNG.
 * @param {string} sceneId
 * @returns {string}
 */
export const getBuildingsPreviewUrl = (sceneId) =>
  `${getBaseUrl()}/scenes/${sceneId}/results/buildings-preview`;

/**
 * URL for damage GeoJSON.
 * @param {string} sceneId
 * @returns {string}
 */
export const getDamageGeoJsonUrl = (sceneId) =>
  `${getBaseUrl()}/scenes/${sceneId}/results/damage-geojson`;

/**
 * URL for damage preview PNG.
 * @param {string} sceneId
 * @returns {string}
 */
export const getDamagePreviewUrl = (sceneId) =>
  `${getBaseUrl()}/scenes/${sceneId}/results/damage-preview`;

// Damage class definitions — SEPARATE from semantic classes
export const DAMAGE_CLASSES = [
  'no-damage',
  'minor-damage',
  'major-damage',
  'destroyed',
];

export const DAMAGE_CLASS_LABELS = {
  'no-damage': 'No Damage',
  'minor-damage': 'Minor Damage',
  'major-damage': 'Major Damage',
  'destroyed': 'Destroyed',
};

// Damage class colors for visualization
export const DAMAGE_COLORS_HEX = {
  'no-damage':    '#4CAF50',
  'minor-damage': '#FFC107',
  'major-damage': '#FF5722',
  'destroyed':    '#D32F2F',
};

// RGBA floats for WebGL
export const DAMAGE_COLORS_RGBA = {
  'no-damage':    [0.298, 0.686, 0.314, 0.8],
  'minor-damage': [1.000, 0.757, 0.027, 0.8],
  'major-damage': [1.000, 0.341, 0.133, 0.8],
  'destroyed':    [0.827, 0.184, 0.184, 0.9],
};

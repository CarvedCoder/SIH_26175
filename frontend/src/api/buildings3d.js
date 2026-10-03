/** @module api/buildings3d
 *
 * API client for the geometry-aware 3D building reconstruction layer
 * (footprints + DSM-derived heights + confidence). Independent of the
 * 2D building-footprint detection and of semantic segmentation.
 */
import { apiFetch, getBaseUrl } from './client.js';

/**
 * Fetch 3D reconstruction metadata and availability.
 * GET /scenes/{scene_id}/buildings3d
 * @param {string} sceneId
 * @returns {Promise<object>}
 */
export const getBuildings3DMeta = (sceneId) =>
  apiFetch(`/scenes/${sceneId}/buildings3d`);

/**
 * Fetch the full reconstruction (footprint polygons + heights).
 * GET /scenes/{scene_id}/results/buildings3d
 * @param {string} sceneId
 * @returns {Promise<object>}
 */
export const getBuildings3D = (sceneId) =>
  apiFetch(`/scenes/${sceneId}/results/buildings3d`);

/**
 * URL for the reconstruction preview PNG.
 * @param {string} sceneId
 * @returns {string}
 */
export const getBuildings3DPreviewUrl = (sceneId) =>
  `${getBaseUrl()}/scenes/${sceneId}/results/buildings3d-preview`;

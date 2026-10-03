/** @module api/buildings3d
 *
 * API client for the geometry-aware 3D building reconstruction layer
 * (footprints + DSM-derived heights + confidence). Independent of the
 * 2D building-footprint detection and of semantic segmentation.
 */
import { apiFetch, assetFetch, getBaseUrl, resolveAssetUrl } from './client.js';

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
 *
 * Two-step transport (the app's established pattern, same as the minimap):
 * the metadata route returns a short-lived presigned URL (or the legacy
 * file route when no object store is configured), which is then fetched
 * via assetFetch. Fetching the result-file route directly would mean
 * following a cross-origin 307, which some browser/network stacks refuse
 * even when both endpoints serve correct CORS headers.
 * @param {string} sceneId
 * @returns {Promise<object>}
 */
export const getBuildings3D = async (sceneId) => {
  const meta = await apiFetch(`/scenes/${sceneId}/buildings3d`);
  if (!meta?.available || !meta.url) {
    return { available: false, count: 0, buildings: [] };
  }
  const res = await assetFetch(resolveAssetUrl(meta.url));
  if (!res.ok) {
    throw Object.assign(new Error(`buildings3d fetch failed: ${res.status}`), {
      status: res.status,
    });
  }
  const json = await res.json();
  // carry the presigned ground-heightmap URL through — the result-file
  // route cannot be fetched directly (307 redirect fails cross-origin)
  return { ...json, ground_heightmap_url: meta.ground_heightmap_url ?? null };
};

/**
 * URL for the reconstruction preview PNG.
 * @param {string} sceneId
 * @returns {string}
 */
export const getBuildings3DPreviewUrl = (sceneId) =>
  `${getBaseUrl()}/scenes/${sceneId}/results/buildings3d-preview`;

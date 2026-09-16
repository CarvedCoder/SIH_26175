/** @module api/upload */
import { apiFetch } from './client.js';

/**
 * Upload an image file and create a scene.
 * POST /scenes
 * @param {File} file
 * @param {{ referenceType?: string, referenceFile?: File, gcpFile?: File }} [opts]
 * @returns {Promise<import('../types/api.js').SceneUploadResponse>}
 */
export async function uploadScene(file, opts = {}) {
  const form = new FormData();
  form.append('file', file);
  if (opts.referenceType) form.append('reference_type', opts.referenceType);
  if (opts.referenceFile) form.append('reference_file', opts.referenceFile);
  if (opts.gcpFile)       form.append('gcp_file', opts.gcpFile);

  return apiFetch('/scenes', { method: 'POST', body: form });
}

/**
 * Validate an uploaded scene before processing (optional endpoint).
 * POST /scenes/{id}/validate
 * @param {string} sceneId
 * @returns {Promise<import('../types/api.js').ValidationCheckResponse>}
 */
export async function validateScene(sceneId) {
  return apiFetch(`/scenes/${sceneId}/validate`, { method: 'POST' });
}

/**
 * Attempt to merge additional GeoTIFF files into an existing scene as a
 * mosaic (adjacent-tile auto-detection). Rejects with an error whose
 * `status` is 400 when the inputs are NOT genuinely adjacent tiles
 * (CRS / ground-resolution / footprint mismatch) — callers should fall
 * back to treating each file as its own scene.
 * POST /scenes/{id}/mosaic
 * @param {string} sceneId
 * @param {File[]} files - 2+ GeoTIFF files (the already-uploaded scene
 *   input re-sent first, then the remaining tiles)
 * @returns {Promise<import('../types/api.js').SceneUploadResponse>}
 */
export async function mosaicScenes(sceneId, files) {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  return apiFetch(`/scenes/${sceneId}/mosaic`, { method: 'POST', body: form });
}

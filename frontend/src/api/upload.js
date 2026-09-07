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

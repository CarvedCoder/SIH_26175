/** @module api/processing */
import { apiFetch } from './client.js';

/**
 * Start a processing job for a scene.
 * POST /scenes/{id}/process
 * @param {string} sceneId
 * @param {{ model?: string, tileSize?: number, overlap?: number, enableRefinement?: boolean, enableReferenceCalibration?: boolean }} [opts]
 * @returns {Promise<import('../types/api.js').JobStartResponse>}
 */
export async function startProcessing(sceneId, opts = {}) {
  return apiFetch(`/scenes/${sceneId}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model:                       opts.model                      ?? 'depth-anything-v2',
      tile_size:                   opts.tileSize                   ?? 1024,
      overlap:                     opts.overlap                    ?? 0.15,
      enable_refinement:           opts.enableRefinement           ?? true,
      enable_reference_calibration: opts.enableReferenceCalibration ?? true,
    }),
  });
}

/**
 * Poll job status.
 * GET /jobs/{jobId}
 * @param {string} jobId
 * @returns {Promise<import('../types/api.js').JobStatus>}
 */
export async function getJobStatus(jobId) {
  return apiFetch(`/jobs/${jobId}`);
}

/**
 * Cancel a running job.
 * POST /jobs/{jobId}/cancel
 * @param {string} jobId
 */
export async function cancelJob(jobId) {
  return apiFetch(`/jobs/${jobId}/cancel`, { method: 'POST' });
}

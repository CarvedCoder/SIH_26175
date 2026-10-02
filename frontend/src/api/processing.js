/** @module api/processing */
import { apiFetch } from './client.js';

/**
 * Start a processing job for a scene.
 * POST /scenes/{id}/process
 *
 * The backend runs the DAv2 depth backbone plus the selected height-model
 * backend with a fixed 1024 training tile contract; it accepts `mode`,
 * `architecture` and `ground_elev` and rejects unknown options with a 422,
 * so no phantom knobs are sent here.
 *
 * @param {string} sceneId
 * @param {{
 *   mode?: 'auto'|'crop'|'resize'|'tiles',
 *   architecture?: 'rdah'|'calibration_net'|'terraheight_s',
 *   groundElev?: number,
 * }} [opts]
 * @returns {Promise<import('../types/api.js').JobStartResponse>}
 */
export async function startProcessing(sceneId, opts = {}) {
  const body = {};
  if (opts.mode) body.mode = opts.mode;
  if (opts.architecture) body.architecture = opts.architecture;
  if (opts.groundElev != null) body.ground_elev = opts.groundElev;

  return apiFetch(`/scenes/${sceneId}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
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

/**
 * Submit a local tile refinement job for a selected bounding box.
 * POST /scenes/{id}/refine
 * @param {string} sceneId
 * @param {{ bbox: { x_min: number, y_min: number, x_max: number, y_max: number }, resolution?: 'standard'|'high', architecture?: 'rdah'|'calibration_net'|'terraheight_s' }} opts
 * @returns {Promise<{ job_id: string, status: string }>}
 */
export async function refineScene(sceneId, opts) {
  return apiFetch(`/scenes/${sceneId}/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      bbox: opts.bbox,
      resolution: opts.resolution ?? 'high',
      architecture: opts.architecture ?? 'rdah',
    }),
  });
}

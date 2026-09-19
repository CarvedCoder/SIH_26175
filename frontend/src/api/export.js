/** @module api/export */
import { downloadArtifact } from './client.js';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1').replace(/\/$/, '');

/**
 * Trigger an authenticated file download: fetches with the Supabase
 * token attached (the backend streams the file or redirects to a
 * short-lived signed URL) and hands the blob to the browser.
 */
function download(id, type) {
  return downloadArtifact(`${BASE_URL}/scenes/${id}/export/${type}`, `${id}_${type}`);
}

export const exportDsm        = (id) => download(id, 'dsm');
export const exportDepth      = (id) => download(id, 'depth');
export const exportValidation = (id) => download(id, 'validation');
export const exportTerrain    = (id) => download(id, 'terrain');

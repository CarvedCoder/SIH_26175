/** @module api/export */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1').replace(/\/$/, '');

/**
 * Trigger a file download by navigating to the export URL.
 * The backend either streams the file or returns a redirect to a temp URL.
 */
function triggerDownload(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  if (filename) a.download = filename;
  a.click();
}

export const exportDsm        = (id) => triggerDownload(`${BASE_URL}/scenes/${id}/export/dsm`);
export const exportDepth      = (id) => triggerDownload(`${BASE_URL}/scenes/${id}/export/depth`);
export const exportValidation = (id) => triggerDownload(`${BASE_URL}/scenes/${id}/export/validation`);
export const exportTerrain    = (id) => triggerDownload(`${BASE_URL}/scenes/${id}/export/terrain`);

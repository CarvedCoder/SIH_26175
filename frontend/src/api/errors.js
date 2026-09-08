/**
 * DepthWizard — Error Mapping & Diagnostic System (Phase 18, Task 18.2)
 *
 * Implements spec §31 Rule 6:
 *   "Errors should be understandable:
 *    - What happened
 *    - Why it happened
 *    - What can I do next?"
 *
 * Standard API error contract per spec §69.
 */

export const ERROR_DEFINITIONS = {
  INVALID_FILE: {
    title: 'Invalid File Structure',
    what: 'The uploaded file could not be decoded as a valid raster image.',
    why: 'The file header is corrupted or bytes do not match the expected image format.',
    whatNext: 'Check the file on your local machine and verify it opens in standard image viewers before re-uploading.',
    recoverable: true,
  },
  UNSUPPORTED_FORMAT: {
    title: 'Unsupported Image Format',
    what: 'This file format is not accepted by the DepthWizard reconstruction pipeline.',
    why: 'The pipeline accepts standard PNG, JPEG/JPG, and 16/32-bit GeoTIFF imagery.',
    whatNext: 'Export or convert your imagery to PNG, JPG, or GeoTIFF before uploading.',
    recoverable: true,
  },
  FILE_TOO_LARGE: {
    title: 'File Exceeds Size Limit',
    what: 'The image resolution or filesize exceeds the memory budget for monocular processing.',
    why: 'Images larger than 4096×4096 pixels or 100 MB may exhaust GPU memory during tile inference.',
    whatNext: 'Crop a focused region of interest or downsample the image resolution before uploading.',
    recoverable: true,
  },
  INVALID_GEOTIFF: {
    title: 'Invalid GeoTIFF Raster',
    what: 'The GeoTIFF tags and metadata could not be parsed.',
    why: 'Missing standard raster tags, unsupported compression scheme, or damaged header directory.',
    whatNext: 'Export your GeoTIFF with Deflate or LZW compression using standard GDAL/QGIS tooling.',
    recoverable: true,
  },
  NO_GEOREFERENCE: {
    title: 'No Georeference Tags Found',
    what: 'This image lacks coordinate reference system (CRS) and spatial resolution tags.',
    why: 'Absolute elevation calibration requires real-world coordinates and geographic bounds.',
    whatNext: 'You can proceed using the Relative DSM pipeline (in scene units) or upload a georeferenced GeoTIFF.',
    recoverable: true,
    canFallbackRelative: true,
  },
  REFERENCE_DATA_UNAVAILABLE: {
    title: 'Reference Elevation Unavailable',
    what: 'No ground-truth reference DEM was found covering this bounding box.',
    why: 'SRTM, ALOS, or local reference DEM coverage is unavailable for these coordinates.',
    whatNext: 'Proceed with uncalibrated Relative DSM, or provide a local DEM / GCP companion file.',
    recoverable: true,
    canFallbackRelative: true,
  },
  DEPTH_MODEL_ERROR: {
    title: 'Depth Estimation Interrupted',
    what: 'The monocular depth neural network encountered an error during inference.',
    why: 'Model execution interrupted due to CUDA memory exhaustion or tensor dimension mismatch.',
    whatNext: 'Click Retry to re-run with fallback tile size, or restart the backend model service.',
    recoverable: true,
  },
  DSM_GENERATION_ERROR: {
    title: 'DSM Generation Interrupted',
    what: 'Digital Surface Model generation and scale calibration could not complete.',
    why: 'Reference DEM alignment failed to converge or elevation scale optimization was singular.',
    whatNext: 'Click "Continue with Relative DSM" to view estimated terrain without metric calibration, or Retry.',
    recoverable: true,
    canFallbackRelative: true,
  },
  TERRAIN_GENERATION_ERROR: {
    title: '3D Terrain Mesh Build Failed',
    what: 'Could not construct the 3D surface mesh from the heightmap buffer.',
    why: 'WebGL vertex buffer allocation failed or heightmap displacement data is invalid.',
    whatNext: 'Click Retry or reduce the visual exaggeration factor in the terrain menu.',
    recoverable: true,
  },
  VALIDATION_ERROR: {
    title: 'Validation Metrics Incomplete',
    what: 'Could not calculate RMSE, MAE, or Pearson correlation vs reference DEM.',
    why: 'Insufficient overlap between estimated elevation grid and reference terrain samples.',
    whatNext: 'Explore the reconstructed 3D surface directly; qualitative analysis is unaffected.',
    recoverable: true,
  },
  JOB_NOT_FOUND: {
    title: 'Processing Job Not Found',
    what: 'The requested job identifier does not exist on the processing backend.',
    why: 'The job expired, the server was restarted, or the ID is invalid.',
    whatNext: 'Start a new reconstruction by uploading an image or selecting a sample scene.',
    recoverable: false,
  },
  SCENE_NOT_FOUND: {
    title: 'Scene Not Found',
    what: 'The requested terrain scene could not be located.',
    why: 'Temporary scene storage may have been cleared or the scene ID is outdated.',
    whatNext: 'Select a project from Recent Projects or upload a new image to recreate the scene.',
    recoverable: false,
  },
  RESOURCE_LIMIT: {
    title: 'Server Busy / Resource Limit',
    what: 'The backend inference engine is currently operating at full capacity.',
    why: 'All GPU worker slots are engaged with active tiling jobs.',
    whatNext: 'Wait a few moments and click Retry to resume your request.',
    recoverable: true,
  },
  NETWORK_ERROR: {
    title: 'Backend Unreachable',
    what: 'Could not connect to the DepthWizard API service.',
    why: 'The FastAPI backend (http://localhost:8000) is currently offline or unreachable.',
    whatNext: 'Verify that the backend server is running with `uvicorn main:app --reload`, or review cached projects.',
    recoverable: true,
  },
  INTERNAL_ERROR: {
    title: 'Internal Service Interruption',
    what: 'An unexpected processing error occurred.',
    why: 'An unhandled server exception occurred during execution.',
    whatNext: 'Check the backend console logs for diagnostic traceback and click Retry.',
    recoverable: true,
  },
};

/**
 * Format any error into a structured 3-part diagnosis per spec §31 Rule 6.
 * @param {any} error - raw or API error
 * @returns {{
 *   code: string,
 *   title: string,
 *   what: string,
 *   why: string,
 *   whatNext: string,
 *   recoverable: boolean,
 *   canFallbackRelative: boolean,
 *   rawMessage: string,
 * }}
 */
export function formatApiError(error) {
  if (!error) {
    return {
      code: 'UNKNOWN',
      title: 'Unknown Error',
      what: 'An unexpected problem occurred.',
      why: 'No error details were reported.',
      whatNext: 'Try again or refresh the browser.',
      recoverable: true,
      canFallbackRelative: false,
      rawMessage: '',
    };
  }

  const code = error.code || (error.status === 0 ? 'NETWORK_ERROR' : 'INTERNAL_ERROR');
  const matched = ERROR_DEFINITIONS[code] || ERROR_DEFINITIONS.INTERNAL_ERROR;

  return {
    code,
    title: matched.title,
    what: matched.what,
    why: error.message || matched.why,
    whatNext: matched.whatNext,
    recoverable: error.recoverable ?? matched.recoverable,
    canFallbackRelative: matched.canFallbackRelative ?? false,
    rawMessage: error.message || String(error),
  };
}

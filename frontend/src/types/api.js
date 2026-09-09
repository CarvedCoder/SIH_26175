/**
 * DepthWizard — API Type Definitions (JSDoc schemas)
 * These describe the backend response shapes. See spec §43–§83.
 */

/**
 * @typedef {Object} SceneUploadResponse
 * @property {string}  scene_id
 * @property {string}  filename
 * @property {string}  format          - 'GeoTIFF' | 'PNG' | 'JPG'
 * @property {number}  width
 * @property {number}  height
 * @property {number}  channels
 * @property {boolean} georeferenced
 * @property {string|null} crs
 * @property {{ min_x: number, min_y: number, max_x: number, max_y: number }|null} bounds
 * @property {string}  processing_path - 'absolute_dsm' | 'relative_dsm'
 * @property {boolean} reference_available
 */

/**
 * @typedef {Object} ValidationCheckResponse
 * @property {boolean}  valid
 * @property {string[]} warnings
 * @property {string[]} errors
 * @property {string[]} recommendations
 */

/**
 * @typedef {Object} JobStartResponse
 * @property {string} job_id
 * @property {string} scene_id
 * @property {string} status - 'queued'
 */

/**
 * @typedef {Object} JobStatus
 * @property {string}      job_id
 * @property {string}      scene_id
 * @property {string}      status       - 'queued'|'preprocessing'|'depth_estimation'|'geospatial_alignment'|'scale_calibration'|'refinement'|'dsm_generation'|'validation'|'terrain_generation'|'completed'|'failed'|'cancelled'
 * @property {string}      stage
 * @property {number}      progress     - 0–100
 * @property {number|null} current_tile
 * @property {number|null} total_tiles
 * @property {string|null} message
 */

/**
 * @typedef {Object} SceneSummary
 * @property {string}  scene_id
 * @property {string}  status
 * @property {{ filename: string, width: number, height: number, format: string, georeferenced: boolean, crs: string|null }} input
 * @property {{ model: string, tile_size: number, overlap: number }} processing
 * @property {{ depth: boolean, dsm: boolean, terrain: boolean, validation: boolean }} outputs
 */

/**
 * @typedef {Object} ResultsMeta
 * @property {string}   scene_id
 * @property {string[]} available_layers - e.g. ['rgb','depth','dsm','reference_dem','error','slope']
 * @property {string}   elevation_mode   - 'absolute' | 'relative'
 * @property {string}   units            - 'meters' | 'scene_units'
 * @property {Object}   bounds
 * @property {{ width: number, height: number }} resolution
 */

/**
 * @typedef {Object} DepthResult
 * @property {string}  scene_id
 * @property {string}  url
 * @property {number}  min
 * @property {number}  max
 * @property {string}  format
 */

/**
 * @typedef {Object} DsmResult
 * @property {string}  scene_id
 * @property {string}  mode          - 'absolute' | 'relative'
 * @property {string}  units
 * @property {string}  format
 * @property {string}  download_url
 * @property {number}  min_elevation
 * @property {number}  max_elevation
 */

/**
 * @typedef {Object} TerrainMeta
 * @property {string}  scene_id
 * @property {string}  terrain_type   - 'heightfield'
 * @property {number}  width
 * @property {number}  height
 * @property {Object}  bounds
 * @property {number}  height_scale
 * @property {number}  min_elevation
 * @property {number}  max_elevation
 * @property {string}  heightmap_url
 * @property {string}  texture_url
 */

/**
 * @typedef {Object} MinimapMeta
 * @property {string}  image_url
 * @property {number}  width
 * @property {number}  height
 * @property {{ min_x: number, min_y: number, max_x: number, max_y: number }} bounds
 * @property {string}  coordinate_system - 'EPSG:4326' | 'image'
 */

/**
 * @typedef {Object} ValidationResult
 * @property {boolean} available
 * @property {{ rmse: number, mae: number, correlation: number }|null} metrics
 * @property {string}  units
 * @property {string}  reference
 * @property {Object}  evaluated_area
 */

/**
 * @typedef {Object} ErrorMapResult
 * @property {string}  url
 * @property {string}  units
 * @property {number}  min_error
 * @property {number}  max_error
 */

/**
 * @typedef {Object} ApiError
 * @property {string}  code
 * @property {string}  message
 * @property {boolean} recoverable
 * @property {any}     details
 */

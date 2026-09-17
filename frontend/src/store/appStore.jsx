/**
 * DepthWizard — App State Machine
 * 
 * Named states drive every routing and control-guard decision.
 * See DECISIONS.md §D02 and spec §37, §73.
 */
import { createContext, useContext, useReducer, useMemo } from 'react';

/** @type {Record<string, string>} */
export const AppState = {
  NO_SCENE:       'NO_SCENE',
  UPLOADING:      'UPLOADING',
  SCENE_READY:    'SCENE_READY',
  PROCESSING:     'PROCESSING',
  RESULTS_READY:  'RESULTS_READY',
  TERRAIN_LOADING:'TERRAIN_LOADING',
  TERRAIN_READY:  'TERRAIN_READY',
  ANALYSIS:       'ANALYSIS',
  FAILED:         'FAILED',
};

/** @type {Record<string, string>} */
const Action = {
  START_UPLOAD:       'START_UPLOAD',
  UPLOAD_SUCCESS:     'UPLOAD_SUCCESS',
  UPLOAD_BATCH:       'UPLOAD_BATCH',
  UPLOAD_FAIL:        'UPLOAD_FAIL',
  SWITCH_SCENE:       'SWITCH_SCENE',
  START_PROCESSING:   'START_PROCESSING',
  PROCESSING_UPDATE:  'PROCESSING_UPDATE',
  PROCESSING_DONE:    'PROCESSING_DONE',
  PROCESSING_FAIL:    'PROCESSING_FAIL',
  START_TERRAIN_LOAD: 'START_TERRAIN_LOAD',
  TERRAIN_READY:      'TERRAIN_READY',
  TERRAIN_FAIL:       'TERRAIN_FAIL',
  ENTER_ANALYSIS:     'ENTER_ANALYSIS',
  RETRY:              'RETRY',
  FALLBACK_RELATIVE:  'FALLBACK_RELATIVE',
  SET_ERROR:          'SET_ERROR',
  RESET:              'RESET',
  RESUME_SESSION:     'RESUME_SESSION',
  REMOVE_RECENT_PROJECT: 'REMOVE_RECENT_PROJECT',
  CLEAR_RECENT_PROJECTS: 'CLEAR_RECENT_PROJECTS',
};

/**
 * @typedef {Object} SceneInfo
 * @property {string} scene_id
 * @property {string} filename
 * @property {string} format
 * @property {number} width
 * @property {number} height
 * @property {boolean} georeferenced
 * @property {string|null} crs
 * @property {string} processing_path  - 'absolute_dsm' | 'relative_dsm'
 * @property {boolean} reference_available
 */

/**
 * @typedef {Object} JobStatus
 * @property {string} job_id
 * @property {string} scene_id
 * @property {string} status
 * @property {string} stage
 * @property {number} progress
 * @property {number|null} current_tile
 * @property {number|null} total_tiles
 * @property {string|null} message
 */

/**
 * @typedef {Object} AppError
 * @property {string} code
 * @property {string} message
 * @property {boolean} recoverable
 * @property {any} [details]
 */

/**
 * @typedef {Object} AppStoreState
 * @property {string} status - one of AppState values
 * @property {SceneInfo|null} scene
 * @property {string|null} jobId
 * @property {JobStatus|null} job
 * @property {Object|null} results  - from GET /scenes/{id}/results
 * @property {Object|null} terrain  - from GET /scenes/{id}/terrain
 * @property {Object|null} validation
 * @property {AppError|null} error
 * @property {SceneInfo[]} sceneQueue - current multi-upload batch (session-only)
 * @property {Object[]} recentScenes
 */

/** @type {AppStoreState} */
const initialState = {
  status: AppState.NO_SCENE,
  scene: null,
  jobId: null,
  job: null,
  results: null,
  terrain: null,
  validation: null,
  error: null,
  // Current multi-upload batch, in upload order. Session-persisted so a
  // page refresh mid-batch keeps the switcher working (recentScenes
  // carries the completed results; see sessionStorage below). Adjacent
  // GeoTIFF tiles that merged into one mosaic scene never land here.
  sceneQueue: (() => {
    try {
      const stored = sessionStorage.getItem('dw_scene_queue');
      const parsed = stored ? JSON.parse(stored) : null;
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  })(),
  recentScenes: (() => {
    try {
      const stored = localStorage.getItem('dw_recent');
      if (stored) return JSON.parse(stored);
      // Pre-seed sample projects as illustrated in spec §28:
      const sample = [
        {
          scene_id: 'scene_042',
          filename: 'Scene_042.tif',
          format: 'GeoTIFF',
          processing_path: 'absolute_dsm',
          elevation_mode: 'absolute',
          status: 'completed',
          ts: Date.now() - 2 * 60 * 1000, // 2 min ago
          results: {
            scene_id: 'scene_042',
            elevation_mode: 'absolute',
            units: 'm',
            reference_source: 'SRTM GL1 30m',
            reference_dem_available: true,
            min_elevation: 142.5,
            max_elevation: 846.2,
            outputs: ['rgb', 'depth', 'dsm', 'reference_dem', 'error', 'terrain'],
          },
        },
        {
          scene_id: 'hill_area',
          filename: 'Hill_Area.png',
          format: 'PNG',
          processing_path: 'relative_dsm',
          elevation_mode: 'relative',
          status: 'completed',
          ts: Date.now() - 24 * 60 * 60 * 1000, // yesterday
          results: {
            scene_id: 'hill_area',
            elevation_mode: 'relative',
            units: 'scene units',
            reference_source: null,
            reference_dem_available: false,
            min_elevation: 0.0,
            max_elevation: 1.0,
            outputs: ['rgb', 'depth', 'terrain'],
          },
        },
        {
          scene_id: 'urban_block',
          filename: 'Urban_Block.tif',
          format: 'GeoTIFF',
          processing_path: 'absolute_dsm',
          elevation_mode: 'absolute',
          status: 'completed',
          ts: Date.now() - 26 * 60 * 60 * 1000, // yesterday
          results: {
            scene_id: 'urban_block',
            elevation_mode: 'absolute',
            units: 'm',
            reference_source: 'ALOS AW3D30',
            reference_dem_available: true,
            min_elevation: 32.0,
            max_elevation: 118.4,
            outputs: ['rgb', 'depth', 'dsm', 'reference_dem', 'error', 'terrain'],
          },
        },
      ];
      try { localStorage.setItem('dw_recent', JSON.stringify(sample)); } catch {}
      return sample;
    } catch {
      return [];
    }
  })(),
};

/** @param {AppStoreState} state @param {{ type: string, payload?: any }} action */
function reducer(state, action) {
  switch (action.type) {
    case Action.START_UPLOAD:
      return { ...state, status: AppState.UPLOADING, error: null };

    case Action.UPLOAD_SUCCESS: {
      const scene = action.payload;
      const recent = [
        { scene_id: scene.scene_id, filename: scene.filename, processing_path: scene.processing_path, ts: Date.now() },
        ...state.recentScenes.filter(s => s.scene_id !== scene.scene_id),
      ].slice(0, 10);
      try { localStorage.setItem('dw_recent', JSON.stringify(recent)); } catch {}
      return { ...state, status: AppState.SCENE_READY, scene, error: null, recentScenes: recent };
    }

    case Action.UPLOAD_FAIL:
      return { ...state, status: AppState.FAILED, error: action.payload };

    case Action.UPLOAD_BATCH: {
      // Multi-file upload of NON-adjacent imagery: `scene` is the first /
      // active one, `queue` holds every uploaded scene in order. All are
      // registered in recents; the switcher cycles the queue.
      const { scene, queue } = action.payload;
      const now = Date.now();
      const batchEntries = queue.map(s => ({
        scene_id: s.scene_id,
        filename: s.filename,
        processing_path: s.processing_path,
        ts: now,
      }));
      const merged = [...batchEntries];
      for (const s of state.recentScenes) {
        if (!merged.some(m => m.scene_id === s.scene_id)) merged.push(s);
      }
      const recent = merged.slice(0, 10);
      try {
        localStorage.setItem('dw_recent', JSON.stringify(recent));
        sessionStorage.setItem('dw_scene_queue', JSON.stringify(queue));
      } catch {}
      return {
        ...state,
        status: AppState.SCENE_READY,
        scene,
        sceneQueue: queue,
        error: null,
        recentScenes: recent,
      };
    }

    case Action.SWITCH_SCENE: {
      // Jump to another scene from the current batch (terrain workspace
      // switcher). Completed scenes land in RESULTS_READY so the terrain
      // reloads immediately; unprocessed ones go back to SCENE_READY.
      const targetId = action.payload;
      if (targetId === state.scene?.scene_id) return state;
      const fromQueue = state.sceneQueue.find(s => s.scene_id === targetId);
      const fromRecent = state.recentScenes.find(s => s.scene_id === targetId);
      const source = fromQueue ?? fromRecent;
      if (!source) return state;
      const hasResults = !!source.results || !!fromRecent?.results;
      const results = source.results ?? fromRecent?.results ?? null;
      const scene = {
        scene_id: source.scene_id,
        filename: source.filename,
        format: source.format ?? (source.filename?.match(/\.tiff?$/i) ? 'GeoTIFF' : 'PNG'),
        width: source.width ?? 0,
        height: source.height ?? 0,
        georeferenced: source.processing_path === 'absolute_dsm' || source.elevation_mode === 'absolute',
        crs: source.crs ?? null,
        processing_path: source.processing_path ?? (source.elevation_mode === 'relative' ? 'relative_dsm' : 'absolute_dsm'),
        reference_available: !!results?.reference_dem_available,
      };
      return {
        ...state,
        status: hasResults ? AppState.RESULTS_READY : AppState.SCENE_READY,
        scene,
        results,
        terrain: null,
        validation: null,
        job: null,
        jobId: null,
        error: null,
      };
    }

    case Action.START_PROCESSING:
      return { ...state, status: AppState.PROCESSING, jobId: action.payload, job: null, error: null };

    case Action.PROCESSING_UPDATE:
      return { ...state, job: action.payload };

    case Action.PROCESSING_DONE: {
      const results = action.payload;
      const recent = state.recentScenes.map(s => {
        if (s.scene_id === state.scene?.scene_id) {
          return {
            ...s,
            status: 'completed',
            elevation_mode: results?.elevation_mode ?? (s.processing_path === 'absolute_dsm' ? 'absolute' : 'relative'),
            results,
            ts: Date.now(),
          };
        }
        return s;
      });
      try { localStorage.setItem('dw_recent', JSON.stringify(recent)); } catch {}
      return { ...state, status: AppState.RESULTS_READY, results: action.payload, error: null, recentScenes: recent };
    }

    case Action.PROCESSING_FAIL:
      return { ...state, status: AppState.FAILED, error: action.payload };

    case Action.START_TERRAIN_LOAD:
      return { ...state, status: AppState.TERRAIN_LOADING, error: null };

    case Action.TERRAIN_READY:
      return { ...state, status: AppState.TERRAIN_READY, terrain: action.payload, error: null };

    case Action.TERRAIN_FAIL:
      return { ...state, status: AppState.FAILED, error: action.payload };

    case Action.ENTER_ANALYSIS:
      return { ...state, status: AppState.ANALYSIS };

    case Action.SET_ERROR:
      return { ...state, error: action.payload };

    case Action.RETRY:
      return { ...state, status: AppState.SCENE_READY, error: null };

    case Action.FALLBACK_RELATIVE:
      return {
        ...state,
        status: AppState.SCENE_READY,
        scene: state.scene ? { ...state.scene, processing_path: 'relative_dsm' } : state.scene,
        error: null,
      };

    case Action.RESUME_SESSION: {
      const project = action.payload;
      const isAbsolute = project.elevation_mode === 'absolute' || project.processing_path === 'absolute_dsm';
      const scene = project.scene ?? {
        scene_id: project.scene_id,
        filename: project.filename,
        format: project.format ?? (project.filename?.endsWith('.tif') || project.filename?.endsWith('.tiff') ? 'GeoTIFF' : 'PNG'),
        width: project.width ?? 1024,
        height: project.height ?? 1024,
        georeferenced: isAbsolute,
        crs: isAbsolute ? 'EPSG:32643' : null,
        processing_path: isAbsolute ? 'absolute_dsm' : 'relative_dsm',
        reference_available: isAbsolute,
      };
      const results = project.results ?? {
        scene_id: project.scene_id,
        elevation_mode: isAbsolute ? 'absolute' : 'relative',
        units: isAbsolute ? 'm' : 'scene units',
        reference_source: isAbsolute ? 'SRTM' : null,
        reference_dem_available: isAbsolute,
        min_elevation: isAbsolute ? 100 : 0,
        max_elevation: isAbsolute ? 850 : 1,
        outputs: isAbsolute ? ['rgb', 'depth', 'dsm', 'reference_dem', 'error', 'terrain'] : ['rgb', 'depth', 'terrain'],
      };
      const targetStatus = project.targetStatus ?? (project.status === 'completed' || project.results ? AppState.RESULTS_READY : AppState.SCENE_READY);
      return {
        ...state,
        status: targetStatus,
        scene,
        results,
        error: null,
      };
    }

    case Action.REMOVE_RECENT_PROJECT: {
      const sceneId = action.payload;
      const recent = state.recentScenes.filter(s => s.scene_id !== sceneId);
      try { localStorage.setItem('dw_recent', JSON.stringify(recent)); } catch {}
      return { ...state, recentScenes: recent };
    }

    case Action.CLEAR_RECENT_PROJECTS: {
      try { localStorage.removeItem('dw_recent'); } catch {}
      return { ...state, recentScenes: [] };
    }

    case Action.RESET:
      try { sessionStorage.removeItem('dw_scene_queue'); } catch {}
      return { ...initialState, recentScenes: state.recentScenes, sceneQueue: [] };

    default:
      return state;
  }
}

const AppContext = createContext(null);

/** @param {{ children: React.ReactNode }} props */
export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Memoized so `actions` keeps a stable identity across renders — dispatch
  // itself is guaranteed stable by useReducer, so an empty dep array is safe.
  // Without this, a brand-new `actions` object was created on every render,
  // which broke any effect depending on `actions` (e.g. TerrainCanvas's
  // terrain-load effect), causing it to re-fire every render in an
  // infinite fetch → dispatch → re-render → re-fetch loop.
  const actions = useMemo(() => ({
    startUpload: () => dispatch({ type: Action.START_UPLOAD }),
    uploadSuccess: (scene) => dispatch({ type: Action.UPLOAD_SUCCESS, payload: scene }),
    uploadBatch: (scene, queue) => dispatch({ type: Action.UPLOAD_BATCH, payload: { scene, queue } }),
    uploadFail: (err) => dispatch({ type: Action.UPLOAD_FAIL, payload: err }),
    switchScene: (sceneId) => dispatch({ type: Action.SWITCH_SCENE, payload: sceneId }),
    startProcessing: (jobId) => dispatch({ type: Action.START_PROCESSING, payload: jobId }),
    processingUpdate: (job) => dispatch({ type: Action.PROCESSING_UPDATE, payload: job }),
    processingDone: (results) => dispatch({ type: Action.PROCESSING_DONE, payload: results }),
    processingFail: (err) => dispatch({ type: Action.PROCESSING_FAIL, payload: err }),
    startTerrainLoad: () => dispatch({ type: Action.START_TERRAIN_LOAD }),
    terrainReady: (terrain) => dispatch({ type: Action.TERRAIN_READY, payload: terrain }),
    terrainFail: (err) => dispatch({ type: Action.TERRAIN_FAIL, payload: err }),
    enterAnalysis: () => dispatch({ type: Action.ENTER_ANALYSIS }),
    setError: (err) => dispatch({ type: Action.SET_ERROR, payload: err }),
    retry: () => dispatch({ type: Action.RETRY }),
    fallbackRelative: () => dispatch({ type: Action.FALLBACK_RELATIVE }),
    reset: () => dispatch({ type: Action.RESET }),
    resumeSession: (project, targetStatus) => dispatch({
      type: Action.RESUME_SESSION,
      payload: { ...project, ...(targetStatus ? { targetStatus } : {}) },
    }),
    removeRecentProject: (sceneId) => dispatch({
      type: Action.REMOVE_RECENT_PROJECT,
      payload: sceneId,
    }),
    clearRecentProjects: () => dispatch({
      type: Action.CLEAR_RECENT_PROJECTS,
    }),
  }), [dispatch]);

  // Memoized so consumers relying on reference equality (e.g. effects that
  // depend on `state` or `actions` from useApp()) don't see a new object
  // identity unless state or actions actually changed.
  const value = useMemo(() => ({ state, actions }), [state, actions]);

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
}

/** @returns {{ state: AppStoreState, actions: Object }} */
export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}

/** True when terrain analysis tools are available */
export function isTerrainReady(status) {
  return status === AppState.TERRAIN_READY || status === AppState.ANALYSIS;
}
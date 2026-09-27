/**
 * DepthWizard — TerrainWorkspace page (Phase 4)
 *
 * Full-screen terrain viewer shell. Shown when state:
 *   TERRAIN_LOADING → (TerrainCanvas mounts → actions.terrainReady) → TERRAIN_READY → ANALYSIS
 *
 * Layout:
 *   - 48px Header (top)
 *   - Remaining viewport: TerrainCanvas (full-bleed)
 *   - 48px bottom Toolbar strip with TerrainControls + camera reset
 *
 * Phase 5 (camera system) and Phase 6 (minimap) will add more into this shell.
 * Phase 8 (layers), Phase 11 (full toolbar), etc. will also plug in here.
 *
 * DESIGN.md: Terrain primary — 70–80% usable screen; panels narrow + dark.
 */
import { useRef, useEffect, useState, useCallback } from 'react';
import Header from '../components/common/Header.jsx';
import TerrainCanvas from '../components/TerrainViewer/TerrainCanvas.jsx';
import Minimap from '../components/TerrainViewer/Minimap.jsx';
import SceneSwitcher from '../components/TerrainViewer/SceneSwitcher.jsx';
import CameraHUD from '../components/TerrainViewer/CameraHUD.jsx';
import ControlsHint from '../components/TerrainViewer/ControlsHint.jsx';
import Joystick from '../components/TerrainViewer/Joystick.jsx';
import WalkthroughPrompt from '../components/TerrainViewer/WalkthroughPrompt.jsx';
import ViewModeBar from '../components/TerrainViewer/ViewModeBar.jsx';
import TerrainLeftPanel from '../components/TerrainViewer/TerrainLeftPanel.jsx';
import TerrainRightPanel from '../components/TerrainViewer/TerrainRightPanel.jsx';
import StatusStrip from '../components/TerrainViewer/StatusStrip.jsx';
import { useFpsMeter } from '../hooks/useFpsMeter.js';
import { exportDsm, exportDepth, exportValidation, exportTerrain } from '../api/export.js';
import Toolbar from '../components/common/Toolbar.jsx';
import ElevationProbe from '../components/Analysis/ElevationProbe.jsx';
import HeightMeasurement from '../components/Analysis/HeightMeasurement.jsx';
import DistanceMeasurement from '../components/Analysis/DistanceMeasurement.jsx';
import SlopeMeasurement from '../components/Analysis/SlopeMeasurement.jsx';
import StructureInspector from '../components/Analysis/StructureInspector.jsx';
import RouteAssist from '../components/Analysis/RouteAssist.jsx';
import ToolGuard from '../components/Analysis/ToolGuard.jsx';
import RegionSelector from '../components/Analysis/RegionSelector.jsx';
import AnalysisPanel from '../components/common/AnalysisPanel.jsx';
import SemanticInspector from '../components/Analysis/SemanticInspector.jsx';
import { loadSemanticData } from '../lib/semanticSampler.js';
import { getSemanticMeta } from '../api/semantic.js';
import { useCameraController } from '../hooks/useCameraController.js';
import { getMinimap, getElevation } from '../api/terrain.js';
import { getResults, getDepth, getDsm, getReference, getScene } from '../api/results.js';
import { getValidation } from '../api/validation.js';
import { getErrorMap } from '../api/validation.js';
import { assessRoute, VERDICT_META } from '../api/route.js';
import { resolveAssetUrl } from '../api/client.js';
import { useApp, AppState } from '../store/appStore.jsx';
import {
  RotateCcw,
  Crosshair,
  ArrowUpDown,
  Ruler,
  TrendingUp,
  Building2,
  X,
} from 'lucide-react';

export default function TerrainWorkspace() {
  const { state } = useApp();
  const terrainRef = useRef(null);

  const isLoading = state.status === AppState.TERRAIN_LOADING;

  // ── Camera controller ──
  // viewportRef points to the terrain viewport container div — the camera
  // controller's resolveCanvas() will find the actual R3F <canvas> inside it.
  const viewportRef = useRef(null);
  const glPlaceholder = useRef({ current: {} });

  const {
    mode: cameraMode,
    setMode: setCameraMode,
    tickWalkthrough,
    setJoystickInput,
    setJoystickVertical,
  } = useCameraController({
    canvasRef: viewportRef,
    glRef: glPlaceholder,
  });

  // After TerrainCanvas mounts, connect the real glRef into the camera controller
  useEffect(() => {
    if (!terrainRef.current) return;
    const realGl = terrainRef.current.getRef?.();
    if (realGl) Object.assign(glPlaceholder, { current: realGl.current ?? realGl });
  });

  // Register walkthrough tick in canvas render loop
  useEffect(() => {
    terrainRef.current?.setFpTick(cameraMode === 'first-person' ? tickWalkthrough : null);
    terrainRef.current?.setCameraMode(cameraMode);
  }, [cameraMode, tickWalkthrough]);

  // ── Minimap metadata ──
  const [minimapMeta, setMinimapMeta] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);

  useEffect(() => {
    const sceneId = state.scene?.scene_id;
    if (!sceneId || isLoading) return;
    let cancelled = false;
    getMinimap(sceneId)
      .then(meta => { if (!cancelled) setMinimapMeta(meta); })
      .catch(() => { /* minimap is optional — silently skip */ });
    return () => { cancelled = true; };
  }, [state.scene?.scene_id, isLoading]);

  // Scene switches (batch switcher) must not carry the previous scene's
  // selections, measurements or layer texture cache into the new terrain.
  useEffect(() => {
    setSelectedPoint(null);
    setSelectedLocation(null);
    setSelectedStructure(null);
    setRefineBbox(null);
    setActiveTool('none');
    setActiveLayer('rgb');
    setViewTab('rgb');
    setWireframeState(false);
    setContourEnabled(false);
    layerCache.current = {};
    prevLayerRef.current = 'rgb';
    elevCache.current.clear();
    terrainRef.current?.setMeasurePoints?.(null);
  }, [state.scene?.scene_id]);

  // ── Layer system (Phase 8) ──
  // DEFAULT: Photorealistic satellite/aerial imagery draped over metric 3D elevation
  const [activeLayer, setActiveLayer] = useState('rgb');
  const [viewTab, setViewTab] = useState('rgb');
  // Cache of layer URL → { url, colormapMode } to avoid re-fetching
  const layerCache = useRef({});
  // Last layer that was successfully displayed — reverted to when a new
  // layer fails to load so the viewport never silently keeps a stale state.
  const prevLayerRef = useRef('rgb');

  // Colormap mode per layer (matches fragment shader uniforms)
  const COLORMAP_MODE = { rgb: 0, depth: 1, dsm: 2, reference_dem: 2, error: 3, slope: 2, buildings: 4, passability: 0, semantics: 0, route_risk: 0 };

  // ── Layer availability (Compare menu) — real backend state, not guesses ──
  const [layerAvail, setLayerAvail] = useState({ dsm: true, reference: true, error: true, semantics: true, route_risk: true });
  const [semanticData, setSemanticData] = useState(null);

  useEffect(() => {
    const sceneId = state.scene?.scene_id;
    if (!sceneId || isLoading) return;
    let cancelled = false;
    // Optimistic defaults keep the menu responsive; the fetches below then
    // disable exactly the products the scene is missing.
    setLayerAvail({ dsm: true, reference: true, error: true, semantics: true, route_risk: true });
    Promise.all([
      getResults(sceneId).catch(() => null),
      getReference(sceneId).catch(() => null),
      getSemanticMeta(sceneId).catch(() => null),
    ]).then(([results, reference, semMeta]) => {
      if (cancelled) return;
      const semAvail = !!semMeta?.available;
      setLayerAvail({
        dsm: !!(results?.dsm?.available ?? results?.dsm),
        reference: !!reference?.available,
        error: !!results?.capabilities?.validation,
        semantics: semAvail,
        route_risk: semAvail,
      });
    });
    return () => { cancelled = true; };
  }, [state.scene?.scene_id, isLoading]);

  // Load semantic segmentation arrays on scene change
  useEffect(() => {
    const sceneId = state.scene?.scene_id;
    if (!sceneId) return;
    let active = true;
    loadSemanticData(sceneId).then((data) => {
      if (!active) return;
      setSemanticData(data);
      if (data?.available && data?.labels && terrainRef.current) {
        terrainRef.current.updateSemanticTexture?.(data.labels, data.width, data.height);
      }
    });
    return () => { active = false; };
  }, [state.scene?.scene_id]);

  // ── Workspace chrome state (docked TerraLens-style panels) ──
  // Default to 1.0x true metric scale (1 Three.js world unit = 1 real meter)
  const [exaggeration, setExaggerationState] = useState(1.0);
  const [contourEnabled, setContourEnabled] = useState(false);
  const [contourInterval, setContourInterval] = useState(5);
  const [fog, setFogState] = useState(false);
  const [wireframe, setWireframeState] = useState(false);
  const [calibration, setCalibration] = useState(null);   // { min, max } applied display range
  const [dsmRange, setDsmRange] = useState(null);         // { min, max } from the scene's DSM
  const [depthStats, setDepthStats] = useState(null);     // statistics from /results
  const [validation, setValidation] = useState(null);
  const [validationLoading, setValidationLoading] = useState(false);
  const fps = useFpsMeter(!isLoading);
  // Walkthrough immersion: the exaggeration in force before Fly mode —
  // restored when returning to orbit/top so the overview never changes.
  const preWalkExagRef = useRef(null);
  // Side panels hidden for easy viewing (auto-hides in walkthrough mode;
  // the top-bar toggle brings them back / hides them any time).
  const [panelsHidden, setPanelsHidden] = useState(false);
  const prevModeRef = useRef(cameraMode);
  // User height-scale calibration: one known building height rescales all
  // estimated heights (monocular depth under-scales absolute heights).
  const [heightScale, setHeightScale] = useState(1);
  useEffect(() => {
    const sceneId = state.scene?.scene_id;
    if (!sceneId) return;
    try { setHeightScale(parseFloat(localStorage.getItem(`dw_hscale_${sceneId}`)) || 1); } catch {}
  }, [state.scene?.scene_id]);
  const handleHeightScaleChange = (f) => {
    const sceneId = state.scene?.scene_id;
    setHeightScale(f);
    try { if (sceneId) localStorage.setItem(`dw_hscale_${sceneId}`, String(f)); } catch {}
  };

  // DSM range (legend defaults) + depth statistics + validation metrics.
  // Statistics prefer the already-loaded results state; when the session
  // was resumed from a stub (no depth block yet), fetch them once.
  useEffect(() => {
    const sceneId = state.scene?.scene_id;
    if (!sceneId || isLoading) return;
    let cancelled = false;

    const applyStats = (stats) => {
      if (cancelled || !stats) return;
      setDepthStats(stats);
      if (typeof stats.minimum === 'number' && typeof stats.maximum === 'number') {
        setDsmRange({ min: stats.minimum, max: stats.maximum });
      }
    };

    const have = state.results?.depth?.statistics;
    // One retry: the Supabase token can be mid-refresh when the terrain
    // turns ready, and the panel would otherwise stay empty forever.
    if (have) {
      applyStats(have);
    } else {
      const fetchStats = (attempt = 0) => {
        getResults(sceneId)
          .then(r => applyStats(r?.depth?.statistics))
          .catch(() => { if (attempt < 1) setTimeout(() => fetchStats(1), 3000); });
      };
      fetchStats();
    }

    setValidationLoading(true);
    setValidation(null);
    const fetchValidation = (attempt = 0) => {
      getValidation(sceneId)
        .then(v => { if (!cancelled) { setValidation(v); setValidationLoading(false); } })
        .catch(() => {
          if (attempt < 1) setTimeout(() => fetchValidation(1), 3000);
          else if (!cancelled) setValidationLoading(false);
        });
    };
    fetchValidation();
    return () => { cancelled = true; };
  }, [state.scene?.scene_id, isLoading, state.results?.depth?.statistics]);

  /** View tabs: layer swaps + shader render modes over the current texture */
  const handleViewTab = (tabId) => {
    setViewTab(tabId);
    // Wireframe: a toggle tab — selecting it flips the flag; any other tab clears it
    const nextWireframe = tabId === 'wireframe' ? !wireframe : false;
    setWireframeState(nextWireframe);
    terrainRef.current?.setWireframe?.(nextWireframe);
    // Contour: selecting the tab enables lines at the current interval
    const nextContours = tabId === 'contour';
    setContourEnabled(nextContours);
    terrainRef.current?.setContours?.(nextContours, contourInterval);
    if (tabId === 'rgb' || tabId === 'depth' || tabId === 'dsm' || tabId === 'semantics') {
      handleLayerChange(tabId);
    } else if (tabId === 'hybrid') {
      handleLayerChange('solid');
    }
  };

  const handleExaggeration = (v) => {
    setExaggerationState(v);
    terrainRef.current?.setExaggeration?.(v);
  };

  // Entering walkthrough hides the docked panels for an unobstructed view;
  // returning to orbit/top brings them back (unless toggled manually).
  useEffect(() => {
    if (isLoading) return;
    if (prevModeRef.current !== cameraMode) {
      setPanelsHidden(cameraMode === 'first-person');
      prevModeRef.current = cameraMode;
    }
    // Guard against horizontal scroll drift: off-screen drawers create
    // scrollable overflow that clicks can scroll into view.
    if (viewportRef.current) viewportRef.current.scrollLeft = 0;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraMode, isLoading]);

  // Walkthrough mode respects user-configured scale
  useEffect(() => {
    if (isLoading) return;
  }, [cameraMode, isLoading]);

  const handleContour = (interval, enabled) => {
    setContourInterval(interval);
    setContourEnabled(enabled);
    terrainRef.current?.setContours?.(enabled, interval);
    if (enabled) {
      setViewTab('contour');
    } else if (viewTab === 'contour') {
      setViewTab('hybrid');
      handleLayerChange('solid');
    }
  };

  const handleFog = (v) => {
    setFogState(v);
    terrainRef.current?.setFog?.(v);
  };

  const handleWireframeToggle = (v) => {
    setWireframeState(v);
    terrainRef.current?.setWireframe?.(v);
    setViewTab(curr => {
      if (v) return 'wireframe';
      return curr === 'wireframe' ? 'hybrid' : curr;
    });
    if (!v && viewTab === 'wireframe') terrainRef.current?.setContours?.(contourEnabled, contourInterval);
  };

  const handleSlopeOverlay = (v) => {
    handleLayerChange(v ? 'slope' : 'rgb');
  };

  const handleApplyCalibration = (min, max) => {
    if (typeof min !== 'number' || typeof max !== 'number' || Number.isNaN(min) || Number.isNaN(max) || max <= min) {
      showToast('Enter a valid elevation range — max must be greater than min.');
      return;
    }
    setCalibration({ min, max });
    showToast(`Calibration applied — elevation display range set to ${min}–${max} m.`);
  };

  /** Export rows for the right panel — every enabled row hits a real endpoint */
  const buildDownloads = (sceneId) => [
    { id: 'depth_png', label: 'Depth PNG', sub: 'grayscale raster', run: () => exportDepth(sceneId) },
    { id: 'depth_npy', label: 'Depth NPY', sub: 'raw float array', disabled: true },
    { id: 'mesh', label: '3D Mesh (OBJ/GLB)', sub: 'textured scene mesh', run: () => exportTerrain(sceneId) },
    { id: 'dsm', label: 'DSM GeoTIFF', sub: 'georeferenced surface', run: () => exportDsm(sceneId) },
    { id: 'contour_png', label: 'Contour PNG', disabled: true },
    { id: 'metadata', label: 'Metadata JSON', sub: 'validation metrics', run: () => exportValidation(sceneId) },
  ];

  // ── Toast (non-blocking feedback when a tool/layer can't do its job) ──
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);
  const showToast = useCallback((message) => {
    setToast(message);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 4000);
  }, []);
  useEffect(() => () => { if (toastTimer.current) clearTimeout(toastTimer.current); }, []);

  /** Fetch the texture URL for a layer and swap the terrain texture (task 8.2) */
  async function handleLayerChange(layerId) {
    if (layerId === activeLayer) return;
    setActiveLayer(layerId);
    // Keep the view-tab highlight in sync with toolbar-driven layer swaps
    if (layerId === 'solid') setViewTab('hybrid');
    else if (layerId === 'rgb' || layerId === 'depth' || layerId === 'dsm' || layerId === 'semantics') setViewTab(layerId);

    const sceneId = state.scene?.scene_id;
    if (!sceneId) return;

    // Check cache first
    if (layerCache.current[layerId]) {
      const { url, colormapMode } = layerCache.current[layerId];
      terrainRef.current?.setLayerTexture(url, colormapMode);
      prevLayerRef.current = layerId;
      return;
    }

    const LAYER_LABEL = {
      rgb: 'RGB', depth: 'Depth map', dsm: 'Estimated DSM',
      reference_dem: 'Reference DEM', error: 'Error map', slope: 'Slope layer',
      passability: 'Passability map', semantics: 'Semantic map', route_risk: 'Route-risk map',
    };

    try {
      let url = null;
      const colormapMode = COLORMAP_MODE[layerId] ?? 0;

      if (layerId === 'solid') {
        // default view: solid shaded surface + mesh, no imagery
        terrainRef.current?.setSolidView();
        prevLayerRef.current = 'solid';
        return;
      }

      if (layerId === 'rgb') {
        // RGB uses the terrain texture (already loaded)
        const meta = state.terrain;
        url = meta?.texture_url ?? null;
      } else if (layerId === 'depth') {
        const depth = await getDepth(sceneId);
        url = depth?.url ?? depth?.download_url ?? null;
      } else if (layerId === 'dsm') {
        // `url` is the browser-renderable PNG preview; `download_url` is the
        // raw GeoTIFF/npy science product, which a WebGL texture can't decode.
        const dsm = await getDsm(sceneId);
        url = dsm?.url ?? null;
      } else if (layerId === 'semantics') {
        url = `/api/v1/scenes/${sceneId}/results/semantic`;
        const res = await fetch(resolveAssetUrl(url));
        if (!res.ok) throw new Error('semantic layer unavailable');
      } else if (layerId === 'route_risk') {
        url = `/api/v1/scenes/${sceneId}/results/route-risk?vehicle=fire_truck`;
        const res = await fetch(resolveAssetUrl(url));
        if (!res.ok) throw new Error('route-risk layer unavailable');
      } else if (layerId === 'error') {
        const errMap = await getErrorMap(sceneId);
        url = errMap?.url ?? null;
      } else if (layerId === 'reference_dem') {
        const refDem = await getReference(sceneId);
        url = refDem?.visualization_url ?? refDem?.download_url ?? null;
      } else if (layerId === 'slope') {
        // Generated on demand by the backend from the scene's DSM array.
        const slopeUrl = `/api/v1/scenes/${sceneId}/results/slope`;
        const res = await fetch(resolveAssetUrl(slopeUrl));
        if (!res.ok) throw new Error('slope preview unavailable');
        url = slopeUrl;
      } else if (layerId === 'passability') {
        // Route Assist heat map (jury round 2): traffic-light vehicle
        // passability PNG — already colored, so colormapMode stays RGB.
        url = `/api/v1/scenes/${sceneId}/results/passability?vehicle=fire_truck`;
        const res = await fetch(resolveAssetUrl(url));
        if (!res.ok) throw new Error('passability layer unavailable');
      } else {
        // Other layers: try results endpoint for URL
        const results = await getResults(sceneId);
        url = results?.[layerId + '_url'] ?? results?.layers?.[layerId] ?? null;
      }

      if (url) {
        layerCache.current[layerId] = { url, colormapMode };
        terrainRef.current?.setLayerTexture(url, colormapMode);
        prevLayerRef.current = layerId;
      } else {
        setActiveLayer(prevLayerRef.current);
        showToast(`${LAYER_LABEL[layerId] ?? 'That layer'} is not available for this scene yet.`);
      }
    } catch {
      setActiveLayer(prevLayerRef.current);
      showToast(`${LAYER_LABEL[layerId] ?? 'That layer'} could not be loaded for this scene.`);
    }
  }

  // ── Analysis tools state (Phase 9, 10 & 13) ──
  const [activeTool, setActiveTool] = useState('none');
  const heightToolRef = useRef(null);
  const distToolRef   = useRef(null);
  const slopeToolRef  = useRef(null);
  const structToolRef = useRef(null);
  const routeToolRef  = useRef(null);
  const routeStartRef = useRef(null);
  const semanticInspectorRef = useRef(null);

  // ── Detail Mode Refinement state (Phase 13, §18, §65) ──
  const [refineBbox, setRefineBbox]               = useState(null);
  const [isSelectingRegion, setIsSelectingRegion] = useState(false);

  const handleSelectTool = (tool) => {
    setActiveTool(curr => {
      const next = curr === tool ? 'none' : tool;
      if (next === 'refine') {
        setIsSelectingRegion(true);
        setAnalysisPanelOpen(true);
      }
      // Leaving the distance tool removes its A/B pins + label from the mesh.
      if (curr === 'distance' && next !== 'distance') {
        terrainRef.current?.setMeasurePoints?.(null);
      }
      // Leaving Route Assist clears the path overlay and the pick state.
      if (curr === 'route' && next !== 'route') {
        terrainRef.current?.setRoutePath?.(null);
        routeToolRef.current?.resetPicks?.();
        routeStartRef.current = null;
      }
      // Leaving Inspect / probe mode clears semantic highlight
      if (curr === 'probe' && next !== 'probe') {
        terrainRef.current?.setSemanticHighlightClass?.(-1);
        semanticInspectorRef.current?.reset?.();
      }
      return next;
    });
  };

  // ── Side Analysis Panel state (Phase 10, 12, 14) ──
  const [analysisPanelOpen, setAnalysisPanelOpen] = useState(false);
  const [analysisPanelTab, setAnalysisPanelTab]   = useState('overview');
  const [selectedLocation, setSelectedLocation]   = useState(null);
  const [selectedStructure, setSelectedStructure] = useState(null);
  const [scenario, setScenario]                   = useState('exploration');

  /** Handle clicks on the terrain canvas to feed active measurement tool */
  const elevCache = useRef(new Map());

  /** Upgrade a visual heightmap estimate to the METERED DSM value (exact
   *  float32 sample from the backend) with its accuracy statement. Points
   *  without pixel coords, or when the backend is unavailable, fall back
   *  to the visual estimate unchanged. */
  const resolveExactPoint = useCallback(async (pt) => {
    const sceneId = state.scene?.scene_id;
    const g = terrainRef.current?.getRef?.()?.current;
    if (!sceneId || !pt || typeof pt.u !== 'number' || !g?.hmWidth) return pt;
    const px = Math.min(g.hmWidth - 1, Math.max(0, Math.round(pt.u * (g.hmWidth - 1))));
    const py = Math.min(g.hmHeight - 1, Math.max(0, Math.round(pt.v * (g.hmHeight - 1))));
    const key = `${px},${py}`;
    const cached = elevCache.current.get(key);
    if (cached) return { ...pt, ...cached };
    try {
      const r = await getElevation(sceneId, px, py);
      const patch = {
        elevation: r.elevation ?? pt.elevation,
        px,
        py,
        metered: !!r.metered,
        precision_m: r.precision_m ?? null,
        confidence: r.confidence ?? null,
        ground_elevation: r.ground_elevation ?? null,
        height_above_ground_m: r.height_above_ground_m ?? null,
        is_structure: !!r.is_structure,
        height_confidence: r.height_confidence ?? null,
      };
      elevCache.current.set(key, patch);
      return { ...pt, ...patch };
    } catch {
      return pt;
    }
  }, [state.scene?.scene_id]);

  const handleTerrainClick = async (e) => {
    if (isLoading) return;
    // In Walkthrough, canvas clicks capture the cursor for mouse look —
    // measurement picks stay in Orbit / Top View where clicking makes sense.
    if (cameraMode === 'first-person') return;
    const rawPt = terrainRef.current?.getTerrainPointFromEvent?.(e);
    if (!rawPt) return;
    // Metered elevation (exact DSM sample + confidence) replaces the
    // heightmap estimate before any tool or panel consumes the point.
    const pt = await resolveExactPoint(rawPt);

    setSelectedPoint({ x: pt.x, z: pt.z, elevation: pt.elevation });

    // Update selectedLocation for context switching (§25)
    setSelectedLocation({
      x: pt.x,
      z: pt.z,
      elevation: pt.elevation,
    });

    if (activeTool === 'height') {
      heightToolRef.current?.handleTerrainClick(pt);
    } else if (activeTool === 'distance') {
      distToolRef.current?.handleSelectPoint(pt);
    } else if (activeTool === 'slope') {
      slopeToolRef.current?.handleSelectPoint(pt);
    } else if (activeTool === 'route') {
      // Route Assist: picks are in SOURCE-RASTER pixel coords (the API's
      // coordinate space), derived from the normalized pick point.
      const rasterW = state.scene?.width ?? 1024;
      const rasterH = state.scene?.height ?? 1024;
      const pixel = {
        x: Math.min(Math.max(Math.round((rawPt.u ?? 0.5) * (rasterW - 1)), 0), rasterW - 1),
        y: Math.min(Math.max(Math.round((rawPt.v ?? 0.5) * (rasterH - 1)), 0), rasterH - 1),
      };
      const picked = routeToolRef.current?.handleTerrainClick(pixel);
      if (picked?.picked === 'start') {
        routeStartRef.current = pixel;
        terrainRef.current?.setRoutePath?.([pixel, pixel], '#4f8cff');
      } else if (picked?.picked === 'end') {
        const startPx = routeStartRef.current ?? pixel;
        const vehicles = routeToolRef.current?.getVehicles?.() ?? ['fire_truck'];
        routeToolRef.current?.setAssessing?.();
        assessRoute(state.scene.scene_id, startPx, pixel, vehicles)
          .then((response) => {
            routeToolRef.current?.setAssessment?.(response);
            // Draw the first vehicle that produced a route; color by verdict.
            const routed = response.vehicles?.find((v) => v.path?.length);
            if (routed) {
              const color =
                VERDICT_META[routed.verdict]?.color ?? '#2ecc71';
              terrainRef.current?.setRoutePath?.(routed.path, color, {
                // Aerial vehicles touch down on the landing zone, not the
                // destination itself — label the endpoint accordingly.
                endLabel: routed.landing_zone ? 'LZ' : 'END',
              });
            } else {
              // no route anywhere — leave the pins, drop the line
              terrainRef.current?.setRoutePath?.(null);
            }
          })
          .catch((err) => {
            routeToolRef.current?.setAssessError?.(
              err?.message ?? 'Route assessment failed.',
            );
          });
      }
    } else if (activeTool === 'structure') {
      // Absolute rebasing: when the user calibrates the display range
      // (DEM min/max), metered elevations shift by the same offset so
      // readouts are absolute against the entered datum.
      pt.calibrated = calibration && dsmRange
        ? {
            ground: (pt.ground_elevation ?? 0) + (calibration.min - dsmRange.min),
            top: pt.elevation + (calibration.min - dsmRange.min),
          }
        : null;
      structToolRef.current?.inspectPoint(pt);
      const idNum = Math.abs(Math.round(pt.x * 100 + pt.z * 100)) % 999;
      setSelectedStructure({
        id: `STR-${String(idNum).padStart(3, '0')}`,
        // Metered DSM values from the backend — ground level, roof sample
        // and geometric height above ground, with the honest confidence.
        ground: pt.ground_elevation ?? pt.elevation,
        top: pt.elevation,
        height: pt.height_above_ground_m ?? 0,
        isStructure: !!pt.is_structure,
        confidence: pt.height_confidence ?? pt.confidence ?? null,
      });
    } else if (activeTool === 'probe') {
      semanticInspectorRef.current?.handleTerrainClick?.(rawPt);
    }
  };

  const handleViewportMouseMove = useCallback((e) => {
    if (activeTool === 'probe' && semanticInspectorRef.current) {
      const pt = terrainRef.current?.getTerrainPointFromEvent?.(e);
      if (pt) {
        semanticInspectorRef.current.handleTerrainHover?.(pt);
      }
    }
  }, [activeTool]);

  const handleViewportMouseLeave = useCallback(() => {
    if (activeTool === 'probe' && semanticInspectorRef.current) {
      semanticInspectorRef.current.handleTerrainHover?.(null);
    }
  }, [activeTool]);

  /** Capture client-side viewport snapshot from OGL canvas (task 15.1, 15.3) */
  const handleCaptureSnapshot = () => {
    try {
      const dataUrl = terrainRef.current?.captureSnapshot?.() ??
        terrainRef.current?.getCanvas?.()?.current?.toDataURL('image/png');
      if (!dataUrl) return;
      const a = document.createElement('a');
      a.href = dataUrl;
      const sceneName = state.scene?.filename?.replace(/\.[^/.]+$/, '') ?? 'terrain';
      a.download = `${sceneName}-snapshot.png`;
      a.click();
    } catch (err) {
      console.error('[TerrainWorkspace] Snapshot capture failed', err);
    }
  };

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--dw-void)',
      overflow: 'hidden',
    }}>
      {/* 48px Header */}
      <Header />

      {/* Terrain viewport — fills remaining space */}
      <div
        ref={viewportRef}
        onClick={handleTerrainClick}
        onMouseMove={handleViewportMouseMove}
        onMouseLeave={handleViewportMouseLeave}
        style={{
          flex: 1,
          position: 'relative',
          overflow: 'hidden',
          cursor: activeTool !== 'none' ? 'crosshair' : 'default',
        }}
      >
        {/* Full-bleed OGL canvas */}
        <TerrainCanvas ref={terrainRef} />

        {/* Docked left panel — reference + quality */}
        {!isLoading && (
          <TerrainLeftPanel
            hidden={panelsHidden}
            dsmRange={dsmRange}
            calibration={calibration ?? dsmRange}
            onApplyCalibration={handleApplyCalibration}
            validation={validation}
            validationLoading={validationLoading}
            depthStats={depthStats}
          />
        )}

        {/* Minimap overlay — top-left of the live viewport (RGB drape) */}
        {!isLoading && (
          <Minimap
            terrainRef={terrainRef}
            cameraMode={cameraMode}
            minimapMeta={minimapMeta}
            terrainMeta={state.terrain}
            selectedPoint={selectedPoint}
            leftOffset={292}
            rgbUrl={state.terrain?.texture_url ?? null}
          />
        )}

        {/* Top in-viewport control bar — view tabs, camera, interaction mode */}
        {!isLoading && (
          <ViewModeBar
            viewTab={viewTab}
            onViewTab={handleViewTab}
            cameraMode={cameraMode}
            onCameraMode={(m) => {
              terrainRef.current?.setCameraMode?.(m);
              setCameraMode(m);
            }}
            toolMode={
              activeTool === 'probe' ? 'inspect'
                : activeTool === 'none' ? 'navigate'
                  : 'measure'
            }
            onToolMode={(m) => {
              if (m === 'inspect') handleSelectTool('probe');
              else if (m === 'measure') handleSelectTool('distance');
              else if (activeTool !== 'none') handleSelectTool(activeTool);
            }}
            disabled={false}
            analysisOpen={analysisPanelOpen}
            onToggleAnalysis={() => setAnalysisPanelOpen(v => !v)}
            panelsHidden={panelsHidden}
            onTogglePanels={() => setPanelsHidden(v => !v)}
          />
        )}

        {/* Minimap moved into the left panel; this overlay only shows when
            the panel is closed — keep as expand fallback is unnecessary */}
        {/* Docked right panel — terrain controls, downloads, legends */}
        {!isLoading && (
          <TerrainRightPanel
            hidden={panelsHidden}
            exaggeration={exaggeration}
            onExaggeration={handleExaggeration}
            contourEnabled={contourEnabled}
            contourInterval={contourInterval}
            onContour={handleContour}
            fog={fog}
            onFog={handleFog}
            wireframe={wireframe}
            onWireframe={handleWireframeToggle}
            slopeOverlay={activeLayer === 'slope'}
            onSlopeOverlay={handleSlopeOverlay}
            downloads={buildDownloads(state.scene?.scene_id)}
            legendRange={calibration ?? dsmRange}
          />
        )}

        {/* Multi-image batch switcher — top-centre; arrows cycle, numbered
            chips jump. Only appears when a batch of separate scenes exists. */}
        {!isLoading && (
          <SceneSwitcher disabled={isLoading} />
        )}

        {/* Elevation Probe (task 9.1) — paused during Walkthrough, where the
            cursor is captured and hover sampling has no meaningful target */}
        {!isLoading && (
          <ElevationProbe
            terrainRef={terrainRef}
            enabled={(activeTool === 'probe' || activeTool === 'none') && cameraMode !== 'first-person'}
            semanticData={semanticData}
          />
        )}

        {/* Semantic Inspector overlay — active during Inspect Mode */}
        {!isLoading && activeTool === 'probe' && semanticData?.available && (
          <SemanticInspector
            ref={semanticInspectorRef}
            terrainRef={terrainRef}
            semanticData={semanticData}
            active={activeTool === 'probe'}
            onClose={() => handleSelectTool('none')}
          />
        )}

        {/* Navigation HUD — bottom-right, walkthrough mode only (Phase 7) */}
        {!isLoading && (
          <CameraHUD
            terrainRef={terrainRef}
            cameraMode={cameraMode}
            elevationMode={state.results?.elevation_mode ?? 'relative'}
            rightOffset={296}
          />
        )}

        {/* Walkthrough entry cue — click-to-capture hint, hidden once locked */}
        {!isLoading && (
          <WalkthroughPrompt cameraMode={cameraMode} />
        )}

        {/* Touch joystick — walkthrough only; feeds the SAME movement code
            path as the keyboard (setJoystickInput merges in the tick) */}
        {!isLoading && cameraMode === 'first-person' && (
          <Joystick onMove={setJoystickInput} onVertical={setJoystickVertical} />
        )}

        {/* 3D Viewport Controls Guide (bottom-left) */}
        {!isLoading && (
          <ControlsHint cameraMode={cameraMode} leftOffset={296} />
        )}

        {/* Active Analysis tool readout panel (Phase 9, tasks 9.2-9.6) */}        {!isLoading && activeTool !== 'none' && activeTool !== 'probe' && (
          <div style={{
            position: 'absolute',
            top: 56,
            right: 12,
            maxWidth: 'calc(100vw - 24px)',
            transform: `translateX(-${analysisPanelOpen ? 'min(320px, calc(100vw - 40px))' : '296px'})`,
            zIndex: 12,
            transition: 'transform 200ms ease-out',
          }}>
            <ToolGuard>
              {activeTool === 'height' && (
                <HeightMeasurement ref={heightToolRef} active={true} />
              )}
              {activeTool === 'distance' && (
                <DistanceMeasurement
                  ref={distToolRef}
                  active={true}
                  onMeasureChange={(a, b, label) => {
                    terrainRef.current?.setMeasurePoints?.(a, b, label);
                  }}
                />
              )}
              {activeTool === 'slope' && (
                <SlopeMeasurement ref={slopeToolRef} active={true} />
              )}
              {activeTool === 'structure' && (
                <StructureInspector
                  ref={structToolRef}
                  terrainRef={terrainRef}
                  active={true}
                  selectedPoint={selectedPoint}
                  onClear={() => setSelectedPoint(null)}
                  heightScale={heightScale}
                  onHeightScaleChange={handleHeightScaleChange}
                />
              )}
              {activeTool === 'route' && (
                <RouteAssist ref={routeToolRef} active={true} />
              )}
            </ToolGuard>
          </div>
        )}

        {/* Rubber-band region selector for small-structure detail refinement (§18, §65) */}
        {!isLoading && (
          <RegionSelector
            active={isSelectingRegion || activeTool === 'refine'}
            selectedBbox={refineBbox}
            onBboxChange={(bbox) => {
              setRefineBbox(bbox);
              setIsSelectingRegion(false);
              setAnalysisPanelOpen(true);
            }}
            onCompleteSelection={() => setIsSelectingRegion(false)}
          />
        )}

        {/* Side Analysis Panel — collapsible 320px right drawer (Phase 10 & 12, §25, §16) */}
        {!isLoading && (
          <AnalysisPanel
            open={analysisPanelOpen}
            rightOffset={panelsHidden ? 0 : 280}
            onToggle={() => setAnalysisPanelOpen(v => !v)}
            selectedLocation={selectedLocation}
            selectedStructure={selectedStructure}
            onClearSelection={() => {
              setSelectedLocation(null);
              setSelectedStructure(null);
              setSelectedPoint(null);
              setRefineBbox(null);
            }}
            activeLayer={activeLayer}
            onSelectLayer={handleLayerChange}
            panelTab={analysisPanelTab}
            onSelectTab={setAnalysisPanelTab}
            activeTool={activeTool}
            refineBbox={refineBbox}
            onStartRegionSelect={() => setIsSelectingRegion(true)}
            onClearRefineBbox={() => setRefineBbox(null)}
            isSelectingRegion={isSelectingRegion}
            onRefineComplete={() => {
              layerCache.current = {};
              handleLayerChange(activeLayer);
            }}
            scenario={scenario}
            onSelectScenario={setScenario}
          />
        )}

        {/* Loading overlay — while TERRAIN_LOADING */}
        {isLoading && (
          <div
            role="status"
            aria-label="Loading terrain"
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(9,9,11,0.7)',
              gap: 12,
            }}
          >
            <TerrainLoadingIndicator />
          </div>
        )}

        {/* Toast — honest feedback when a layer/tool has nothing to show */}
        {toast && (
          <div
            role="status"
            aria-live="polite"
            style={{
              position: 'absolute',
              bottom: 16,
              left: '50%',
              transform: 'translateX(-50%)',
              maxWidth: 'min(480px, calc(100vw - 32px))',
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-rim)',
              borderLeft: '2px solid var(--dw-accent)',
              borderRadius: 'var(--dw-radius-sm)',
              padding: '10px 14px',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              lineHeight: 1.45,
              color: 'var(--dw-fg)',
              zIndex: 40,
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.35)',
            }}
          >
            {toast}
          </div>
        )}
      </div>

      {/* 56px unified bottom toolbar (Phase 11, §26) */}
      <Toolbar
        terrainRef={terrainRef}
        cameraMode={cameraMode}
        onSetCameraMode={setCameraMode}
        activeLayer={activeLayer}
        onSelectLayer={handleLayerChange}
        activeTool={activeTool}
        onSelectTool={handleSelectTool}
        disabled={isLoading}
        layerAvailability={layerAvail}
        onOpenValidation={() => {
          setAnalysisPanelOpen(true);
          setAnalysisPanelTab('validation');
        }}
        onCaptureSnapshot={handleCaptureSnapshot}
      />

      {/* 28px telemetry strip — picked point, FPS, mesh stats, geospatial HUD toggle */}
      <StatusStrip
        selectedPoint={selectedPoint}
        fps={fps}
        terrainMeta={state.terrain}
        exaggeration={exaggeration}
        onToggleDebugHud={() => terrainRef.current?.toggleDebugHUD?.()}
      />
    </div>
  );
}

/**
 * Terrain-specific loading indicator.
 * Shows a minimalist pulse — not a generic spinner.
 * DESIGN.md: "Generic AI spinner — always show the actual pipeline stage."
 */
function TerrainLoadingIndicator() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
      {/* Terrain profile glyph — animated */}
      <svg
        width="48"
        height="24"
        viewBox="0 0 40 20"
        fill="none"
        aria-hidden="true"
        style={{ animation: 'dw-terrain-pulse 1.8s ease-in-out infinite' }}
      >
        <style>{`
          @keyframes dw-terrain-pulse {
            0%, 100% { opacity: 0.3; }
            50%       { opacity: 0.9; }
          }
          @media (prefers-reduced-motion: reduce) {
            @keyframes dw-terrain-pulse { 0%, 100% { opacity: 0.6; } }
          }
        `}</style>
        <polyline
          points="2,18 8,10 14,14 22,4 30,8 38,6"
          stroke="var(--dw-accent)"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <line
          x1="2" y1="18" x2="38" y2="18"
          stroke="var(--dw-rim)"
          strokeWidth="1"
          strokeLinecap="round"
        />
      </svg>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 13,
        fontWeight: 500,
        color: 'var(--dw-fg)',
        letterSpacing: '0.06em',
      }}>
        BUILDING TERRAIN
      </span>
    </div>
  );
}

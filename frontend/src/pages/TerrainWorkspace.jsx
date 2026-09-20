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
import LayerControl, { LAYER_META } from '../components/TerrainViewer/LayerControl.jsx';
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
import { useCameraController } from '../hooks/useCameraController.js';
import { getMinimap, getElevation } from '../api/terrain.js';
import { getResults, getDepth, getDsm, getReference } from '../api/results.js';
import { getErrorMap } from '../api/validation.js';
import { assessRoute, VERDICT_META } from '../api/route.js';
import { resolveAssetUrl } from '../api/client.js';
import { useApp, AppState } from '../store/appStore.jsx';
import {
  RotateCcw,
  Layers,
  Crosshair,
  ArrowUpDown,
  Ruler,
  TrendingUp,
  Building2,
  PanelRight,
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
    setActiveLayer('buildings');
    layerCache.current = {};
    prevLayerRef.current = 'buildings';
    elevCache.current.clear();
    terrainRef.current?.setMeasurePoints?.(null);
  }, [state.scene?.scene_id]);

  // ── Layer system (Phase 8) ──
  // DEFAULT: solid-colour building blocks — structures render as flat-
  // coloured volumes instead of the RGB drape; RGB stays one click away.
  const [activeLayer, setActiveLayer] = useState('buildings');
  const [layerPanelOpen, setLayerPanelOpen] = useState(false);
  // Cache of layer URL → { url, colormapMode } to avoid re-fetching
  const layerCache = useRef({});
  // Last layer that was successfully displayed — reverted to when a new
  // layer fails to load so the viewport never silently keeps a stale state.
  const prevLayerRef = useRef('solid');

  // Colormap mode per layer (matches fragment shader uniforms)
  const COLORMAP_MODE = { rgb: 0, depth: 1, dsm: 2, reference_dem: 2, error: 3, slope: 2, buildings: 4, passability: 0 };

  // ── Layer availability (Compare menu) — real backend state, not guesses ──
  const [layerAvail, setLayerAvail] = useState({ dsm: true, reference: true, error: true });
  useEffect(() => {
    const sceneId = state.scene?.scene_id;
    if (!sceneId || isLoading) return;
    let cancelled = false;
    // Optimistic defaults keep the menu responsive; the fetches below then
    // disable exactly the products the scene is missing.
    setLayerAvail({ dsm: true, reference: true, error: true });
    Promise.all([
      getResults(sceneId).catch(() => null),
      getReference(sceneId).catch(() => null),
    ]).then(([results, reference]) => {
      if (cancelled) return;
      setLayerAvail({
        dsm: !!(results?.dsm?.available ?? results?.dsm),
        reference: !!reference?.available,
        error: !!results?.capabilities?.validation,
      });
    });
    return () => { cancelled = true; };
  }, [state.scene?.scene_id, isLoading]);

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

    const sceneId = state.scene?.scene_id;
    if (!sceneId) return;

    // Buildings view is generated in-shader from the heightmap — no texture
    if (layerId === 'buildings') {
      terrainRef.current?.setBuildingsView?.();
      prevLayerRef.current = 'buildings';
      return;
    }

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
      passability: 'Passability map',
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
              terrainRef.current?.setRoutePath?.(routed.path, color);
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
      structToolRef.current?.inspectPoint(pt);
      const idNum = Math.abs(Math.round(pt.x * 100 + pt.z * 100)) % 999;
      setSelectedStructure({
        id: `STR-${String(idNum).padStart(3, '0')}`,
        ground: pt.elevation,
        top: pt.elevation * 1.15,
        height: pt.elevation * 0.15,
      });
    }
  };

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
        style={{
          flex: 1,
          position: 'relative',
          overflow: 'hidden',
          cursor: activeTool !== 'none' ? 'crosshair' : 'default',
        }}
      >
        {/* Full-bleed OGL canvas */}
        <TerrainCanvas ref={terrainRef} />

        {/* Minimap overlay — top-left of terrain viewport (task 6.1) */}
        {!isLoading && (
          <Minimap
            terrainRef={terrainRef}
            cameraMode={cameraMode}
            minimapMeta={minimapMeta}
            terrainMeta={state.terrain}
            selectedPoint={selectedPoint}
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
          />
        )}

        {/* Navigation HUD — bottom-right, walkthrough mode only (Phase 7) */}
        {!isLoading && (
          <CameraHUD
            terrainRef={terrainRef}
            cameraMode={cameraMode}
            elevationMode={state.results?.elevation_mode ?? 'relative'}
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
          <ControlsHint cameraMode={cameraMode} />
        )}

        {/* Active Analysis tool readout panel (Phase 9, tasks 9.2-9.6) */}        {!isLoading && activeTool !== 'none' && activeTool !== 'probe' && (
          <div style={{
            position: 'absolute',
            top: 56,
            right: 12,
            maxWidth: 'calc(100vw - 24px)',
            transform: `translateX(-${analysisPanelOpen ? 'min(320px, calc(100vw - 40px))' : (layerPanelOpen ? 'min(240px, calc(100vw - 40px))' : '0px')})`,
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

        {/* Layer panel — collapsible right overlay (Phase 8, Phase 17) */}
        {!isLoading && (
          <div style={{
            position: 'absolute',
            top: 0,
            right: 0,
            bottom: 0,
            width: 'min(260px, 100vw)',
            transform: layerPanelOpen ? 'translateX(0)' : 'translateX(100%)',
            transition: 'transform 200ms ease-out',
            zIndex: 14,
            pointerEvents: layerPanelOpen ? 'auto' : 'none',
          }}>
            <div style={{
              width: '100%',
              height: '100%',
              background: 'var(--dw-panel)',
              borderLeft: '1px solid var(--dw-rim)',
              padding: 14,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}>
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  onClick={() => setLayerPanelOpen(false)}
                  aria-label="Close layers"
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--dw-fg-muted)',
                    cursor: 'pointer',
                    padding: 6,
                    display: 'flex',
                    alignItems: 'center',
                  }}
                >
                  <X size={16} strokeWidth={1.5} />
                </button>
              </div>
              <LayerControl
                activeLayer={activeLayer}
                onLayerChange={handleLayerChange}
              />
            </div>
          </div>
        )}

        {/* Side Analysis Panel — collapsible 320px right drawer (Phase 10 & 12, §25, §16) */}
        {!isLoading && (
          <AnalysisPanel
            open={analysisPanelOpen}
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

        {/* Right side panel buttons (Layers & Analysis) — always accessible */}
        {!isLoading && (
          <div
            style={{
              position: 'absolute',
              top: 14,
              right: 14,
              transform: `translateX(-${analysisPanelOpen ? 320 : (layerPanelOpen ? 260 : 0)}px)`,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              zIndex: 16,
              transition: 'transform 200ms ease-out',
            }}
          >
            {/* Layers toggle */}
            <button
              onClick={() => {
                setLayerPanelOpen(v => {
                  const next = !v;
                  if (next) setAnalysisPanelOpen(false);
                  return next;
                });
              }}
              aria-expanded={layerPanelOpen}
              aria-label="Toggle layer panel"
              title={layerPanelOpen ? 'Close layers' : 'Open layers'}
              style={{
                height: 36,
                padding: '0 12px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 7,
                background: layerPanelOpen ? 'var(--dw-surface)' : 'rgba(13,17,23,0.92)',
                border: layerPanelOpen ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 13.5,
                fontWeight: 500,
                color: layerPanelOpen ? 'var(--dw-accent)' : 'var(--dw-fg)',
                cursor: 'pointer',
                outline: 'none',
                transition: 'border-color 120ms ease, color 120ms ease, background 120ms ease',
              }}
              onFocus={e => {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '2px';
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              <Layers size={16} strokeWidth={1.5} aria-hidden="true" />
              Layers
            </button>

            {/* Analysis Panel toggle */}
            <button
              onClick={() => {
                setAnalysisPanelOpen(v => {
                  const next = !v;
                  if (next) setLayerPanelOpen(false);
                  return next;
                });
              }}
              aria-expanded={analysisPanelOpen}
              aria-label="Toggle analysis panel"
              title={analysisPanelOpen ? 'Close analysis' : 'Open analysis'}
              style={{
                height: 36,
                padding: '0 12px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 7,
                background: analysisPanelOpen ? 'var(--dw-surface)' : 'rgba(13,17,23,0.92)',
                border: analysisPanelOpen ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 13.5,
                fontWeight: 500,
                color: analysisPanelOpen ? 'var(--dw-accent)' : 'var(--dw-fg)',
                cursor: 'pointer',
                outline: 'none',
                transition: 'border-color 120ms ease, color 120ms ease, background 120ms ease',
              }}
              onFocus={e => {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '2px';
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              <PanelRight size={16} strokeWidth={1.5} aria-hidden="true" />
              Analysis
            </button>
          </div>
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
              background: 'rgba(7,9,14,0.7)',
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

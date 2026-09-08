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
import { useRef, useEffect, useState } from 'react';
import Header from '../components/common/Header.jsx';
import TerrainCanvas from '../components/TerrainViewer/TerrainCanvas.jsx';
import Minimap from '../components/TerrainViewer/Minimap.jsx';
import CameraHUD from '../components/TerrainViewer/CameraHUD.jsx';
import LayerControl, { LAYER_META } from '../components/TerrainViewer/LayerControl.jsx';
import Toolbar from '../components/common/Toolbar.jsx';
import ElevationProbe from '../components/Analysis/ElevationProbe.jsx';
import HeightMeasurement from '../components/Analysis/HeightMeasurement.jsx';
import DistanceMeasurement from '../components/Analysis/DistanceMeasurement.jsx';
import SlopeMeasurement from '../components/Analysis/SlopeMeasurement.jsx';
import StructureInspector from '../components/Analysis/StructureInspector.jsx';
import ToolGuard from '../components/Analysis/ToolGuard.jsx';
import RegionSelector from '../components/Analysis/RegionSelector.jsx';
import AnalysisPanel from '../components/common/AnalysisPanel.jsx';
import { useCameraController } from '../hooks/useCameraController.js';
import { getMinimap } from '../api/terrain.js';
import { getResults, getDepth, getDsm, getReference } from '../api/results.js';
import { getErrorMap } from '../api/validation.js';
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
} from 'lucide-react';

export default function TerrainWorkspace() {
  const { state } = useApp();
  const terrainRef = useRef(null);

  const isLoading = state.status === AppState.TERRAIN_LOADING;

  // ── Camera controller ──
  // canvasRef and glRef are exposed from TerrainCanvas via terrainRef.getCanvas/getRef
  // We wire these after first mount via a stable placeholder ref
  const canvasPlaceholder = useRef({ current: null });
  const glPlaceholder     = useRef({ current: {} });

  const {
    mode: cameraMode,
    setMode: setCameraMode,
    tickFirstPerson,
  } = useCameraController({
    canvasRef: canvasPlaceholder,
    glRef: glPlaceholder,
  });

  // After TerrainCanvas mounts, connect the real refs into the camera controller
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!terrainRef.current) return;
      const realCanvas = terrainRef.current.getCanvas?.();
      const realGl     = terrainRef.current.getRef?.();
      if (realCanvas) canvasPlaceholder.current = realCanvas.current ? realCanvas : { current: realCanvas };
      if (realGl)     Object.assign(glPlaceholder, { current: realGl.current ?? realGl });
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  // Register first-person tick in canvas render loop
  useEffect(() => {
    terrainRef.current?.setFpTick(cameraMode === 'first-person' ? tickFirstPerson : null);
    terrainRef.current?.setCameraMode(cameraMode);
  }, [cameraMode, tickFirstPerson]);

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

  // ── Layer system (Phase 8) ──
  const [activeLayer, setActiveLayer] = useState('rgb');
  const [layerPanelOpen, setLayerPanelOpen] = useState(false);
  // Cache of layer URL → { url, colormapMode } to avoid re-fetching
  const layerCache = useRef({});

  // Colormap mode per layer (matches fragment shader uniforms)
  const COLORMAP_MODE = { rgb: 0, depth: 1, dsm: 2, reference_dem: 2, error: 3, slope: 2 };

  /** Fetch the texture URL for a layer and swap the terrain texture (task 8.2) */
  async function handleLayerChange(layerId) {
    if (layerId === activeLayer) return;
    setActiveLayer(layerId);

    const sceneId = state.scene?.scene_id;
    if (!sceneId) return;

    // Check cache first
    if (layerCache.current[layerId]) {
      const { url, colormapMode } = layerCache.current[layerId];
      terrainRef.current?.setLayerTexture(url, colormapMode);
      return;
    }

    try {
      let url = null;
      const colormapMode = COLORMAP_MODE[layerId] ?? 0;

      if (layerId === 'rgb') {
        // RGB uses the terrain texture (already loaded)
        const meta = state.terrain;
        url = meta?.texture_url ?? null;
      } else if (layerId === 'depth') {
        const depth = await getDepth(sceneId);
        url = depth?.url ?? null;
      } else if (layerId === 'dsm') {
        const dsm = await getDsm(sceneId);
        url = dsm?.download_url ?? null;
      } else if (layerId === 'error') {
        const errMap = await getErrorMap(sceneId);
        url = errMap?.url ?? null;
      } else if (layerId === 'reference_dem') {
        const refDem = await getReference(sceneId);
        url = refDem?.visualization_url ?? refDem?.download_url ?? null;
      } else {
        // Other layers: try results endpoint for URL
        const results = await getResults(sceneId);
        url = results?.[layerId + '_url'] ?? results?.layers?.[layerId] ?? null;
      }

      if (url) {
        layerCache.current[layerId] = { url, colormapMode };
        terrainRef.current?.setLayerTexture(url, colormapMode);
      }
    } catch {
      // Layer fetch failed — keep current layer
      setActiveLayer(activeLayer);
    }
  }

  // ── Analysis tools state (Phase 9, 10 & 13) ──
  const [activeTool, setActiveTool] = useState('none');
  const heightToolRef = useRef(null);
  const distToolRef   = useRef(null);
  const slopeToolRef  = useRef(null);
  const structToolRef = useRef(null);

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
  const handleTerrainClick = (e) => {
    if (isLoading) return;
    const pt = terrainRef.current?.getTerrainPointFromEvent?.(e);
    if (!pt) return;

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
            selectedPoint={selectedPoint}
          />
        )}

        {/* Elevation Probe (task 9.1) */}
        {!isLoading && (
          <ElevationProbe
            terrainRef={terrainRef}
            enabled={activeTool === 'probe' || activeTool === 'none'}
          />
        )}

        {/* Navigation HUD — bottom-right, first-person mode only (Phase 7) */}
        {!isLoading && (
          <CameraHUD
            terrainRef={terrainRef}
            cameraMode={cameraMode}
            elevationMode={state.results?.elevation_mode ?? 'relative'}
          />
        )}

        {/* Active Analysis tool readout panel (Phase 9, tasks 9.2-9.6) */}
        {!isLoading && activeTool !== 'none' && activeTool !== 'probe' && (
          <div style={{
            position: 'absolute',
            top: 56,
            right: 12,
            transform: `translateX(-${analysisPanelOpen ? 280 : (layerPanelOpen ? 220 : 0)}px)`,
            zIndex: 12,
            transition: 'transform 200ms ease-out',
          }}>
            <ToolGuard>
              {activeTool === 'height' && (
                <HeightMeasurement ref={heightToolRef} active={true} />
              )}
              {activeTool === 'distance' && (
                <DistanceMeasurement ref={distToolRef} active={true} />
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

        {/* Layer panel — collapsible right overlay (Phase 8) */}
        {!isLoading && (
          <div style={{
            position: 'absolute',
            top: 0,
            right: 0,
            bottom: 0,
            width: 220,
            transform: layerPanelOpen ? 'translateX(0)' : 'translateX(100%)',
            transition: 'transform 200ms ease-out',
            zIndex: 14,
            pointerEvents: layerPanelOpen ? 'auto' : 'none',
          }}>
            <div style={{
              width: 220,
              height: '100%',
              background: 'var(--dw-panel)',
              borderLeft: '1px solid var(--dw-rim)',
              padding: 12,
              overflowY: 'auto',
            }}>
              <LayerControl
                activeLayer={activeLayer}
                onLayerChange={handleLayerChange}
              />
            </div>
          </div>
        )}

        {/* Side Analysis Panel — collapsible 280px right drawer (Phase 10 & 12, §25, §16) */}
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
              top: 12,
              right: 12,
              transform: `translateX(-${analysisPanelOpen ? 280 : (layerPanelOpen ? 220 : 0)}px)`,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
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
                height: 32,
                padding: '0 10px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                background: layerPanelOpen ? 'var(--dw-surface)' : 'rgba(13,17,23,0.88)',
                border: layerPanelOpen ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 12,
                color: layerPanelOpen ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
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
              <Layers size={13} strokeWidth={1.5} aria-hidden="true" />
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
                height: 32,
                padding: '0 10px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                background: analysisPanelOpen ? 'var(--dw-surface)' : 'rgba(13,17,23,0.88)',
                border: analysisPanelOpen ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 12,
                color: analysisPanelOpen ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
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
              <PanelRight size={13} strokeWidth={1.5} aria-hidden="true" />
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
      </div>

      {/* 48px unified bottom toolbar (Phase 11, §26) */}
      <Toolbar
        terrainRef={terrainRef}
        cameraMode={cameraMode}
        onSetCameraMode={setCameraMode}
        activeLayer={activeLayer}
        onSelectLayer={handleLayerChange}
        activeTool={activeTool}
        onSelectTool={handleSelectTool}
        disabled={isLoading}
        onOpenValidation={() => {
          setAnalysisPanelOpen(true);
          setAnalysisPanelTab('validation');
        }}
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
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      {/* Terrain profile glyph — animated */}
      <svg
        width="40"
        height="20"
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
        fontSize: 11,
        color: 'var(--dw-fg-muted)',
        letterSpacing: '0.06em',
      }}>
        BUILDING TERRAIN
      </span>
    </div>
  );
}

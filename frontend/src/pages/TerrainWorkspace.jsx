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
import { RotateCcw, Layers } from 'lucide-react';
import Header from '../components/common/Header.jsx';
import TerrainCanvas from '../components/TerrainViewer/TerrainCanvas.jsx';
import TerrainControls from '../components/TerrainViewer/TerrainControls.jsx';
import Minimap from '../components/TerrainViewer/Minimap.jsx';
import CameraHUD from '../components/TerrainViewer/CameraHUD.jsx';
import LayerControl, { LAYER_META } from '../components/TerrainViewer/LayerControl.jsx';
import { useCameraController } from '../hooks/useCameraController.js';
import { getMinimap } from '../api/terrain.js';
import { getResults, getDepth, getDsm } from '../api/results.js';
import { useApp, AppState } from '../store/appStore.jsx';

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
      <div style={{
        flex: 1,
        position: 'relative',
        overflow: 'hidden',
      }}>
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

        {/* Navigation HUD — bottom-right, first-person mode only (Phase 7) */}
        {!isLoading && (
          <CameraHUD
            terrainRef={terrainRef}
            cameraMode={cameraMode}
            elevationMode={state.results?.elevation_mode ?? 'relative'}
          />
        )}

        {/* Layer panel — collapsible right overlay (Phase 8) */}
        {!isLoading && (
          <div style={{
            position: 'absolute',
            top: 0,
            right: 0,
            bottom: 0,
            width: layerPanelOpen ? 220 : 0,
            overflow: 'hidden',
            transition: 'width 200ms ease-out',
            zIndex: 11,
            display: 'flex',
          }}>
            <div style={{
              width: 220,
              height: '100%',
              background: 'var(--dw-panel)',
              borderLeft: '1px solid var(--dw-rim)',
              padding: 12,
              overflowY: 'auto',
              flexShrink: 0,
            }}>
              <LayerControl
                activeLayer={activeLayer}
                onLayerChange={handleLayerChange}
              />
            </div>
          </div>
        )}

        {/* Layer toggle button — shown on right edge, always accessible */}
        {!isLoading && (
          <button
            onClick={() => setLayerPanelOpen(v => !v)}
            aria-expanded={layerPanelOpen}
            aria-label="Toggle layer panel"
            title={layerPanelOpen ? 'Close layers' : 'Open layers'}
            style={{
              position: 'absolute',
              top: 12,
              right: layerPanelOpen ? 232 : 12,
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
              zIndex: 12,
              transition: 'right 200ms ease-out, border-color 120ms ease, color 120ms ease',
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

      {/* 48px bottom toolbar */}
      <div style={{
        height: 'var(--dw-toolbar-h)',
        background: 'var(--dw-panel)',
        borderTop: '1px solid var(--dw-rim)',
        display: 'flex',
        alignItems: 'center',
        gap: 0,
        flexShrink: 0,
      }}>
        <TerrainControls terrainRef={terrainRef} disabled={isLoading} />

        {/* Camera mode switcher — task 5.4: [ First Person ] [ Orbit ] [ Top View ] */}
        <CameraModeSwitcher
          mode={cameraMode}
          onSetMode={setCameraMode}
          disabled={isLoading}
          terrainRef={terrainRef}
        />

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Camera reset button */}
        <div style={{ padding: '0 12px', display: 'flex', alignItems: 'center' }}>
          <button
            onClick={() => terrainRef.current?.resetCamera()}
            disabled={isLoading}
            aria-label="Reset camera to default position"
            title="Reset camera"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              height: 32,
              padding: '0 10px',
              background: 'none',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              color: 'var(--dw-fg-muted)',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              opacity: isLoading ? 0.4 : 1,
              outline: 'none',
            }}
            onFocus={e => {
              if (!isLoading) {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '2px';
              }
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            onMouseEnter={e => {
              if (!isLoading) e.currentTarget.style.borderColor = 'var(--dw-fg-ghost)';
            }}
            onMouseLeave={e => {
              if (!isLoading) e.currentTarget.style.borderColor = 'var(--dw-rim)';
            }}
          >
            <RotateCcw size={13} strokeWidth={1.5} aria-hidden="true" />
            Reset view
          </button>
        </div>
      </div>
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

/**
 * Camera mode switcher — task 5.4
 * [ First Person ] [ Orbit ] [ Top View ] button group.
 * DESIGN.md: buttons 32px high, border-radius 4px, active state: --dw-surface + --dw-accent border.
 */
const CAMERA_MODES = [
  { id: 'first-person', label: 'First Person' },
  { id: 'orbit',        label: 'Orbit' },
  { id: 'top',          label: 'Top View' },
];

function CameraModeSwitcher({ mode, onSetMode, disabled, terrainRef }) {
  function handleClick(newMode) {
    if (newMode === mode) return;
    terrainRef.current?.setCameraMode(newMode);
    onSetMode(newMode);
  }

  return (
    <div
      role="group"
      aria-label="Camera mode"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        padding: '0 8px',
        opacity: disabled ? 0.4 : 1,
        pointerEvents: disabled ? 'none' : 'auto',
      }}
    >
      <div style={{ width: 1, height: 24, background: 'var(--dw-rim)', marginRight: 8 }} />
      {CAMERA_MODES.map((m) => {
        const isActive = mode === m.id;
        return (
          <button
            key={m.id}
            onClick={() => handleClick(m.id)}
            aria-pressed={isActive}
            style={{
              height: 32,
              padding: '0 10px',
              background: isActive ? 'var(--dw-surface)' : 'none',
              border: isActive ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12,
              color: isActive ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
              cursor: 'pointer',
              outline: 'none',
              transition: 'border-color 120ms ease, color 120ms ease, background 120ms ease',
              whiteSpace: 'nowrap',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '2px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            onMouseEnter={e => {
              if (!isActive) e.currentTarget.style.borderColor = 'var(--dw-fg-ghost)';
            }}
            onMouseLeave={e => {
              if (!isActive) e.currentTarget.style.borderColor = 'var(--dw-rim)';
            }}
          >
            {m.label}
          </button>
        );
      })}
    </div>
  );
}

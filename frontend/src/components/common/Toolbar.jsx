/**
 * DepthWizard — Toolbar (Phase 11, Tasks 11.1, 11.2, 11.3)
 *
 * Full-width 48px bottom toolbar with popover submenus (§26).
 * Structure:
 *   [Layers] [Measure] [Compare] [Terrain] [Camera] [Reset]
 *
 * Submenus:
 *   - Layers:  RGB / Depth / DSM / Reference DEM / Slope / Error Map
 *   - Measure: Elevation Probe / Height / Distance / Slope
 *   - Compare: Estimated DSM / Reference DEM / Error Map
 *   - Terrain: Exaggeration (1×–5×) / Wireframe / Contours
 *   - Camera:  Orbit / Walkthrough / Top View
 *   - Reset:   Return to default camera and visual settings
 *
 * DESIGN.md:
 *   - Height: 48px (var(--dw-toolbar-h)), background: var(--dw-panel)
 *   - 1px top border: var(--dw-rim)
 *   - Buttons: 32px height, 4px radius, 10px padding
 *   - Active button/tool state: background: var(--dw-surface), border: 1px solid var(--dw-accent)
 *   - Popovers: dock above toolbar with 1px var(--dw-rim), no card shadows
 *
 * Spec §26.
 */
import { useState, useRef, useEffect, useCallback } from 'react';
import { useApp } from '../../store/appStore.jsx';
import {
  Layers,
  Ruler,
  RotateCcw,
  Grid3x3,
  Spline,
  Eye,
  Crosshair,
  ArrowUpDown,
  TrendingUp,
  ShieldCheck,
  Building2,
  Sparkles,
  Download,
  Route as RouteIcon,
} from 'lucide-react';
import ExportPanel from '../Export/ExportPanel.jsx';

/**
 * @param {{
 *   terrainRef: React.RefObject,
 *   cameraMode: string,
 *   onSetCameraMode: (mode: string) => void,
 *   activeLayer: string,
 *   onSelectLayer: (layerId: string) => void,
 *   activeTool: string,
 *   onSelectTool: (toolId: string) => void,
 *   disabled?: boolean,
 *   onOpenValidation?: () => void,
 *   onCaptureSnapshot?: () => void,
 * }} props
 */
export default function Toolbar({
  terrainRef,
  cameraMode = 'orbit',
  onSetCameraMode,
  activeLayer = 'rgb',
  onSelectLayer,
  activeTool = 'none',
  onSelectTool,
  disabled = false,
  onOpenValidation,
  onCaptureSnapshot,
  layerAvailability,
}) {
  // Real per-layer availability, fetched from the backend by the workspace
  // (results + reference endpoints). Defaults to "available" so the popover
  // stays interactive when the workspace hasn't reported yet; a `false`
  // entry disables the matching Compare/Layers item with a visible reason.
  const avail = { dsm: true, reference: true, error: true, ...(layerAvailability ?? {}) };
  const { state } = useApp();
  const [openMenu, setOpenMenu] = useState(null); // 'layers' | 'measure' | 'export' | null
  const toolbarRef = useRef(null);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel = 'm'; // world scale is metres (1 m/pixel documented fallback)

  // Close menus on outside click or ESC key
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') setOpenMenu(null);
    }
    function handleClickOutside(e) {
      if (toolbarRef.current && !toolbarRef.current.contains(e.target)) {
        setOpenMenu(null);
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const toggleMenu = (menuId) => {
    setOpenMenu(curr => curr === menuId ? null : menuId);
  };

  /* ── Reset action (§26) ── */
  const handleReset = useCallback(() => {
    terrainRef.current?.resetCamera();
    terrainRef.current?.setExaggeration(2.5);
    terrainRef.current?.setWireframe(false);
    terrainRef.current?.setContours(false, 5);
    terrainRef.current?.setFog?.(false);
    onSetCameraMode('orbit');
    onSelectTool('none');
    setOpenMenu(null);
  }, [terrainRef, onSetCameraMode, onSelectTool]);

  return (
    <footer
      ref={toolbarRef}
      aria-label="Terrain view toolbar"
      style={{
        position: 'relative',
        height: 'var(--dw-toolbar-h)',
        background: 'var(--dw-panel)',
        borderTop: '1px solid var(--dw-rim)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 12px',
        gap: 6,
        zIndex: 20,
        opacity: disabled ? 0.4 : 1,
        pointerEvents: disabled ? 'none' : 'auto',
      }}
    >
      {/* ── 1. Layers Menu ── */}
      <ToolbarItem
        id="layers"
        label="Layers"
        icon={Layers}
        isOpen={openMenu === 'layers'}
        isActive={activeLayer !== 'rgb'}
        onToggle={() => toggleMenu('layers')}
      />

      {/* ── 2. Measure Menu ── */}
      <ToolbarItem
        id="measure"
        label="Measure"
        icon={Ruler}
        isOpen={openMenu === 'measure'}
        isActive={activeTool !== 'none'}
        onToggle={() => toggleMenu('measure')}
      />

      {/* ── 3. Compare Menu ── */}
      {/* ── 4. Terrain Menu ── */}
      {/* ── 5. Camera Menu ── */}
      {/* ── Spacer ── */}
      <div style={{ flex: 1 }} />

      {/* ── 6. Export Menu (§27) ── */}
      <ToolbarItem
        id="export"
        label="Export"
        icon={Download}
        isOpen={openMenu === 'export'}
        isActive={openMenu === 'export'}
        onToggle={() => toggleMenu('export')}
      />

      {/* ── 7. Reset Button (§26) ── */}
      <button
        onClick={handleReset}
        aria-label="Reset camera and terrain settings to default"
        title="Reset view"
        style={{
          height: 36,
          padding: '0 12px',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          background: 'none',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          color: 'var(--dw-fg-muted)',
          cursor: 'pointer',
          outline: 'none',
          transition: 'border-color 120ms ease, color 120ms ease',
        }}
        onFocus={e => {
          e.currentTarget.style.outline = '2px solid var(--dw-accent)';
          e.currentTarget.style.outlineOffset = '2px';
        }}
        onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--dw-fg-ghost)'; }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--dw-rim)'; }}
      >
        <RotateCcw size={16} strokeWidth={1.5} aria-hidden="true" />
        Reset
      </button>

      {/* ── Popover Submenus (§26, task 11.2) ── */}

      {/* Popover 1: Layers */}
      {openMenu === 'layers' && (
        <PopoverPanel title="LAYERS" onClose={() => setOpenMenu(null)}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {[
              { id: 'solid',         label: 'Solid (Shaded Relief)' },
              { id: 'rgb',           label: 'RGB (Color Photo)' },
              { id: 'depth',         label: 'Depth Map (Greyscale)' },
              { id: 'passability',   label: 'Passability Heat Map (Vehicle Risk)' },
              { id: 'dsm',           label: isAbsolute ? 'Absolute DSM (Metric)' : 'Relative DSM', disabled: !avail.dsm },
              { id: 'reference_dem', label: 'Reference DEM (SRTM)', disabled: !avail.reference },
              { id: 'slope',         label: 'Slope Layer (Viridis)' },
              { id: 'error',         label: 'Error Map (Diverging)', disabled: !avail.error },
            ].map(l => (
              <PopoverButton
                key={l.id}
                label={l.label}
                active={activeLayer === l.id}
                disabled={l.disabled}
                onClick={() => {
                  onSelectLayer(l.id);
                  setOpenMenu(null);
                }}
              />
            ))}
            {onOpenValidation && (
              <>
                <div style={{ height: 1, background: 'var(--dw-rim)', margin: '4px 0' }} />
                <PopoverButton
                  label="Validation Accuracy Metrics"
                  icon={ShieldCheck}
                  onClick={() => {
                    onOpenValidation();
                    setOpenMenu(null);
                  }}
                />
              </>
            )}
          </div>
        </PopoverPanel>
      )}

      {/* Popover 2: Measure */}
      {openMenu === 'measure' && (
        <PopoverPanel title="MEASURE" onClose={() => setOpenMenu(null)}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {[
              { id: 'probe',     label: 'Elevation Probe',     icon: Crosshair },
              { id: 'height',    label: 'Height Measurement',   icon: ArrowUpDown },
              { id: 'distance',  label: 'Distance Measurement', icon: Ruler },
              { id: 'slope',     label: 'Slope Analysis',       icon: TrendingUp },
              { id: 'structure', label: 'Structure Inspector',  icon: Building2 },
              { id: 'route',     label: 'Route Assist',         icon: RouteIcon },
              { id: 'refine',    label: 'Detail Mode (Refine)', icon: Sparkles },
            ].map(m => (
              <PopoverButton
                key={m.id}
                label={m.label}
                icon={m.icon}
                active={activeTool === m.id}
                onClick={() => {
                  // Pass the tool id — the workspace owns the toggle logic
                  // (selecting the active tool again deactivates it).
                  onSelectTool(m.id);
                  setOpenMenu(null);
                }}
              />
            ))}
          </div>
        </PopoverPanel>
      )}

      {/* Popover 6: Export (§27, §66) */}
      {openMenu === 'export' && (
        <PopoverPanel title="EXPORT" onClose={() => setOpenMenu(null)} width={340} alignRight={true}>
          <ExportPanel
            onCaptureSnapshot={onCaptureSnapshot}
            onClose={() => setOpenMenu(null)}
            compact={true}
          />
        </PopoverPanel>
      )}
    </footer>
  );
}

/** Individual toolbar action button */
function ToolbarItem({ id, label, icon: Icon, isOpen, isActive, onToggle }) {
  return (
    <button
      id={`toolbar-${id}-btn`}
      onClick={onToggle}
      aria-expanded={isOpen}
      aria-haspopup="dialog"
      aria-label={`${label} menu`}
      style={{
        height: 36,
        padding: '0 12px',
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        background: isOpen || isActive ? 'var(--dw-surface)' : 'none',
        border: isOpen || isActive ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 13.5,
        color: isOpen || isActive ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
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
        if (!isOpen && !isActive) e.currentTarget.style.borderColor = 'var(--dw-fg-ghost)';
      }}
      onMouseLeave={e => {
        if (!isOpen && !isActive) e.currentTarget.style.borderColor = 'var(--dw-rim)';
      }}
    >
      <Icon size={16} strokeWidth={1.5} aria-hidden="true" />
      {label}
    </button>
  );
}

/** Floating docked popover container */
function PopoverPanel({ title, children, onClose, width = 280, alignRight = false }) {
  return (
    <div
      role="dialog"
      aria-label={title}
      style={{
        position: 'absolute',
        bottom: 'calc(100% + 8px)',
        left: alignRight ? 'auto' : 12,
        right: alignRight ? 12 : 'auto',
        background: 'var(--dw-panel)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        padding: 14,
        width: `min(calc(100vw - 24px), ${width}px)`,
        maxWidth: 'calc(100vw - 24px)',
        maxHeight: 'calc(100vh - 100px)',
        overflowY: 'auto',
        zIndex: 30,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
    >
      <div style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 11.5,
        fontWeight: 600,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-ghost)',
        marginBottom: 2,
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

/** Button item inside popover */
function PopoverButton({ label, icon: Icon, active, disabled = false, onClick }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      style={{
        height: 34,
        padding: '0 10px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-start',
        gap: 8,
        background: active ? 'var(--dw-surface)' : 'none',
        border: active ? '1px solid var(--dw-accent)' : '1px solid transparent',
        borderRadius: 'var(--dw-radius-sm)',
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 13,
        color: disabled ? 'var(--dw-fg-ghost)' : active ? 'var(--dw-accent)' : 'var(--dw-fg)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        outline: 'none',
        textAlign: 'left',
        width: '100%',
      }}
      onMouseEnter={e => {
        if (!active && !disabled) e.currentTarget.style.background = 'var(--dw-surface)';
      }}
      onMouseLeave={e => {
        if (!active && !disabled) e.currentTarget.style.background = 'none';
      }}
    >
      {Icon && <Icon size={15} strokeWidth={1.5} aria-hidden="true" />}
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {label}
      </span>
    </button>
  );
}

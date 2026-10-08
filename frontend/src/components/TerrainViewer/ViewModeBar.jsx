/**
 * DepthWizard — ViewModeBar
 *
 * Top in-viewport control bar of the terrain workspace (TerraLens-style):
 *   [ RGB | Depth | Metric DSM | Hybrid | Wireframe | Contour ]   [Orbit|Fly]   [Inspect|Measure|Navigate Only]
 *
 * Wiring:
 *   - View tabs drive the existing layer system (rgb / depth / dsm) plus
 *     render modes: Hybrid = shaded relief (solid view), Wireframe and
 *     Contour are shader toggles over the current texture.
 *   - Orbit/Fly maps to the camera controller's orbit / first-person modes.
 *   - Inspect/Measure/Navigate maps to the probe / distance / no tool.
 *     Finer tools (height, slope, structure, route) stay in the bottom toolbar.
 */
import { Orbit, Footprints, Search, Ruler, Navigation, Map, PanelRight, Eye, EyeOff, Building2 } from 'lucide-react';

export const VIEW_TABS = [
  { id: 'rgb', label: 'RGB' },
  { id: 'depth', label: 'Depth' },
  { id: 'dsm', label: 'Metric DSM' },
  { id: 'semantics', label: 'Semantics' },
  { id: 'hybrid', label: 'Hybrid' },
  { id: 'wireframe', label: 'Wireframe' },
  { id: 'contour', label: 'Contour' },
];

const pillBase = {
  height: 32,
  padding: '0 13px',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  background: 'transparent',
  border: 'none',
  borderRadius: 'var(--dw-radius-sm)',
  fontFamily: 'var(--dw-font-ui)',
  fontSize: 13,
  fontWeight: 500,
  color: 'var(--dw-fg-muted)',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
  transition: 'background 120ms ease, color 120ms ease',
};

function Segmented({ options, value, onChange, disabled, ariaLabel }) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 2,
        height: 36,
        padding: 2,
        background: 'rgba(16,16,18,0.92)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius)',
        pointerEvents: 'auto',
      }}
    >
      {options.map((opt) => {
        const active = value === opt.id;
        return (
          <button
            key={opt.id}
            role="radio"
            aria-checked={active}
            disabled={disabled}
            onClick={() => onChange(opt.id)}
            style={{
              ...pillBase,
              height: 28,
              padding: '0 11px',
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.45 : 1,
              background: active ? 'var(--dw-accent)' : 'transparent',
              color: active ? '#fff' : 'var(--dw-fg-muted)',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '1px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            {opt.icon}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

export default function ViewModeBar({
  viewTab,
  onViewTab,
  cameraMode,
  onCameraMode,
  toolMode,
  onToolMode,
  disabled = false,
  // When set, the bar docks at this left offset instead of top-centre
  // (top-centre belongs to the batch SceneSwitcher).
  leftOffset = null,
  // Scene Analysis drawer toggle — lives in this bar so it can never
  // collide with the docked panels.
  analysisOpen = false,
  onToggleAnalysis,
  // Side-panel visibility toggle (walkthrough auto-hides the docks)
  panelsHidden = false,
  onTogglePanels,
  // 3D buildings view mode — mutually exclusive with the elevated relief:
  // ON renders the building-removed ground + 3D blocks/trees/objects,
  // OFF restores the elevated DSM (with flat-topped buildings).
  buildings3d = null,
  onToggleBuildings3d,
  // Deferred analysis (lazy pipeline): when the scene is dsm_ready the
  // 3D-buildings artifacts don't exist yet — the button stays VISIBLE and
  // starts POST /scenes/{id}/analyze instead of disappearing.
  analysisPending = false,
  analysisRunning = false,
  analysisProgress = null,
  onRunAnalysis,
}) {
  return (
    <div
      style={{
        position: 'absolute',
        top: 12,
        left: leftOffset ?? '50%',
        transform: leftOffset == null ? 'translateX(-50%)' : undefined,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        flexWrap: 'wrap',
        justifyContent: 'center',
        maxWidth: 'calc(100% - 24px)',
        zIndex: 11,
        pointerEvents: 'none',
      }}
    >
      {/* View tabs */}
      <div
        role="tablist"
        aria-label="Terrain view mode"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 2,
          height: 36,
          padding: 2,
          background: 'rgba(16,16,18,0.92)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius)',
          pointerEvents: 'auto',
        }}
      >
        {VIEW_TABS.map((tab) => {
          const active = viewTab === tab.id;
          return (
            <button
              key={tab.id}
              role="tab"
              aria-selected={active}
              disabled={disabled}
              onClick={() => onViewTab(tab.id)}
              style={{
                ...pillBase,
                cursor: disabled ? 'not-allowed' : 'pointer',
                opacity: disabled ? 0.45 : 1,
                background: active ? 'var(--dw-accent)' : 'transparent',
                color: active ? '#fff' : 'var(--dw-fg-muted)',
              }}
              onFocus={e => {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '1px';
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Camera mode */}
      <Segmented
        ariaLabel="Camera mode"
        disabled={disabled}
        value={cameraMode === 'first-person' ? 'fly' : cameraMode === 'top' ? 'top' : 'orbit'}
        onChange={(id) => onCameraMode(id === 'fly' ? 'first-person' : id)}
        options={[
          { id: 'orbit', label: 'Orbit', icon: <Orbit size={13} strokeWidth={1.5} aria-hidden="true" /> },
          { id: 'fly', label: 'Fly', icon: <Footprints size={13} strokeWidth={1.5} aria-hidden="true" /> },
          { id: 'top', label: 'Top', icon: <Map size={13} strokeWidth={1.5} aria-hidden="true" /> },
        ]}
      />

      {/* 3D buildings view mode — a first-class top-bar toggle. When the
          deferred analysis hasn't run yet, the button remains visible and
          click-starts the analysis (with live progress) instead. */}
      {(buildings3d?.available || analysisPending || analysisRunning) && (
        <button
          onClick={buildings3d?.available ? onToggleBuildings3d : onRunAnalysis}
          disabled={disabled || (analysisRunning && !buildings3d?.available)}
          aria-pressed={!!buildings3d?.enabled}
          title={buildings3d?.available
            ? (buildings3d.enabled
              ? '3D objects ON — flat ground with building blocks/trees/objects (click to return to the elevated terrain)'
              : '3D objects OFF — elevated terrain (click to switch to the 3D objects view)')
            : analysisRunning
              ? 'Full analysis is running — 3D buildings become available when it completes'
              : '3D buildings need the full analysis — click to run it now (validation, disaster, buildings)'}
          style={{
            height: 36,
            padding: '0 12px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 7,
            background: buildings3d?.available && buildings3d.enabled
              ? 'var(--dw-accent)' : 'rgba(16,16,18,0.92)',
            border: buildings3d?.available && buildings3d.enabled
              ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
            borderRadius: 8,
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13,
            fontWeight: 600,
            color: buildings3d?.available && buildings3d.enabled
              ? 'var(--dw-fg-invert)' : 'var(--dw-fg-muted)',
            cursor: (disabled || (analysisRunning && !buildings3d?.available))
              ? 'wait' : 'pointer',
            transition: 'background 120ms ease, color 120ms ease',
          }}
        >
          <Building2 size={15} strokeWidth={1.5} aria-hidden="true" />
          {buildings3d?.available
            ? '3D Objects'
            : analysisRunning
              ? `3D Objects · analysing ${analysisProgress != null ? Math.round(analysisProgress) : 0}%`
              : '3D Objects · Run Analysis'}
        </button>
      )}

      {/* Interaction mode */}
      <Segmented
        ariaLabel="Interaction mode"
        disabled={disabled}
        value={toolMode}
        onChange={onToolMode}
        options={[
          { id: 'inspect', label: 'Inspect', icon: <Search size={13} strokeWidth={1.5} aria-hidden="true" /> },
          { id: 'measure', label: 'Measure', icon: <Ruler size={13} strokeWidth={1.5} aria-hidden="true" /> },
          { id: 'navigate', label: 'Navigate', icon: <Navigation size={13} strokeWidth={1.5} aria-hidden="true" /> },
        ]}
      />

      {/* Side panels visibility — hidden automatically in walkthrough */}
      {onTogglePanels && (
        <button
          onClick={onTogglePanels}
          aria-pressed={!panelsHidden}
          aria-label={panelsHidden ? 'Show side panels' : 'Hide side panels'}
          title={panelsHidden ? 'Show side panels' : 'Hide side panels'}
          style={{
            height: 36,
            padding: '0 12px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 7,
            background: 'rgba(16,16,18,0.92)',
            border: panelsHidden ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13.5,
            fontWeight: 500,
            color: panelsHidden ? 'var(--dw-accent)' : 'var(--dw-fg)',
            cursor: 'pointer',
            outline: 'none',
            pointerEvents: 'auto',
            transition: 'border-color 120ms ease, color 120ms ease',
          }}
          onFocus={e => {
            e.currentTarget.style.outline = '2px solid var(--dw-accent)';
            e.currentTarget.style.outlineOffset = '2px';
          }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        >
          {panelsHidden
            ? <Eye size={15} strokeWidth={1.5} aria-hidden="true" />
            : <EyeOff size={15} strokeWidth={1.5} aria-hidden="true" />}
          {panelsHidden ? 'Show Panels' : 'Hide Panels'}
        </button>
      )}

      {/* Scene Analysis drawer toggle */}
      {onToggleAnalysis && (
        <button
          onClick={onToggleAnalysis}
          aria-expanded={analysisOpen}
          aria-label="Toggle analysis panel"
          title={analysisOpen ? 'Close analysis' : 'Open analysis'}
          style={{
            height: 36,
            padding: '0 12px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 7,
            background: analysisOpen ? 'var(--dw-surface)' : 'rgba(16,16,18,0.92)',
            border: analysisOpen ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13.5,
            fontWeight: 500,
            color: analysisOpen ? 'var(--dw-accent)' : 'var(--dw-fg)',
            cursor: 'pointer',
            outline: 'none',
            pointerEvents: 'auto',
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
      )}
    </div>
  );
}

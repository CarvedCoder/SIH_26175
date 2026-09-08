/**
 * DepthWizard — AnalysisPanel (Phase 10, Tasks 10.1, 10.2, 10.3)
 *
 * Collapsible side analysis panel (§25).
 * Contains context-sensitive information:
 *
 * 1. Default context:
 *    SCENE:
 *      Image: scene_042.tif
 *      Resolution: 4096 × 4096
 *      Mode: Absolute DSM / Relative DSM
 *      Reference: SRTM / None
 *    MODEL:
 *      Depth Model: Depth Anything V2
 *      Status: Complete
 *
 * 2. When user selects a point (context: location):
 *    SELECTED LOCATION:
 *      Elevation: 142.63 m
 *      Slope: 12.4°
 *      Position: X / Z
 *
 * 3. When user inspects a structure (context: structure):
 *    SELECTED STRUCTURE:
 *      Structure ID: STR-042
 *      Height: 18.4 m
 *      Ground: 126.2 m
 *      Top: 144.6 m
 *
 * DESIGN.md:
 *   - 280px width, right-side dock
 *   - Collapsible with 200ms ease-out GPU transform
 *   - Data values in monospace data face, labels in UI face
 *   - Uppercase compact group headers
 *   - Thin 1px --dw-rim dividers
 *   - No card shadows
 *
 * Spec §25, §D05, §D10.
 */
import { useState } from 'react';
import { useApp } from '../../store/appStore.jsx';
import {
  ChevronRight,
  ChevronLeft,
  Info,
  MapPin,
  Building2,
  X,
} from 'lucide-react';

/**
 * @param {{
 *   open?: boolean,
 *   onToggle?: () => void,
 *   selectedLocation?: { x: number, z: number, elevation: number, slope?: number } | null,
 *   selectedStructure?: { id: string, ground: number, top: number, height: number } | null,
 *   onClearSelection?: () => void,
 * }} props
 */
export default function AnalysisPanel({
  open = true,
  onToggle,
  selectedLocation = null,
  selectedStructure = null,
  onClearSelection,
}) {
  const { state } = useApp();

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel  = isAbsolute ? 'm' : 'scene units';

  // Determine current context view
  const isStructure = !!selectedStructure;
  const isLocation  = !isStructure && !!selectedLocation;

  return (
    <aside
      aria-label="Side analysis panel"
      aria-expanded={open}
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        bottom: 0,
        width: 280,
        background: 'var(--dw-panel)',
        borderLeft: '1px solid var(--dw-rim)',
        zIndex: 15,
        display: 'flex',
        flexDirection: 'column',
        transform: open ? 'translateX(0)' : 'translateX(100%)',
        transition: 'transform 200ms ease-out',
        pointerEvents: open ? 'auto' : 'none',
      }}
    >
      {/* Panel Header */}
      <div style={{
        height: 44,
        padding: '0 12px',
        borderBottom: '1px solid var(--dw-rim)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {isStructure ? (
            <Building2 size={14} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          ) : isLocation ? (
            <MapPin size={14} strokeWidth={1.5} color="var(--dw-probe)" aria-hidden="true" />
          ) : (
            <Info size={14} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          )}
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg)',
            fontWeight: 500,
          }}>
            {isStructure ? 'Selected Structure' : isLocation ? 'Selected Location' : 'Analysis'}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {(isStructure || isLocation) && (
            <button
              onClick={onClearSelection}
              aria-label="Back to scene overview"
              title="Overview"
              style={{
                height: 24,
                padding: '0 6px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                background: 'none',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 10,
                color: 'var(--dw-fg-muted)',
                cursor: 'pointer',
                outline: 'none',
              }}
            >
              <X size={10} strokeWidth={1.5} aria-hidden="true" />
              Reset
            </button>
          )}

          {onToggle && (
            <button
              onClick={onToggle}
              aria-label={open ? 'Collapse analysis panel' : 'Expand analysis panel'}
              title={open ? 'Collapse' : 'Expand'}
              style={{
                width: 24,
                height: 24,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'none',
                border: 'none',
                borderRadius: 'var(--dw-radius-sm)',
                color: 'var(--dw-fg-muted)',
                cursor: 'pointer',
                outline: 'none',
              }}
            >
              <ChevronRight size={14} strokeWidth={1.5} aria-hidden="true" />
            </button>
          )}
        </div>
      </div>

      {/* Scrollable Content Body */}
      <div style={{
        flex: 1,
        padding: '16px 14px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 20,
      }}>
        {/* Context 1: Structure Selected */}
        {isStructure && (
          <section aria-labelledby="selected-structure-heading">
            <h2
              id="selected-structure-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 10,
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                color: 'var(--dw-fg-ghost)',
                margin: '0 0 8px 0',
              }}
            >
              SELECTED STRUCTURE
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <DataRow label="Structure ID" value={selectedStructure.id ?? 'STR-042'} />
              <DataRow
                label="Estimated Height"
                value={`${selectedStructure.height?.toFixed(1) ?? '—'} ${unitLabel}`}
                highlight
              />
              <DataRow
                label="Ground Elevation"
                value={`${selectedStructure.ground?.toFixed(1) ?? '—'} ${unitLabel}`}
              />
              <DataRow
                label="Top Elevation"
                value={`${selectedStructure.top?.toFixed(1) ?? '—'} ${unitLabel}`}
              />
            </div>
          </section>
        )}

        {/* Context 2: Location Point Selected */}
        {isLocation && (
          <section aria-labelledby="selected-location-heading">
            <h2
              id="selected-location-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 10,
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                color: 'var(--dw-fg-ghost)',
                margin: '0 0 8px 0',
              }}
            >
              SELECTED LOCATION
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <DataRow
                label="Elevation"
                value={`${selectedLocation.elevation?.toFixed(2) ?? '—'} ${unitLabel}`}
                highlight
              />
              <DataRow
                label="Slope"
                value={`${selectedLocation.slope != null ? selectedLocation.slope.toFixed(1) : '—'}°`}
              />
              <DataRow
                label="Terrain Coordinates"
                value={`${selectedLocation.x?.toFixed(3) ?? '0.000'}, ${selectedLocation.z?.toFixed(3) ?? '0.000'}`}
              />
            </div>
          </section>
        )}

        {/* Context 3 (Default): Scene and Model Info (§25) */}
        {!isStructure && !isLocation && (
          <>
            {/* SCENE Section */}
            <section aria-labelledby="scene-info-heading">
              <h2
                id="scene-info-heading"
                style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 10,
                  letterSpacing: '0.07em',
                  textTransform: 'uppercase',
                  color: 'var(--dw-fg-ghost)',
                  margin: '0 0 8px 0',
                }}
              >
                SCENE
              </h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                <DataRow
                  label="Image"
                  value={state.scene?.filename ?? state.scene?.image_name ?? 'scene_042.tif'}
                />
                <DataRow
                  label="Resolution"
                  value={
                    state.scene?.width && state.scene?.height
                      ? `${state.scene.width} × ${state.scene.height}`
                      : '4096 × 4096'
                  }
                />
                <DataRow
                  label="Mode"
                  value={isAbsolute ? 'Absolute DSM' : 'Relative DSM'}
                  highlight={isAbsolute}
                />
                <DataRow
                  label="Reference"
                  value={state.results?.reference_source ?? (isAbsolute ? 'SRTM' : 'None')}
                />
                <DataRow
                  label="Georeferenced"
                  value={state.scene?.is_georeferenced ? 'Yes (EPSG:4326)' : 'No (Relative)'}
                />
              </div>
            </section>

            <div style={{ height: 1, background: 'var(--dw-rim)', margin: '4px 0' }} />

            {/* MODEL Section */}
            <section aria-labelledby="model-info-heading">
              <h2
                id="model-info-heading"
                style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 10,
                  letterSpacing: '0.07em',
                  textTransform: 'uppercase',
                  color: 'var(--dw-fg-ghost)',
                  margin: '0 0 8px 0',
                }}
              >
                MODEL
              </h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                <DataRow
                  label="Depth Model"
                  value={state.scene?.model ?? 'Depth Anything V2'}
                />
                <DataRow
                  label="Status"
                  value={
                    state.status === 'TERRAIN_READY' || state.status === 'ANALYSIS'
                      ? 'Complete'
                      : state.status
                  }
                  accent
                />
                <DataRow
                  label="Elevation Calibration"
                  value={isAbsolute ? 'Metric (Ground GCPs)' : 'Relative (Estimated)'}
                />
                <DataRow
                  label="Terrain Segments"
                  value="256 × 256 (Hi-Res)"
                />
              </div>
            </section>
          </>
        )}
      </div>
    </aside>
  );
}

/** Two-column key-value row formatted per DESIGN.md */
function DataRow({ label, value, highlight = false, accent = false }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      gap: 8,
    }}>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 11,
        color: 'var(--dw-fg-muted)',
        flexShrink: 0,
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 12,
        fontWeight: highlight ? 600 : 400,
        color: accent
          ? 'var(--dw-confirm)'
          : highlight
          ? 'var(--dw-accent)'
          : 'var(--dw-fg)',
        textAlign: 'right',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}>
        {value}
      </span>
    </div>
  );
}

/**
 * DepthWizard — DisasterAssessmentPanel (Phase 14, Task 14.2)
 *
 * Dedicated analysis preset for Disaster Management evaluation.
 * Prioritizes topographic indicators:
 * - Elevation relief & span (z_min, z_max, delta)
 * - Slope incline hazard profiling (Steep slopes >30° vs Low-lying basins <5°)
 * - Terrain accessibility & structural clearance
 * - Direct cross-validation with reference DEM
 *
 * Spec §23:
 * - Clearly marked with "Terrain intelligence / preliminary terrain assessment support".
 * - Never claims automated flood/landslide predictions.
 * - Dark mode mission-control styling.
 */
import { AlertTriangle, TrendingUp, Mountain, Building2, ShieldCheck, MapPin } from 'lucide-react';

/**
 * @param {{
 *   terrainMeta?: { min_elevation?: number, max_elevation?: number, width?: number, height?: number } | null,
 *   selectedLocation?: { elevation?: number, slope?: number, x?: number, z?: number } | null,
 *   selectedStructure?: { id?: string, height?: number, ground?: number, top?: number } | null,
 *   isAbsolute?: boolean,
 *   unitLabel?: string,
 *   onSelectLayer?: (layerId: string) => void,
 *   onOpenValidation?: () => void,
 * }} props
 */
export default function DisasterAssessmentPanel({
  terrainMeta = null,
  selectedLocation = null,
  selectedStructure = null,
  isAbsolute = true,
  unitLabel = 'm',
  onSelectLayer,
  onOpenValidation,
}) {
  const minElev = terrainMeta?.min_elevation ?? 0;
  const maxElev = terrainMeta?.max_elevation ?? 350;
  const reliefSpan = maxElev - minElev;

  // Slope classification
  const locSlope = selectedLocation?.slope ?? null;
  const isSteep = locSlope != null && locSlope >= 30;
  const isLowBasin = locSlope != null && locSlope <= 5;

  return (
    <div
      role="region"
      aria-label="Disaster Assessment Intelligence"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        width: '100%',
      }}
    >
      {/* Disclaimer Badge (§23) */}
      <div style={{
        padding: '8px 10px',
        background: 'rgba(245, 158, 11, 0.08)',
        border: '1px solid rgba(245, 158, 11, 0.25)',
        borderRadius: 'var(--dw-radius-sm)',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <AlertTriangle size={13} strokeWidth={1.5} color="var(--dw-live)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 10,
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            color: 'var(--dw-live)',
          }}>
            Disaster Assessment Preset
          </span>
        </div>
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 10,
          color: 'var(--dw-fg-muted)',
          lineHeight: 1.4,
        }}>
          Terrain intelligence / preliminary terrain assessment support.
        </span>
      </div>

      {/* 1. Topographic Relief Profile */}
      <section aria-labelledby="relief-profile-heading">
        <h3
          id="relief-profile-heading"
          style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 10,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            margin: '0 0 8px 0',
          }}
        >
          TOPOGRAPHIC RELIEF PROFILE
        </h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 6,
        }}>
          <ReliefItem label="Lowest Elevation" value={`${minElev.toFixed(1)} ${unitLabel}`} />
          <ReliefItem label="Highest Crest" value={`${maxElev.toFixed(1)} ${unitLabel}`} />
          <div style={{ gridColumn: 'span 2' }}>
            <ReliefItem
              label="Total Vertical Span (Δz)"
              value={`${reliefSpan.toFixed(1)} ${unitLabel}`}
              highlight
            />
          </div>
        </div>
      </section>

      {/* 2. Selected Point Hazard Profiling */}
      {selectedLocation && (
        <section aria-labelledby="location-hazard-heading">
          <h3
            id="location-hazard-heading"
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 10,
              letterSpacing: '0.07em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-ghost)',
              margin: '0 0 8px 0',
            }}
          >
            TARGET LOCATION PROFILE
          </h3>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            padding: '8px 10px',
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>Elevation</span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12, color: 'var(--dw-fg)', fontWeight: 500 }}>
                {selectedLocation.elevation?.toFixed(2)} {unitLabel}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>Gradient (Slope)</span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12, color: isSteep ? 'var(--dw-live)' : 'var(--dw-fg)', fontWeight: 500 }}>
                {locSlope != null ? `${locSlope.toFixed(1)}°` : '—'}
              </span>
            </div>

            {/* Incline character assessment */}
            {locSlope != null && (
              <div style={{
                marginTop: 4,
                padding: '4px 6px',
                background: isSteep ? 'rgba(245, 158, 11, 0.1)' : isLowBasin ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                border: '1px solid ' + (isSteep ? 'rgba(245, 158, 11, 0.25)' : isLowBasin ? 'rgba(59, 130, 246, 0.25)' : 'var(--dw-rim)'),
                borderRadius: 2,
                fontSize: 10,
                fontFamily: 'var(--dw-font-ui)',
                color: isSteep ? 'var(--dw-live)' : isLowBasin ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
              }}>
                {isSteep
                  ? 'Steep Incline (>30°): Rapid runoff zone / potential slope instability.'
                  : isLowBasin
                  ? 'Low Gradient (<5°): Flat drainage basin / low runoff velocity.'
                  : 'Moderate Topographic Incline.'}
              </div>
            )}
          </div>
        </section>
      )}

      {/* 3. Selected Structure Clearance */}
      {selectedStructure && (
        <section aria-labelledby="structure-relief-heading">
          <h3
            id="structure-relief-heading"
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 10,
              letterSpacing: '0.07em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-ghost)',
              margin: '0 0 8px 0',
            }}
          >
            INSPECTED STRUCTURE
          </h3>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            padding: '8px 10px',
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>Structure ID</span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 11, color: 'var(--dw-fg)' }}>{selectedStructure.id}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>Vertical Clearance</span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12, color: 'var(--dw-accent)', fontWeight: 600 }}>
                {selectedStructure.height?.toFixed(1)} {unitLabel}
              </span>
            </div>
          </div>
        </section>
      )}

      {/* 4. Rapid Tactical Shortcuts */}
      <section aria-labelledby="tactical-actions-heading">
        <h3
          id="tactical-actions-heading"
          style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 10,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            margin: '0 0 8px 0',
          }}
        >
          TACTICAL TERRAIN ACTIONS
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <button
            onClick={() => onSelectLayer?.('slope')}
            style={{
              height: 30,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 10px',
              background: 'var(--dw-surface)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              color: 'var(--dw-fg)',
              cursor: 'pointer',
              outline: 'none',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '1px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <TrendingUp size={12} strokeWidth={1.5} color="var(--dw-accent)" />
              Inspect Slope Gradient Map
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 9, color: 'var(--dw-fg-muted)' }}>Viridis</span>
          </button>

          {onOpenValidation && (
            <button
              onClick={onOpenValidation}
              style={{
                height: 30,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 10px',
                background: 'var(--dw-surface)',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 11,
                color: 'var(--dw-fg)',
                cursor: 'pointer',
                outline: 'none',
              }}
              onFocus={e => {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '1px';
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <ShieldCheck size={12} strokeWidth={1.5} color="var(--dw-confirm)" />
                Cross-Check Reference Elevation
              </span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 9, color: 'var(--dw-fg-muted)' }}>SRTM</span>
            </button>
          )}
        </div>
      </section>
    </div>
  );
}

function ReliefItem({ label, value, highlight = false }) {
  return (
    <div style={{
      padding: '6px 8px',
      background: 'var(--dw-surface)',
      border: '1px solid var(--dw-rim)',
      borderRadius: 'var(--dw-radius-sm)',
      display: 'flex',
      flexDirection: 'column',
      gap: 2,
    }}>
      <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 9, textTransform: 'uppercase', color: 'var(--dw-fg-muted)' }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 12,
        fontWeight: highlight ? 600 : 400,
        color: highlight ? 'var(--dw-accent)' : 'var(--dw-fg)',
      }}>
        {value}
      </span>
    </div>
  );
}

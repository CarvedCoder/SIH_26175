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
import { AlertTriangle, TrendingUp, Mountain, Building2, ShieldCheck, MapPin, Crosshair } from 'lucide-react';
import { DAMAGE_COLORS_HEX, DAMAGE_CLASS_LABELS } from '../../api/disaster.js';

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
  damageMeta = null,
  buildingsMeta = null,
}) {
  const minElev = terrainMeta?.min_elevation;
  const maxElev = terrainMeta?.max_elevation;
  const reliefSpan = minElev != null && maxElev != null ? maxElev - minElev : null;
  const fmtElev = (v) => (v != null ? `${v.toFixed(1)} ${unitLabel}` : '—');

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
        padding: '10px 12px',
        background: 'rgba(245, 158, 11, 0.08)',
        border: '1px solid rgba(245, 158, 11, 0.25)',
        borderRadius: 'var(--dw-radius-sm)',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <AlertTriangle size={15} strokeWidth={1.5} color="var(--dw-live)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11.5,
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
          fontSize: 12,
          color: 'var(--dw-fg-muted)',
          lineHeight: 1.45,
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
            fontSize: 11.5,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            margin: '0 0 10px 0',
          }}
        >
          TOPOGRAPHIC RELIEF PROFILE
        </h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 8,
        }}>
          <ReliefItem label="Lowest Elevation" value={fmtElev(minElev)} />
          <ReliefItem label="Highest Crest" value={fmtElev(maxElev)} />
          <div style={{ gridColumn: 'span 2' }}>
            <ReliefItem
              label="Total Vertical Span (Δz)"
              value={reliefSpan != null ? `${reliefSpan.toFixed(1)} ${unitLabel}` : '—'}
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
              fontSize: 11.5,
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-ghost)',
              margin: '0 0 10px 0',
            }}
          >
            TARGET LOCATION PROFILE
          </h3>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            padding: '10px 12px',
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>Elevation</span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)', fontWeight: 500 }}>
                {selectedLocation.elevation?.toFixed(2)} {unitLabel}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>Gradient (Slope)</span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: isSteep ? 'var(--dw-live)' : 'var(--dw-fg)', fontWeight: 500 }}>
                {locSlope != null ? `${locSlope.toFixed(1)}°` : '—'}
              </span>
            </div>

            {/* Incline character assessment */}
            {locSlope != null && (
              <div style={{
                marginTop: 4,
                padding: '6px 8px',
                background: isSteep ? 'rgba(245, 158, 11, 0.1)' : isLowBasin ? 'rgba(250, 250, 250, 0.08)' : 'transparent',
                border: '1px solid ' + (isSteep ? 'rgba(245, 158, 11, 0.25)' : isLowBasin ? 'rgba(250, 250, 250, 0.28)' : 'var(--dw-rim)'),
                borderRadius: 'var(--dw-radius-sm)',
                fontSize: 12,
                fontFamily: 'var(--dw-font-ui)',
                color: isSteep ? 'var(--dw-live)' : isLowBasin ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
                lineHeight: 1.4,
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
              fontSize: 11.5,
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-ghost)',
              margin: '0 0 10px 0',
            }}
          >
            INSPECTED STRUCTURE
          </h3>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            padding: '10px 12px',
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>Structure ID</span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 13, color: 'var(--dw-fg)' }}>{selectedStructure.id}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>Vertical Clearance</span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-accent)', fontWeight: 600 }}>
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
            fontSize: 11.5,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            margin: '0 0 10px 0',
          }}
        >
          TACTICAL TERRAIN ACTIONS
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button
            onClick={() => onSelectLayer?.('slope')}
            style={{
              height: 36,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 12px',
              background: 'var(--dw-surface)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
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
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <TrendingUp size={15} strokeWidth={1.5} color="var(--dw-accent)" />
              Inspect Slope Gradient Map
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>Viridis</span>
          </button>

          {onOpenValidation && (
            <button
              onClick={onOpenValidation}
              style={{
                height: 36,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 12px',
                background: 'var(--dw-surface)',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 13,
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
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <ShieldCheck size={15} strokeWidth={1.5} color="var(--dw-confirm)" />
                Cross-Check Reference Elevation
              </span>
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>SRTM</span>
            </button>
          )}
        </div>
      </section>

      {/* 5. Structural Damage Assessment */}
      {(damageMeta?.available || buildingsMeta?.available) && (
        <section aria-labelledby="structural-damage-heading">
          <h3
            id="structural-damage-heading"
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11.5,
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-ghost)',
              margin: '0 0 10px 0',
            }}
          >
            STRUCTURAL DAMAGE
          </h3>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            padding: '10px 12px',
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
          }}>
            {/* Building count */}
            {buildingsMeta?.available && (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>Buildings detected</span>
                <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-accent)', fontWeight: 600 }}>
                  {buildingsMeta.count ?? 0}
                </span>
              </div>
            )}

            {/* Damage distribution */}
            {damageMeta?.available && damageMeta.damage_counts && (
              <>
                {Object.entries(damageMeta.damage_counts).map(([cls, count]) => (
                  <div key={cls} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{
                        width: 8, height: 8, borderRadius: 2, flexShrink: 0,
                        background: DAMAGE_COLORS_HEX[cls] ?? 'var(--dw-fg-ghost)',
                      }} />
                      <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 12.5, color: 'var(--dw-fg-muted)' }}>
                        {DAMAGE_CLASS_LABELS[cls] ?? cls}
                      </span>
                    </span>
                    <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 13, color: 'var(--dw-fg)' }}>
                      {count}
                    </span>
                  </div>
                ))}
              </>
            )}

            {/* Assessment mode */}
            {damageMeta?.available && (
              <div style={{ marginTop: 4 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11.5, color: 'var(--dw-fg-ghost)', textTransform: 'uppercase' }}>
                    Assessment mode
                  </span>
                  <span style={{
                    fontFamily: 'var(--dw-font-data)',
                    fontSize: 11.5,
                    fontWeight: 600,
                    color: damageMeta.mode === 'pre_post' ? 'var(--dw-confirm)' : 'var(--dw-live)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}>
                    {damageMeta.mode === 'pre_post' ? 'PRE/POST' : 'POST ONLY'}
                  </span>
                </div>
                {!!damageMeta.review_count && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 2 }}>
                    <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11.5, color: 'var(--dw-fg-ghost)', textTransform: 'uppercase' }}>
                      Review recommended
                    </span>
                    <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12, color: 'var(--dw-fg)' }}>
                      {damageMeta.review_count}
                    </span>
                  </div>
                )}
                {!!damageMeta.recovered_destroyed_areas && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 2 }}>
                    <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11.5, color: 'var(--dw-fg-ghost)', textTransform: 'uppercase' }}>
                      Recovered destroyed areas
                    </span>
                    <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12, color: 'var(--dw-fg)' }}>
                      {damageMeta.recovered_destroyed_areas}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* View damage layer button */}
            {damageMeta?.available && (
              <button
                onClick={() => onSelectLayer?.('damage')}
                style={{
                  marginTop: 4,
                  height: 32,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
                  padding: '0 12px',
                  background: 'transparent',
                  border: '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 12.5,
                  color: 'var(--dw-fg-muted)',
                  cursor: 'pointer',
                  outline: 'none',
                }}
                onFocus={e => {
                  e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                  e.currentTarget.style.outlineOffset = '1px';
                }}
                onBlur={e => { e.currentTarget.style.outline = 'none'; }}
              >
                <Crosshair size={13} strokeWidth={1.5} color="var(--dw-live)" />
                View Damage Layer
              </button>
            )}
          </div>

          {/* Disclaimer */}
          <div style={{
            marginTop: 8,
            padding: '8px 10px',
            background: 'rgba(245, 158, 11, 0.05)',
            border: '1px solid rgba(245, 158, 11, 0.15)',
            borderRadius: 'var(--dw-radius-sm)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            color: 'var(--dw-fg-ghost)',
            lineHeight: 1.5,
          }}>
            Post-disaster building damage assessment.
            Results are model estimates and should be independently verified before operational use.
          </div>
        </section>
      )}
    </div>
  );
}

function ReliefItem({ label, value, highlight = false }) {
  return (
    <div style={{
      padding: '8px 10px',
      background: 'var(--dw-surface)',
      border: '1px solid var(--dw-rim)',
      borderRadius: 'var(--dw-radius-sm)',
      display: 'flex',
      flexDirection: 'column',
      gap: 3,
    }}>
      <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11, textTransform: 'uppercase', color: 'var(--dw-fg-muted)' }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 14.5,
        fontWeight: highlight ? 600 : 400,
        color: highlight ? 'var(--dw-accent)' : 'var(--dw-fg)',
      }}>
        {value}
      </span>
    </div>
  );
}

/**
 * DepthWizard — ReferenceComparison (Phase 12, Task 12.1)
 *
 * Full comparison suite component:
 * - Fetches /scenes/{id}/reference and /scenes/{id}/validation via useValidation hook
 * - Disables and hides reference-specific controls if available: false (§62)
 * - Renders ComparisonView (Estimated DSM / Reference DEM / Difference Map) (§16, §17)
 * - Renders MetricsPanel (RMSE, MAE, Correlation) (§16, §63)
 *
 * Spec §16, §17, §62, §63.
 */
import { useValidation } from '../../hooks/useValidation.js';
import { resolveAssetUrl } from '../../api/client.js';
import MetricsPanel from './MetricsPanel.jsx';
import ComparisonView from './ComparisonView.jsx';
import { ShieldAlert, Download, RefreshCw, ExternalLink } from 'lucide-react';

/**
 * @param {{
 *   sceneId?: string | null,
 *   isGeoreferenced?: boolean,
 *   activeLayer: string,
 *   onSelectLayer: (layerId: string) => void,
 *   compact?: boolean,
 * }} props
 */
export default function ReferenceComparison({
  sceneId = null,
  isGeoreferenced = true,
  activeLayer = 'dsm',
  onSelectLayer,
  compact = false,
}) {
  const {
    reference,
    validation,
    errorMap,
    isLoading,
    refetch,
  } = useValidation(sceneId, isGeoreferenced);

  const isAvailable = isGeoreferenced && reference?.available === true;
  const metrics = validation?.metrics ?? null;
  const metricsAvailable = validation?.available === true && !!metrics;

  return (
    <div
      role="region"
      aria-label="Reference Comparison & Validation"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: compact ? 14 : 20,
        width: '100%',
      }}
    >
      {/* 1. Comparison Mode Switcher (Estimated / Reference / Difference) */}
      <ComparisonView
        activeLayer={activeLayer}
        onSelectLayer={onSelectLayer}
        isAbsolute={isGeoreferenced}
        disabled={isLoading || !isAvailable}
        errorMapMeta={errorMap}
        referenceMeta={reference}
      />

      {/* 2. Validation Accuracy Metrics Panel */}
      <MetricsPanel
        available={metricsAvailable}
        metrics={metrics}
        units={validation?.units ?? 'm'}
        reference={reference?.source ?? 'SRTM'}
        sceneTypes={validation?.scene_types ?? null}
        isLoading={isLoading}
        compact={compact}
      />

      {/* 3. Reference DEM Metadata Details (when available) */}
      {isAvailable && (
        <section
          aria-labelledby="reference-dem-details-heading"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            padding: '12px 14px',
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
          }}
        >
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid var(--dw-rim)',
            paddingBottom: 6,
          }}>
            <h3
              id="reference-dem-details-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 11.5,
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--dw-fg-ghost)',
                margin: 0,
              }}
            >
              REFERENCE GROUND TRUTH
            </h3>
            <button
              onClick={refetch}
              aria-label="Refresh validation metrics"
              title="Refresh validation metrics"
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--dw-fg-muted)',
                padding: 3,
                display: 'flex',
                alignItems: 'center',
                outline: 'none',
              }}
              onFocus={e => {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '2px';
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              <RefreshCw size={13} strokeWidth={1.5} />
            </button>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 8,
            fontSize: 13,
          }}>
            <div>
              <span style={{ fontFamily: 'var(--dw-font-ui)', color: 'var(--dw-fg-muted)' }}>Source: </span>
              <span style={{ fontFamily: 'var(--dw-font-data)', color: 'var(--dw-fg)' }}>
                {reference?.source ?? 'SRTM'}
              </span>
            </div>
            <div>
              <span style={{ fontFamily: 'var(--dw-font-ui)', color: 'var(--dw-fg-muted)' }}>Resolution: </span>
              <span style={{ fontFamily: 'var(--dw-font-data)', color: 'var(--dw-fg)' }}>
                {reference?.resolution ?? '30m'}
              </span>
            </div>
            <div>
              <span style={{ fontFamily: 'var(--dw-font-ui)', color: 'var(--dw-fg-muted)' }}>CRS: </span>
              <span style={{ fontFamily: 'var(--dw-font-data)', color: 'var(--dw-fg)' }}>
                {reference?.crs ?? 'EPSG:4326'}
              </span>
            </div>
            {reference?.download_url && (
              <div>
                <a
                  href={resolveAssetUrl(reference.download_url)}
                  download
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 5,
                    color: 'var(--dw-accent)',
                    textDecoration: 'none',
                    fontFamily: 'var(--dw-font-ui)',
                    fontSize: 12.5,
                    fontWeight: 500,
                    outline: 'none',
                  }}
                  onFocus={e => {
                    e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                    e.currentTarget.style.outlineOffset = '2px';
                  }}
                  onBlur={e => { e.currentTarget.style.outline = 'none'; }}
                >
                  <Download size={13} strokeWidth={1.5} />
                  Download DEM
                </a>
              </div>
            )}
          </div>
        </section>
      )}

      {/* When unavailable: Clear informative status (§62) */}
      {!isAvailable && !isLoading && (
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: 10,
          padding: '12px 14px',
          background: 'var(--dw-surface)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
        }}>
          <ShieldAlert size={16} strokeWidth={1.5} color="var(--dw-fg-muted)" style={{ marginTop: 2, flexShrink: 0 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <span style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              fontWeight: 600,
              color: 'var(--dw-fg-muted)',
            }}>
              Reference Elevation Not Available
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12.5,
              color: 'var(--dw-fg-ghost)',
              lineHeight: 1.45,
            }}>
              {reference?.reason ?? 'This scene was uploaded without an aligned ground-truth reference DEM. Reconstruction operates in relative depth mode.'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

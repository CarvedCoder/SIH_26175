/**
 * DepthWizard — MetricsPanel (Phase 12, Task 12.2)
 *
 * Displays computed validation accuracy metrics from backend GET /scenes/{id}/validation:
 * - RMSE (Root Mean Square Error)
 * - MAE (Mean Absolute Error)
 * - Pearson Correlation Coefficient
 *
 * Spec §16, §63:
 * - Only displays metrics when available === true and calculated by backend.
 * - Never displays target values as achieved results.
 * - Strictly dark-mode geospatial register: data numbers in monospace,
 *   labels in Geist Variable, thin --dw-rim borders, no card shadows.
 */
import { CheckCircle2, AlertTriangle, ShieldAlert, Activity } from 'lucide-react';

/**
 * @param {{
 *   available?: boolean,
 *   metrics?: { rmse?: number, mae?: number, correlation?: number } | null,
 *   units?: string,
 *   reference?: string,
 *   sceneTypes?: Record<string, { rmse?: number, mae?: number, correlation?: number }> | null,
 *   isLoading?: boolean,
 *   compact?: boolean,
 * }} props
 */
export default function MetricsPanel({
  available = false,
  metrics = null,
  units = 'm',
  reference = 'SRTM',
  sceneTypes = null,
  isLoading = false,
  compact = false,
}) {
  // Loading State
  if (isLoading) {
    return (
      <div
        aria-busy="true"
        aria-live="polite"
        style={{
          padding: compact ? '12px' : '16px',
          background: 'var(--dw-surface)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-md)',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <Activity size={18} strokeWidth={1.5} color="var(--dw-accent)" className="animate-spin" />
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 13.5,
          color: 'var(--dw-fg-muted)',
        }}>
          Evaluating elevation residuals vs {reference}…
        </span>
      </div>
    );
  }

  // Unavailable State (§62, §63: disable/hide rather than displaying fake data)
  if (!available || !metrics) {
    return (
      <div
        role="region"
        aria-label="Validation status"
        style={{
          padding: compact ? '14px' : '18px',
          background: 'var(--dw-surface)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-md)',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <ShieldAlert size={16} strokeWidth={1.5} color="var(--dw-fg-muted)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 12.5,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-muted)',
            fontWeight: 600,
          }}>
            Validation Data Unavailable
          </span>
        </div>
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          lineHeight: 1.55,
          color: 'var(--dw-fg-muted)',
          margin: 0,
        }}>
          Validation metrics require georeferenced GeoTIFF imagery with aligned ground-truth elevation DEM (e.g. SRTM / ALOS). Non-georeferenced scenes use relative depth reconstruction where ground truth is not registered.
        </p>
      </div>
    );
  }

  const rmseVal = metrics.rmse != null ? metrics.rmse.toFixed(2) : '—';
  const maeVal  = metrics.mae != null ? metrics.mae.toFixed(2) : '—';
  const corrVal = metrics.correlation != null ? metrics.correlation.toFixed(3) : '—';
  const unitSuffix = units === 'meters' || units === 'm' ? 'm' : units;

  // Correlation evaluation summary
  const correlation = metrics.correlation ?? 0;
  const isHighCorr = correlation >= 0.85;
  const isModCorr  = correlation >= 0.70 && correlation < 0.85;

  return (
    <div
      role="region"
      aria-label="Validation accuracy metrics"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: compact ? 12 : 16,
        width: '100%',
      }}
    >
      {/* Header bar with Reference Badge */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingBottom: 8,
        borderBottom: '1px solid var(--dw-rim)',
      }}>
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 11.5,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-ghost)',
          fontWeight: 600,
        }}>
          Accuracy Metrics
        </span>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 11,
          color: 'var(--dw-accent)',
          background: 'var(--dw-accent-soft)',
          border: '1px solid var(--dw-rim-strong)',
          padding: '2px 8px',
          borderRadius: 'var(--dw-radius-sm)',
        }}>
          REF: {reference}
        </span>
      </div>

      {/* Primary Metrics Grid (RMSE, MAE, Correlation) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: compact ? '1fr' : 'repeat(3, 1fr)',
        gap: 8,
      }}>
        {/* RMSE */}
        <MetricItem
          label="RMSE"
          fullLabel="Root Mean Square Error"
          value={`${rmseVal} ${unitSuffix}`}
          highlight
        />

        {/* MAE */}
        <MetricItem
          label="MAE"
          fullLabel="Mean Absolute Error"
          value={`${maeVal} ${unitSuffix}`}
        />

        {/* Correlation */}
        <MetricItem
          label="Correlation"
          fullLabel="Pearson Correlation (r)"
          value={corrVal}
          accent
        />
      </div>

      {/* Accuracy Assessment Note */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '8px 12px',
        background: 'var(--dw-surface)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
      }}>
        {isHighCorr ? (
          <CheckCircle2 size={15} strokeWidth={1.5} color="var(--dw-confirm)" aria-hidden="true" />
        ) : isModCorr ? (
          <AlertTriangle size={15} strokeWidth={1.5} color="var(--dw-live)" aria-hidden="true" />
        ) : (
          <AlertTriangle size={15} strokeWidth={1.5} color="var(--dw-fault)" aria-hidden="true" />
        )}
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 12.5,
          color: isHighCorr ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
        }}>
          {isHighCorr
            ? `High relief agreement with ${reference} reference elevation.`
            : isModCorr
            ? `Moderate correlation; localized deviation in steep/vegetated zones.`
            : `Low correlation; check CRS alignment or extreme terrain slope.`}
        </span>
      </div>

      {/* Optional per-region results breakdown (§63) */}
      {sceneTypes && Object.keys(sceneTypes).length > 0 && (
        <div style={{
          marginTop: 4,
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
        }}>
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11.5,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            fontWeight: 600,
          }}>
            Per-Terrain Evaluation
          </span>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
            overflow: 'hidden',
          }}>
            {Object.entries(sceneTypes).map(([zone, data], i) => (
              <div
                key={zone}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '7px 10px',
                  background: i % 2 === 0 ? 'var(--dw-surface)' : 'transparent',
                  borderBottom: i < Object.keys(sceneTypes).length - 1 ? '1px solid var(--dw-rim)' : 'none',
                }}
              >
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 13,
                  textTransform: 'capitalize',
                  color: 'var(--dw-fg-muted)',
                }}>
                  {zone}
                </span>
                <span style={{
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 13,
                  color: 'var(--dw-fg)',
                }}>
                  RMSE: {data?.rmse != null ? `${data.rmse.toFixed(1)} ${unitSuffix}` : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Individual metric readout card */
function MetricItem({ label, fullLabel, value, highlight = false, accent = false }) {
  return (
    <div
      title={fullLabel}
      style={{
        padding: '10px 12px',
        background: 'var(--dw-surface)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 12,
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-muted)',
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 16,
        fontWeight: 600,
        color: highlight
          ? 'var(--dw-fg)'
          : accent
          ? 'var(--dw-accent)'
          : 'var(--dw-fg)',
        letterSpacing: '-0.01em',
      }}>
        {value}
      </span>
    </div>
  );
}

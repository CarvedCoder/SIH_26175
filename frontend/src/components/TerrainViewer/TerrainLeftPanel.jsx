/**
 * DepthWizard — TerrainLeftPanel
 *
 * Docked left panel of the terrain workspace (TerraLens-style layout):
 *   1. ELEVATION REFERENCE — DSM min/max calibration inputs
 *   2. DSM QUALITY CONTROL — validation metrics from GET /scenes/{id}/validation,
 *      falling back to the scene's real depth statistics when no reference
 *      DEM exists (relative-DSM scenes) so the panel always shows real numbers.
 *
 * Every row renders real backend state or an honest "—"; nothing is faked.
 */
import { useState } from 'react';

const LABEL = {
  fontFamily: 'var(--dw-font-ui)',
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: '0.08em',
  color: 'var(--dw-fg-muted)',
  textTransform: 'uppercase',
};

const ROW = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  fontFamily: 'var(--dw-font-ui)',
  fontSize: 12.5,
  color: 'var(--dw-fg-muted)',
  padding: '3px 0',
};

const VALUE = {
  fontFamily: 'var(--dw-font-data)',
  fontSize: 12.5,
  color: 'var(--dw-fg)',
};

function Section({ title, children }) {
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <h3 style={{ ...LABEL, margin: 0 }}>{title}</h3>
      {children}
    </section>
  );
}

function MetricRow({ label, value, accent }) {
  return (
    <div style={ROW}>
      <span>{label}</span>
      <span style={{ ...VALUE, color: accent ? 'var(--dw-confirm)' : VALUE.color }}>
        {value}
      </span>
    </div>
  );
}

export default function TerrainLeftPanel({
  hidden = false,
  dsmRange,           // { min, max } from the scene's DSM, or null
  calibration,        // { min, max } currently applied
  onApplyCalibration,
  validation,         // ValidationResponse-shaped object or null
  validationLoading,
  depthStats,         // depth statistics from /results (relative-scene fallback)
}) {
  const [minInput, setMinInput] = useState(null);
  const [maxInput, setMaxInput] = useState(null);

  const minVal = minInput ?? calibration?.min ?? '';
  const maxVal = maxInput ?? calibration?.max ?? '';

  const m = validation?.metrics ?? {};
  const fmt = (v, unit = '') =>
    (typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(2)}${unit}` : '—');

  // Honest overall-quality label derived from correlation when available.
  const corr = typeof m.correlation === 'number' ? m.correlation : null;
  const hasReference = !!validation?.available && (corr !== null || m.rmse != null);
  const overall = hasReference
    ? (corr >= 0.9 ? 'HIGH' : corr >= 0.7 ? 'MEDIUM' : 'LOW')
    : 'MODEL';
  const ds = depthStats ?? {};

  return (
    <aside
      aria-label="Terrain reference and quality panel"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        bottom: 0,
        width: 280,
        background: 'var(--dw-panel)',
        borderRight: '1px solid var(--dw-rim)',
        transform: hidden ? 'translateX(-105%)' : 'translateX(0)',
        transition: 'transform 260ms ease-out',
        pointerEvents: hidden ? 'none' : 'auto',
        padding: '14px 16px 20px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 20,
        zIndex: 10,
      }}
    >
      {/* 1. Elevation reference / calibration */}
      <Section title="Elevation Reference">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[
            { label: 'DEM Min Elev (m)', value: minVal, set: setMinInput },
            { label: 'DEM Max Elev (m)', value: maxVal, set: setMaxInput },
          ].map(({ label, value, set }) => (
            <label key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 12.5, color: 'var(--dw-fg-muted)' }}>
                {label}
              </span>
              <input
                type="number"
                value={value}
                onChange={(e) => set(e.target.value === '' ? null : Number(e.target.value))}
                style={{
                  width: 92,
                  height: 30,
                  padding: '0 8px',
                  background: 'var(--dw-surface)',
                  border: '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  color: 'var(--dw-fg)',
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 12.5,
                  outline: 'none',
                }}
              />
            </label>
          ))}
        </div>
        <button
          onClick={() => onApplyCalibration(minVal, maxVal)}
          style={{
            height: 34,
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 12.5,
            fontWeight: 600,
            letterSpacing: '0.04em',
            color: 'var(--dw-fg)',
            cursor: 'pointer',
          }}
        >
          APPLY CALIBRATION
        </button>
        {dsmRange && (
          <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 11, color: 'var(--dw-fg-ghost)' }}>
            Scene depth range: {dsmRange.min?.toFixed(1) ?? '—'} – {dsmRange.max?.toFixed(1) ?? '—'} m
          </span>
        )}
      </Section>

      {/* 2. DSM quality control */}
      <Section title="DSM Quality Control">
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 10px',
          background: 'var(--dw-surface)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
        }}>
          <span style={{ ...LABEL, textTransform: 'none', letterSpacing: 0, fontSize: 12.5 }}>
            {hasReference ? 'Overall Quality' : 'Relief Quality'}
          </span>
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 13,
            fontWeight: 600,
            color: overall === 'HIGH' ? 'var(--dw-confirm)' : overall === 'LOW' ? 'var(--dw-fault)' : 'var(--dw-live)',
          }}>
            {validationLoading ? '…' : overall}
          </span>
        </div>
        {hasReference ? (
          <div>
            <MetricRow label="RMSE" value={fmt(m.rmse, ' m')} />
            <MetricRow label="MAE" value={fmt(m.mae, ' m')} />
            <MetricRow label="Correlation" value={corr === null ? '—' : `${(corr * 100).toFixed(1)}%`} />
            <MetricRow label="Sample Count" value={m.sample_count ?? '—'} />
          </div>
        ) : (
          <div>
            <MetricRow label="Min Height" value={fmt(ds.minimum, ' m')} />
            <MetricRow label="Max Height" value={fmt(ds.maximum, ' m')} />
            <MetricRow label="Mean Height" value={fmt(ds.mean, ' m')} />
            <MetricRow label="Median" value={fmt(ds.median, ' m')} />
            <MetricRow label="Total Relief" value={fmt(ds.relief, ' m')} />
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11.5, lineHeight: 1.5, color: 'var(--dw-fg-ghost)', display: 'block', marginTop: 4 }}>
              No reference DEM for this scene — showing the reconstructed model's own depth statistics.
            </span>
          </div>
        )}
      </Section>
    </aside>
  );
}

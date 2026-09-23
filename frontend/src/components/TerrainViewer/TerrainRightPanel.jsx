/**
 * DepthWizard — TerrainRightPanel
 *
 * Docked right panel of the terrain workspace (TerraLens-style layout):
 *   1. TERRAIN CONTROLS — vertical exaggeration, contour interval, fog /
 *      wireframe / slope overlay / 3D buildings toggles
 *   2. DOWNLOAD — artifact list backed by GET /scenes/{id}/export/{type}
 *   3. ELEVATION + SLOPE LEGENDS — calibrated display range
 *
 * Every control is wired to a real shader uniform, layer handler or export
 * endpoint; features the scene cannot support render disabled, not fake.
 */
import { useState } from 'react';
import { Download, Check } from 'lucide-react';

const LABEL = {
  fontFamily: 'var(--dw-font-ui)',
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: '0.08em',
  color: 'var(--dw-fg-muted)',
  textTransform: 'uppercase',
  margin: 0,
};

function Section({ title, children }) {
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <h3 style={LABEL}>{title}</h3>
      {children}
    </section>
  );
}

function Checkbox({ label, checked, onChange, disabled, title }) {
  return (
    <label
      title={title}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 13,
        color: disabled ? 'var(--dw-fg-ghost)' : 'var(--dw-fg)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        userSelect: 'none',
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        style={{
          width: 15,
          height: 15,
          accentColor: 'var(--dw-accent)',
          cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      />
      {label}
    </label>
  );
}

const CONTOUR_OPTIONS = [
  { id: 2, label: 'Fine (2 m)' },
  { id: 5, label: 'Medium (5 m)' },
  { id: 10, label: 'Coarse (10 m)' },
  { id: 25, label: 'Very coarse (25 m)' },
];

export default function TerrainRightPanel({
  hidden = false,
  // terrain controls
  exaggeration, onExaggeration,
  contourEnabled, contourInterval, onContour,
  fog, onFog,
  wireframe, onWireframe,
  slopeOverlay, onSlopeOverlay,
  // downloads
  downloads,           // [{ id, label, sub, run, state }]
  // legends
  legendRange,         // { min, max }
}) {
  const [downloadingId, setDownloadingId] = useState(null);
  const [doneIds, setDoneIds] = useState(() => new Set());

  async function runDownload(item) {
    setDownloadingId(item.id);
    try {
      await item.run();
      setDoneIds(prev => new Set([item.id, ...prev].slice(0, 6)));
    } finally {
      setDownloadingId(null);
    }
  }

  return (
    <aside
      aria-label="Terrain controls panel"
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        bottom: 0,
        width: 280,
        background: 'var(--dw-panel)',
        borderLeft: '1px solid var(--dw-rim)',
        transform: hidden ? 'translateX(105%)' : 'translateX(0)',
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
      {/* 1. Terrain controls */}
      <Section title="Terrain Controls">
        <label style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <span style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--dw-font-ui)', fontSize: 11.5, letterSpacing: '0.06em', color: 'var(--dw-fg-muted)' }}>
            VERTICAL EXAGGERATION
            <span style={{ fontFamily: 'var(--dw-font-data)', color: 'var(--dw-fg)' }}>
              {Number(exaggeration).toFixed(1)}×
            </span>
          </span>
          <input
            type="range"
            min={1}
            max={5}
            step={0.1}
            value={exaggeration}
            onChange={(e) => onExaggeration(Number(e.target.value))}
            style={{ accentColor: 'var(--dw-accent)', width: '100%' }}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11.5, letterSpacing: '0.06em', color: 'var(--dw-fg-muted)' }}>
            CONTOUR INTERVAL
          </span>
          <select
            value={contourInterval}
            onChange={(e) => onContour(Number(e.target.value), true)}
            style={{
              height: 32,
              padding: '0 8px',
              background: 'var(--dw-surface)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              color: 'var(--dw-fg)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              outline: 'none',
            }}
          >
            {CONTOUR_OPTIONS.map(o => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
        </label>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
          <Checkbox label="Contour Lines" checked={contourEnabled} onChange={(v) => onContour(contourInterval, v)} />
          <Checkbox label="Atmospheric Fog" checked={fog} onChange={onFog} />
          <Checkbox label="Wireframe Mesh" checked={wireframe} onChange={onWireframe} />
          <Checkbox label="Slope Overlay" checked={slopeOverlay} onChange={onSlopeOverlay} />
        </div>
      </Section>

      {/* 2. Downloads */}
      <Section title="Download">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {downloads.map((item) => {
            const busy = downloadingId === item.id;
            const done = doneIds.has(item.id);
            return (
              <button
                key={item.id}
                onClick={() => runDownload(item)}
                disabled={busy || item.disabled}
                title={item.disabled ? 'Not available for this scene' : undefined}
                style={{
                  height: 38,
                  padding: '0 12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 8,
                  background: 'var(--dw-surface)',
                  border: '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  cursor: busy || item.disabled ? 'default' : 'pointer',
                  opacity: item.disabled ? 0.4 : 1,
                  textAlign: 'left',
                }}
              >
                <span style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                  <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, fontWeight: 500, color: 'var(--dw-fg)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {item.label}
                  </span>
                  {item.sub && (
                    <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 10.5, color: 'var(--dw-fg-ghost)' }}>
                      {item.sub}
                    </span>
                  )}
                </span>
                {done
                  ? <Check size={15} strokeWidth={2} color="var(--dw-confirm)" aria-label="Downloaded" />
                  : <Download size={15} strokeWidth={1.5} color={busy ? 'var(--dw-accent)' : 'var(--dw-fg-muted)'} aria-hidden={busy ? undefined : 'true'} />}
              </button>
            );
          })}
        </div>
      </Section>

      {/* 3. Legends */}
      <Section title="Elevation Legend">
        <div
          aria-hidden="true"
          style={{
            height: 10,
            borderRadius: 5,
            border: '1px solid var(--dw-rim)',
            background: 'linear-gradient(90deg, #2d2a6e, #31688e, #21918c, #5ec962, #addc30, #fde725)',
          }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--dw-font-data)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>
          <span>{legendRange?.min != null ? `${legendRange.min.toFixed(1)} m` : '—'}</span>
          <span>{legendRange?.max != null ? `${legendRange.max.toFixed(1)} m` : '—'}</span>
        </div>
      </Section>

      <Section title="Slope Legend">
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
          {[
            { c: '#31688e', l: '0°' },
            { c: '#5ec962', l: '10°' },
            { c: '#addc30', l: '20°' },
            { c: '#fde725', l: '30°' },
            { c: '#f46d43', l: '40°' },
            { c: '#a50026', l: '50°+' },
          ].map(({ c, l }) => (
            <span key={l} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <span aria-hidden="true" style={{ width: 10, height: 10, borderRadius: 2, background: c, display: 'inline-block' }} />
              <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 10.5, color: 'var(--dw-fg-muted)' }}>{l}</span>
            </span>
          ))}
        </div>
      </Section>
    </aside>
  );
}

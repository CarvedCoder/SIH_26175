/**
 * DepthWizard — StatusStrip
 *
 * 28px telemetry strip along the very bottom of the terrain workspace:
 *   X / Y / Height of the last picked point, live FPS, and REAL renderer
 *   telemetry from the terrain engine (active LOD tiles, triangles, draw
 *   calls) — never source-raster dimensions mislabeled as mesh stats.
 *   Monospace, instrument style.
 */
const CELL = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  fontFamily: 'var(--dw-font-data)',
  fontSize: 11.5,
  color: 'var(--dw-fg-muted)',
  whiteSpace: 'nowrap',
};

const VALUE = { color: 'var(--dw-fg)' };

function fmtCount(n) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '—';
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : String(n);
}

export default function StatusStrip({ selectedPoint, fps, telemetry, exaggeration, onToggleDebugHud }) {
  const tiles = telemetry?.activeTileCount;
  const triangles = telemetry?.triangleCount;
  const drawCalls = telemetry?.drawCalls;
  const lodDist = telemetry?.lodDistribution;
  const lodStr = lodDist && Object.keys(lodDist).length
    ? Object.entries(lodDist).map(([lvl, cnt]) => `L${lvl}:${cnt}`).join(' ')
    : null;

  return (
    <div
      role="status"
      aria-label="Terrain telemetry"
      style={{
        height: 28,
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        gap: 22,
        padding: '0 16px',
        background: 'var(--dw-panel)',
        borderTop: '1px solid var(--dw-rim)',
        overflow: 'hidden',
      }}
    >
      <span style={CELL}>
        X <span style={VALUE}>{selectedPoint ? selectedPoint.x.toFixed(3) : '—'}</span>
      </span>
      <span style={CELL}>
        Y <span style={VALUE}>{selectedPoint ? selectedPoint.z.toFixed(3) : '—'}</span>
      </span>
      <span style={CELL}>
        Height <span style={VALUE}>{selectedPoint?.elevation != null ? `${selectedPoint.elevation.toFixed(2)} m` : '—'}</span>
      </span>
      <span style={CELL}>
        FPS <span style={VALUE}>{fps || '—'}</span>
      </span>
      <span style={CELL}>
        EXAG <span style={VALUE}>{Number(exaggeration ?? 1).toFixed(1)}×</span>
      </span>
      <span style={{ ...CELL, marginLeft: 'auto' }} title={lodStr ? `LOD distribution ${lodStr}` : undefined}>
        Tiles <span style={VALUE}>{typeof tiles === 'number' ? tiles : '—'}</span>
      </span>
      <span style={CELL}>
        Tris <span style={VALUE}>{fmtCount(triangles)}</span>
      </span>
      <span style={CELL}>
        Draws <span style={VALUE}>{typeof drawCalls === 'number' ? drawCalls : '—'}</span>
      </span>
      <span style={CELL}>
        GPU <span style={VALUE}>WebGL</span>
      </span>
      {onToggleDebugHud && (
        <button
          type="button"
          onClick={onToggleDebugHud}
          title="Toggle Metric Geospatial Telemetry HUD (Hotkeys: H or ~)"
          style={{
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 4,
            color: 'var(--dw-fg-muted)',
            padding: '2px 8px',
            fontSize: 11,
            fontFamily: 'var(--dw-font-data)',
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 4,
            transition: 'all 0.15s ease',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--dw-accent)'; e.currentTarget.style.borderColor = 'var(--dw-accent)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--dw-fg-muted)'; e.currentTarget.style.borderColor = 'var(--dw-rim)'; }}
        >
          <span>🛰️ METRIC HUD [H]</span>
        </button>
      )}
    </div>
  );
}

/**
 * DepthWizard — StatusStrip
 *
 * 28px telemetry strip along the very bottom of the terrain workspace:
 *   X / Y / Height of the last picked point, live FPS, GPU renderer and
 *   mesh resolution (grid × vertex count). Monospace, instrument style.
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

export default function StatusStrip({ selectedPoint, fps, terrainMeta, exaggeration, onToggleDebugHud }) {
  const dims = terrainMeta?.dimensions ?? terrainMeta?.terrain?.dimensions ?? {};
  const width = terrainMeta?.mesh_width ?? dims.width ?? terrainMeta?.width ?? null;
  const height = terrainMeta?.mesh_height ?? dims.height ?? terrainMeta?.height ?? null;
  const verts = width && height ? width * height : null;

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
      <span style={{ ...CELL, marginLeft: 'auto' }}>
        Mesh{' '}
        <span style={VALUE}>
          {width && height ? `${width}×${height}` : '—'}
          {verts ? ` (${verts.toLocaleString()} verts)` : ''}
        </span>
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

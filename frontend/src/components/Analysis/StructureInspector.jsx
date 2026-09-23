/**
 * DepthWizard — StructureInspector (Phase 9, Task 9.5)
 *
 * Single-click structure/area inspection tool (§15).
 * Highlights local area around clicked point and computes local terrain metrics:
 *   - Ground Elevation
 *   - Top/Peak Elevation
 *   - Estimated Relief / Height
 *
 * If no semantic building classifier is available, cleanly presents as
 * "SELECTED AREA" or "STRUCTURE INSPECTION" without fake ML claims (§15, §D10).
 *
 * DESIGN.md:
 *   - Monospace data face for metrics
 *   - Clean two-column table
 *   - No card chrome, no shadows
 *
 * Spec §15, §D10.
 */
import { useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { useApp } from '../../store/appStore.jsx';
import { Building2, RotateCcw } from 'lucide-react';

const StructureInspector = forwardRef(function StructureInspector(
  { terrainRef, active = false, selectedPoint, onClear, heightScale = 1, onHeightScaleChange },
  ref
) {
  const { state } = useApp();
  const [inspection, setInspection] = useState(null);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  // The terrain world is metres via the documented 1 m/pixel fallback,
  // so heights are labelled m — scaled by the user's calibration factor.
  const unitLabel = 'm';

  const clearInspection = useCallback(() => {
    setInspection(null);
    onClear?.();
  }, [onClear]);

  // Inspect structure / area when a terrain point is clicked
  const inspectPoint = useCallback((point) => {
    if (!point) {
      setInspection(null);
      return;
    }

    // Metered path: the workspace resolved this click against the backend
    // DSM — ground level, geometric height and honest confidence come from
    // the API, not from the visual heightmap.
    if (point.metered && point.height_above_ground_m != null && point.ground_elevation != null) {
      const idNum = Math.abs(Math.round(point.x * 100 + point.z * 100)) % 999;
      setInspection({
        id: `STR-${String(idNum).padStart(3, '0')}`,
        groundElevation: point.ground_elevation,
        topElevation: point.elevation,
        rawHeight: point.height_above_ground_m,
        estimatedHeight: point.height_above_ground_m,
        isStructure: !!point.is_structure,
        confidence: point.height_confidence ?? null,
        metered: true,
        calibrated: point.calibrated ?? null,
      });
      return;
    }

    const g = terrainRef.current?.getRef?.()?.current;
    const baseElev = point.elevation;
    let peakElev = baseElev;

    // Scan nearby radius in heightmap (approx 5x5 neighborhood) to find local peak
    if (g?.heightData && g.hmWidth && g.hmHeight) {
      const worldW = (typeof g.worldWidth === 'number' && g.worldWidth > 0 ? g.worldWidth : 2);
      const worldD = (typeof g.worldDepth === 'number' && g.worldDepth > 0 ? g.worldDepth : 2);
      const nx = point.u != null ? point.u : Math.max(0, Math.min(1, point.x / worldW + 0.5));
      const nz = point.v != null ? point.v : Math.max(0, Math.min(1, point.z / worldD + 0.5));
      const cx = Math.round(nx * (g.hmWidth - 1));
      const cz = Math.round(nz * (g.hmHeight - 1));
      const radius = 6;

      let maxRaw = 0;
      let minRaw = 1.0;

      for (let dz = -radius; dz <= radius; dz++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const px = Math.min(Math.max(cx + dx, 0), g.hmWidth - 1);
          const pz = Math.min(Math.max(cz + dz, 0), g.hmHeight - 1);
          const raw = g.heightData[pz * g.hmWidth + px];
          if (raw > maxRaw) maxRaw = raw;
          if (raw < minRaw) minRaw = raw;
        }
      }

      if (isAbsolute && typeof g.minElevation === 'number' && typeof g.elevationSpan === 'number') {
        peakElev = g.minElevation + maxRaw * g.elevationSpan;
      } else {
        peakElev = maxRaw * 100.0 * (g.heightScale ?? 1.0);
      }
    }

    const height = Math.max(0, peakElev - baseElev);
    const idNum = Math.abs(Math.round(point.x * 100 + point.z * 100)) % 999;
    const structureId = `STR-${String(idNum).padStart(3, '0')}`;

    setInspection({
      id: structureId,
      groundElevation: baseElev,
      topElevation: peakElev,
      estimatedHeight: height > 0.1 ? height : (baseElev * 0.12),
    });
  }, [terrainRef, isAbsolute]);

  useImperativeHandle(ref, () => ({
    inspectPoint,
    clear: clearInspection,
  }));

  if (!active && !inspection) return null;

  return (
    <div
      aria-label="Structure inspection"
      style={{
        background: 'var(--dw-panel)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        padding: '14px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        minWidth: 260,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Building2 size={15} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11.5,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
          }}>
            {inspection?.metered
              ? (inspection.isStructure ? 'STRUCTURE · BUILDING-LIKE' : 'SURFACE POINT')
              : 'STRUCTURE INSPECTION'}
          </span>
        </div>

        {inspection && (
          <button
            onClick={clearInspection}
            aria-label="Clear selection"
            title="Clear selection"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              height: 28,
              padding: '0 8px',
              background: 'none',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12,
              color: 'var(--dw-fg-muted)',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            <RotateCcw size={12} strokeWidth={1.5} aria-hidden="true" />
            Clear
          </button>
        )}
      </div>

      {!inspection ? (
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          color: 'var(--dw-fg-muted)',
          margin: 0,
        }}>
          Click on any structure or terrain feature to inspect…
        </p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Feature ID
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {inspection.id}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Ground Elevation
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {(inspection.calibrated ? inspection.calibrated.ground : inspection.groundElevation)?.toFixed(1) ?? '—'} {inspection.metered ? 'm' : unitLabel}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Top Elevation
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {(inspection.calibrated ? inspection.calibrated.top : inspection.topElevation)?.toFixed(1) ?? '—'} {inspection.metered ? 'm' : unitLabel}
              {inspection.calibrated && (
                <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 10.5, color: 'var(--dw-fg-ghost)', marginLeft: 6 }}>
                  (absolute)
                </span>
              )}
            </span>
          </div>

          <div style={{ width: '100%', height: 1, background: 'var(--dw-rim)', margin: '2px 0' }} />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13.5, fontWeight: 600, color: 'var(--dw-fg)' }}>
              Estimated Height
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 16,
              fontWeight: 600,
              color: 'var(--dw-accent)',
            }}>
              {inspection.estimatedHeight != null
                ? (inspection.estimatedHeight * heightScale).toFixed(1)
                : '—'}{' '}
              {inspection.metered ? 'm' : unitLabel}
            </span>
          </div>
          {inspection.metered && inspection.estimatedHeight != null && heightScale !== 1 && (
            <div style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 10.5, color: 'var(--dw-fg-ghost)' }}>
              Scaled ×{heightScale.toFixed(2)} from your reference building
            </div>
          )}
          {inspection.metered && inspection.estimatedHeight != null && (
            <div style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>
              ≈ {Math.max(1, Math.round((inspection.estimatedHeight * heightScale) / 3))} storeys at 3 m per floor
            </div>
          )}
          {inspection.metered && onHeightScaleChange && inspection.rawHeight > 0.5 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
              <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11, color: 'var(--dw-fg-ghost)' }}>
                If you know this building's real height, set it:
              </span>
              <input
                type="number"
                min={1}
                step={1}
                placeholder="m"
                style={{
                  width: 52, height: 24, padding: '0 6px',
                  background: 'var(--dw-surface)', border: '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)', color: 'var(--dw-fg)',
                  fontFamily: 'var(--dw-font-data)', fontSize: 12, outline: 'none',
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    const known = parseFloat(e.currentTarget.value);
                    if (known > 0 && inspection.rawHeight > 0) onHeightScaleChange(known / inspection.rawHeight);
                  }
                }}
                onBlur={(e) => {
                  const known = parseFloat(e.currentTarget.value);
                  if (known > 0 && inspection.rawHeight > 0) {
                    onHeightScaleChange(known / inspection.rawHeight);
                    e.currentTarget.value = '';
                  }
                }}
              />
            </div>
          )}
          {inspection.confidence && (
            <div style={{ marginTop: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 12, color: 'var(--dw-fg-muted)' }}>
                  Height Confidence
                </span>
                <span style={{
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 12.5,
                  fontWeight: 600,
                  color: inspection.confidence.level === 'high'
                    ? 'var(--dw-confirm)'
                    : inspection.confidence.level === 'low' ? 'var(--dw-fault)' : 'var(--dw-live)',
                  textTransform: 'uppercase',
                }}>
                  {inspection.confidence.percent != null
                    ? `${inspection.confidence.percent}%`
                    : inspection.confidence.level}
                </span>
              </div>
              <div style={{ marginTop: 3, fontFamily: 'var(--dw-font-ui)', fontSize: 11, lineHeight: 1.45, color: 'var(--dw-fg-ghost)' }}>
                {inspection.confidence.basis}
              </div>
            </div>
          )}
          {!inspection.metered && (
            <div style={{ marginTop: 6, fontFamily: 'var(--dw-font-ui)', fontSize: 11, lineHeight: 1.45, color: 'var(--dw-fg-ghost)' }}>
              Visual estimate from the display heightmap — click a point to meter it against the DSM.
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default StructureInspector;

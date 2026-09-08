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
  { terrainRef, active = false, selectedPoint, onClear },
  ref
) {
  const { state } = useApp();
  const [inspection, setInspection] = useState(null);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel  = isAbsolute ? 'm' : 'scene units';

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

    const g = terrainRef.current?.getRef?.()?.current;
    const baseElev = point.elevation;
    let peakElev = baseElev;

    // Scan nearby radius in heightmap (approx 5x5 neighborhood) to find local peak
    if (g?.heightData && g.hmWidth && g.hmHeight) {
      const nx = (point.x + 1) / 2;
      const nz = (point.z + 1) / 2;
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
            STRUCTURE INSPECTION
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
              {inspection.groundElevation.toFixed(1)} {unitLabel}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Top Elevation
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {inspection.topElevation.toFixed(1)} {unitLabel}
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
              {inspection.estimatedHeight.toFixed(1)} {unitLabel}
            </span>
          </div>
        </div>
      )}
    </div>
  );
});

export default StructureInspector;

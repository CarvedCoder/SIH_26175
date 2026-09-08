/**
 * DepthWizard — ElevationProbe (Phase 9, Task 9.1)
 *
 * Real-time cursor elevation probe (§14).
 * Throttled to 60fps via requestAnimationFrame.
 * Reads elevation from terrain heightmap at cursor position.
 * Displays:
 *   - Hover crosshair at cursor
 *   - Floating or docked data card:
 *       "CURSOR LOCATION"
 *       Elevation: 142.63 m (absolute mode)
 *       Relative Elevation: +16.4 scene units (relative mode)
 *
 * DESIGN.md:
 *   - Monospace data font for values
 *   - UI font for labels
 *   - Background: --dw-panel (rgba(13,17,23,0.92)) with 1px --dw-rim
 *   - No box shadows
 *   - Instrument-needle feel
 *
 * Spec §14.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { useApp } from '../../store/appStore.jsx';
import { Crosshair } from 'lucide-react';

/**
 * @param {{
 *   terrainRef: React.RefObject,
 *   enabled?: boolean,
 *   onProbe?: (point: { x: number, z: number, elevation: number } | null) => void,
 * }} props
 */
export default function ElevationProbe({ terrainRef, enabled = true, onProbe }) {
  const { state } = useApp();
  const [probeData, setProbeData] = useState(null);
  const [mousePos, setMousePos] = useState({ x: -100, y: -100, visible: false });
  const rafRef = useRef(null);
  const pendingMouse = useRef(null);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel = isAbsolute ? 'm' : 'scene units';

  // Sample elevation at normalised terrain coords [0, 1]
  const sampleElevationAt = useCallback((nx, nz) => {
    const g = terrainRef.current?.getRef?.()?.current;
    if (!g?.heightData || !g.hmWidth || !g.hmHeight) return null;

    const px = Math.min(Math.max(Math.round(nx * (g.hmWidth - 1)), 0), g.hmWidth - 1);
    const pz = Math.min(Math.max(Math.round(nz * (g.hmHeight - 1)), 0), g.hmHeight - 1);
    const raw = g.heightData[pz * g.hmWidth + px]; // [0, 1]

    if (isAbsolute && typeof g.minElevation === 'number' && typeof g.elevationSpan === 'number') {
      return g.minElevation + raw * g.elevationSpan;
    }
    // Relative mode: scene units scaled by heightScale
    return raw * 100.0 * (g.heightScale ?? 1.0);
  }, [terrainRef, isAbsolute]);

  useEffect(() => {
    if (!enabled) {
      setProbeData(null);
      setMousePos(p => ({ ...p, visible: false }));
      return;
    }

    const canvas = terrainRef.current?.getCanvas?.()?.current;
    if (!canvas) return;

    function onMouseMove(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
        pendingMouse.current = null;
        return;
      }

      // Normalised coordinates within canvas [-1, 1]
      const ndcX = (x / rect.width) * 2 - 1;
      const ndcY = -(y / rect.height) * 2 + 1;

      pendingMouse.current = { clientX: x, clientY: y, ndcX, ndcY, rect };

      if (!rafRef.current) {
        rafRef.current = requestAnimationFrame(processProbe);
      }
    }

    function onMouseLeave() {
      pendingMouse.current = null;
      setMousePos(p => ({ ...p, visible: false }));
      setProbeData(null);
      onProbe?.(null);
    }

    function processProbe() {
      rafRef.current = null;
      if (!pendingMouse.current) {
        setMousePos(p => ({ ...p, visible: false }));
        setProbeData(null);
        return;
      }

      const { clientX, clientY, ndcX, ndcY } = pendingMouse.current;

      const g = terrainRef.current?.getRef?.()?.current;
      if (!g?.camera || !g.mesh) {
        setMousePos({ x: clientX, y: clientY, visible: true });
        return;
      }

      // Planar raycast approximation or top-down mapping
      // For top-view / orbit: project ray to ground plane [-1, 1]
      let nx = (ndcX + 1) / 2;
      let nz = (1 - ndcY) / 2;

      // In perspective/orbit: compute hit on the ground plane Y=0 or camera look direction
      const cam = g.camera;
      if (cam) {
        // Simple perspective ground raycast:
        // Ray from cam.position along unprojected ray
        const rayDirX = ndcX;
        const rayDirY = ndcY;
        // World position approximation from camera target
        const orb = g.orbit;
        if (orb?.target) {
          const targetX = orb.target.x ?? 0;
          const targetZ = orb.target.z ?? 0;
          // Offset by NDC relative to orbit target
          const dist = cam.position.distance ? cam.position.distance(orb.target) : 2.5;
          const wx = targetX + ndcX * dist * 0.45;
          const wz = targetZ - ndcY * dist * 0.45;
          nx = Math.max(0, Math.min(1, (wx + 1) / 2));
          nz = Math.max(0, Math.min(1, (wz + 1) / 2));
        }
      }

      const elev = sampleElevationAt(nx, nz);

      if (elev !== null) {
        const data = {
          x: (nx * 2 - 1).toFixed(3),
          z: (nz * 2 - 1).toFixed(3),
          elevation: elev,
        };
        setProbeData(data);
        onProbe?.({ x: nx * 2 - 1, z: nz * 2 - 1, elevation: elev });
      }

      setMousePos({ x: clientX, y: clientY, visible: true });
    }

    canvas.addEventListener('mousemove', onMouseMove, { passive: true });
    canvas.addEventListener('mouseleave', onMouseLeave);

    return () => {
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mouseleave', onMouseLeave);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [enabled, terrainRef, sampleElevationAt, onProbe]);

  if (!enabled || !mousePos.visible || !probeData) return null;

  return (
    <div
      aria-live="off"
      style={{
        position: 'absolute',
        left: mousePos.x + 16,
        top: mousePos.y + 16,
        pointerEvents: 'none',
        zIndex: 20,
        background: 'rgba(13,17,23,0.92)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        padding: '6px 10px',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        minWidth: 120,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <Crosshair size={11} strokeWidth={1.5} color="var(--dw-probe)" aria-hidden="true" />
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 9,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-ghost)',
        }}>
          CURSOR LOCATION
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 11,
          color: 'var(--dw-fg-muted)',
        }}>
          {isAbsolute ? 'Elevation' : 'Rel. Elevation'}:
        </span>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 13,
          fontWeight: 500,
          color: 'var(--dw-fg)',
        }}>
          {probeData.elevation >= 0 && !isAbsolute ? `+${probeData.elevation.toFixed(1)}` : probeData.elevation.toFixed(1)}
        </span>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 10,
          color: 'var(--dw-fg-ghost)',
        }}>
          {unitLabel}
        </span>
      </div>

      <div style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 9,
        color: 'var(--dw-fg-ghost)',
      }}>
        X: {probeData.x} / Z: {probeData.z}
      </div>
    </div>
  );
}

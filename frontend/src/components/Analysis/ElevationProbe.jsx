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
 *   - Background: --dw-panel (rgba(16,16,18,0.92)) with 1px --dw-rim
 *   - No box shadows
 *   - Instrument-needle feel
 *
 * Spec §14.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { useApp } from '../../store/appStore.jsx';
import { getElevation } from '../../api/terrain.js';
import { Crosshair } from 'lucide-react';
import { sampleSemanticAtUV } from '../../lib/semanticSampler.js';

/**
 * @param {{
 *   terrainRef: React.RefObject,
 *   enabled?: boolean,
 *   onProbe?: (point: { x: number, z: number, elevation: number } | null) => void,
 *   semanticData?: object | null,
 * }} props
 */
export default function ElevationProbe({ terrainRef, enabled = true, onProbe, semanticData = null }) {
  const { state } = useApp();
  const [probeData, setProbeData] = useState(null);
  const [mousePos, setMousePos] = useState({ x: -100, y: -100, visible: false });
  const rafRef = useRef(null);
  const pendingMouse = useRef(null);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel = 'm'; // world scale is metres (1 m/pixel documented fallback)
  const sceneId = state.scene?.scene_id;
  const exactTimer = useRef(null);

  // Debounced metered sample: replaces the instant heightmap estimate with
  // the exact float32 DSM value + accuracy statement once the backend
  // answers (the estimate shows immediately so the card never lags).
  const fetchExactElevation = useCallback((nx, nz) => {
    if (!sceneId) return;
    const g = terrainRef.current?.getRef?.()?.current;
    if (!g?.hmWidth || !g?.hmHeight) return;
    const px = Math.min(g.hmWidth - 1, Math.max(0, Math.round(nx * (g.hmWidth - 1))));
    const py = Math.min(g.hmHeight - 1, Math.max(0, Math.round(nz * (g.hmHeight - 1))));
    if (exactTimer.current) clearTimeout(exactTimer.current);
    exactTimer.current = setTimeout(async () => {
      try {
        const r = await getElevation(sceneId, px, py);
        if (!r?.metered) return;
        setProbeData(curr => curr ? {
          ...curr,
          elevation: r.elevation ?? curr.elevation,
          metered: true,
          precision_m: r.precision_m ?? null,
          confidence: r.confidence ?? null,
        } : curr);
      } catch {
        /* keep the heightmap estimate */
      }
    }, 160);
  }, [sceneId, terrainRef]);

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
      terrainRef.current?.setSemanticHighlightClass?.(-1);
      onProbe?.(null);
    }

    function processProbe() {
      rafRef.current = null;
      if (!pendingMouse.current) {
        setMousePos(p => ({ ...p, visible: false }));
        setProbeData(null);
        terrainRef.current?.setSemanticHighlightClass?.(-1);
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
        const orb = g.orbit;
        if (orb?.target) {
          const targetX = orb.target.x ?? 0;
          const targetZ = orb.target.z ?? 0;
          const dist = cam.position.distance ? cam.position.distance(orb.target) : 2.5;
          const wx = targetX + ndcX * dist * 0.45;
          const wz = targetZ - ndcY * dist * 0.45;
          const w = typeof g.worldWidth === 'number' && g.worldWidth > 0 ? g.worldWidth : 2;
          const d = typeof g.worldDepth === 'number' && g.worldDepth > 0 ? g.worldDepth : 2;
          nx = Math.max(0, Math.min(1, wx / w + 0.5));
          nz = Math.max(0, Math.min(1, wz / d + 0.5));
        }
      }

      const elev = sampleElevationAt(nx, nz);

      let semInfo = null;
      if (semanticData?.available) {
        semInfo = sampleSemanticAtUV(semanticData, nx, nz);
        if (semInfo) {
          terrainRef.current?.setSemanticHighlightClass?.(semInfo.classId);
        }
      }

      if (elev !== null) {
        const data = {
          x: (nx * 2 - 1).toFixed(3),
          z: (nz * 2 - 1).toFixed(3),
          elevation: elev,
          metered: false,
          semantic: semInfo,
        };
        setProbeData(data);
        onProbe?.({ x: nx * 2 - 1, z: nz * 2 - 1, elevation: elev });
        fetchExactElevation(nx, nz);
      }

      setMousePos({ x: clientX, y: clientY, visible: true });
    }

    canvas.addEventListener('mousemove', onMouseMove, { passive: true });
    canvas.addEventListener('mouseleave', onMouseLeave);

    return () => {
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mouseleave', onMouseLeave);
      terrainRef.current?.setSemanticHighlightClass?.(-1);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (exactTimer.current) clearTimeout(exactTimer.current);
    };
  }, [enabled, terrainRef, sampleElevationAt, onProbe, fetchExactElevation, semanticData]);

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
        background: 'rgba(16,16,18,0.92)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        padding: '8px 12px',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        minWidth: 140,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <Crosshair size={13} strokeWidth={1.5} color="var(--dw-probe)" aria-hidden="true" />
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 10.5,
          fontWeight: 600,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-ghost)',
        }}>
          CURSOR LOCATION
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13,
          color: 'var(--dw-fg-muted)',
        }}>
          {probeData.metered ? 'Metered Elevation' : (isAbsolute ? 'Elevation' : 'Rel. Elevation')}:
        </span>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 15,
          fontWeight: 600,
          color: 'var(--dw-fg)',
        }}>
          {probeData.elevation >= 0 && !isAbsolute ? `+${probeData.elevation.toFixed(1)}` : probeData.elevation.toFixed(1)}
        </span>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 12,
          color: 'var(--dw-fg-ghost)',
        }}>
          {unitLabel}
        </span>
      </div>

      {probeData.semantic && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          marginTop: 2,
          paddingTop: 4,
          borderTop: '1px solid rgba(255,255,255,0.08)',
        }}>
          <span style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: probeData.semantic.colorHex,
          }} />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11.5,
            fontWeight: 600,
            color: '#f8fafc',
            textTransform: 'capitalize',
          }}>
            {probeData.semantic.className}
          </span>
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 10.5,
            color: 'var(--dw-fg-ghost)',
          }}>
            ({Math.round(probeData.semantic.confidence * 100)}%)
          </span>
        </div>
      )}

      {probeData.metered && probeData.confidence && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          maxWidth: 220,
        }}>
          <div style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 11,
            color: probeData.confidence.level === 'high' ? '#4ade80'
              : probeData.confidence.level === 'medium' ? '#facc15' : 'var(--dw-fg-ghost)',
          }}>
            {probeData.confidence.percent != null
              ? `${probeData.confidence.percent}% height confidence`
              : `${probeData.confidence.level} height confidence`}
            {probeData.precision_m != null && ` · ± ${probeData.precision_m.toFixed(4)} ${unitLabel}`}
          </div>
          <div style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 10.5,
            lineHeight: 1.35,
            color: 'var(--dw-fg-ghost)',
          }}>
            {probeData.confidence.basis}
          </div>
        </div>
      )}

      <div style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 11.5,
        color: 'var(--dw-fg-ghost)',
      }}>
        X: {probeData.x} / Z: {probeData.z}
      </div>
    </div>
  );
}

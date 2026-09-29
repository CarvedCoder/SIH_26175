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
 *   - Background: --dw-panel (var(--dw-glass)) with 1px --dw-rim
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

      // Absolute client coords — getTerrainPointFromEvent expects them
      pendingMouse.current = { clientX: e.clientX, clientY: e.clientY };

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

      const { clientX, clientY } = pendingMouse.current;

      // REAL raycast against the active terrain chunks — the planar
      // approximation this used to apply mapped the cursor to the wrong
      // ground cell at every non-top-down angle (readouts of ±80 m on an
      // 18 m scene came from here). getTerrainPointFromEvent returns true
      // meters sampled from the terrain dataset at the hit point.
      const hit = terrainRef.current?.getTerrainPointFromEvent?.({
        clientX,
        clientY,
      });
      if (!hit) {
        setMousePos({ x: clientX, y: clientY, visible: true });
        return;
      }

      let semInfo = null;
      if (semanticData?.available && typeof hit.u === 'number') {
        semInfo = sampleSemanticAtUV(semanticData, hit.u, hit.v);
        if (semInfo) {
          terrainRef.current?.setSemanticHighlightClass?.(semInfo.classId);
        }
      }

      const data = {
        x: hit.x.toFixed(1),
        z: hit.z.toFixed(1),
        elevation: hit.elevation,
        metered: false,
        semantic: semInfo,
      };
      setProbeData(data);
      onProbe?.({ x: hit.u, z: hit.v, elevation: hit.elevation });
      fetchExactElevation(hit.u, hit.v);

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
  }, [enabled, terrainRef, onProbe, fetchExactElevation, semanticData]);

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
        background: 'var(--dw-glass)',
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

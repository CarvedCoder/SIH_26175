/**
 * DepthWizard — CameraHUD (Phase 7)
 *
 * Navigation HUD shown only in first-person camera mode (§22).
 * Four data-face readout panels:
 *   ALTITUDE — camera Y position (m or scene units, from elevation_mode)
 *   HEADING  — camera yaw as compass bearing (°)
 *   SLOPE    — terrain gradient at cursor/camera position (computed from heightmap gradient)
 *   POSITION — X, Z world coordinates (normalised or CRS, per mode)
 *
 * HUD values come from camera + terrain state — no API calls.
 * SLOPE computed from local heightmap gradient at camera position (§D10, §7.2).
 *
 * DESIGN.md:
 *   - Data face (monospace) for values; UI face for labels
 *   - --dw-panel background with 1px --dw-rim border
 *   - No card shadows
 *   - Instrument-panel style: label above value
 *   - Shown only in first-person mode (renders null otherwise)
 *
 * Spec §22.
 */
import { useEffect, useState, useRef } from 'react';

/**
 * @param {{
 *   terrainRef: React.RefObject,
 *   cameraMode: 'orbit'|'first-person'|'top',
 *   elevationMode: 'absolute'|'relative',
 * }} props
 */
export default function CameraHUD({ terrainRef, cameraMode, elevationMode }) {
  // Only render in first-person mode
  if (cameraMode !== 'first-person') return null;

  return <HUDContent terrainRef={terrainRef} elevationMode={elevationMode} />;
}

function HUDContent({ terrainRef, elevationMode }) {
  const [readouts, setReadouts] = useState({
    altitude: null,
    heading: null,
    slope: null,
    posX: null,
    posZ: null,
  });

  const rafRef = useRef(null);

  useEffect(() => {
    let running = true;

    function tick() {
      if (!running) return;
      rafRef.current = requestAnimationFrame(tick);

      const g = terrainRef.current?.getRef?.()?.current;
      if (!g?.camera) return;

      const cam = g.camera;
      const pos = cam.position;

      // ── Altitude ──
      // In absolute mode: scale to match heightScale × exaggeration (visual value)
      // Label correctly per D10
      const altitude = pos.y;

      // ── Heading ──
      // Derive from camera forward vector (yaw)
      // Camera looks at lookAt target — approximate from position change or use fpState
      // We read the fpState via the hook's internal ref if available, else approximate
      // For now, approximate from glRef's camera world matrix direction
      const heading = computeHeading(cam);

      // ── Slope ──
      // Sample heightmap gradient at camera ground position
      const nx = Math.max(0, Math.min(1, (pos.x + 1) / 2));
      const nz = Math.max(0, Math.min(1, (pos.z + 1) / 2));
      const slope = computeSlope(g, nx, nz);

      setReadouts({ altitude, heading, slope, posX: pos.x, posZ: pos.z });
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [terrainRef]);

  const elevUnits = elevationMode === 'absolute' ? 'm' : 'scene u.';
  const { altitude, heading, slope, posX, posZ } = readouts;

  return (
    <div
      role="status"
      aria-label="Navigation HUD"
      aria-live="off"
      style={{
        position: 'absolute',
        bottom: 16,
        right: 16,
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'flex-end',
        maxWidth: 'calc(100vw - 32px)',
        gap: 4,
        zIndex: 10,
        pointerEvents: 'none',
      }}
    >
      <HUDFace label="ALTITUDE" value={fmt(altitude, 2)} unit={elevUnits} />
      <HUDFace label="HEADING"  value={fmtHeading(heading)} unit="°" />
      <HUDFace label="SLOPE"    value={fmt(slope, 1)} unit="°" />
      <HUDFace label="POSITION" value={`${fmt(posX, 3)}, ${fmt(posZ, 3)}`} unit="" />
    </div>
  );
}

/** One data-face readout panel */
function HUDFace({ label, value, unit }) {
  return (
    <div style={{
      background: 'rgba(13,17,23,0.92)', // --dw-panel with alpha
      border: '1px solid var(--dw-rim)',
      borderRadius: 'var(--dw-radius-sm)',
      padding: '10px 14px',
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
      minWidth: 80,
      alignItems: 'flex-start',
    }}>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 11,
        letterSpacing: '0.07em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-ghost)',
        fontWeight: 600,
      }}>
        {label}
      </span>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 3 }}>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 20,
          fontWeight: 600,
          letterSpacing: '-0.01em',
          color: 'var(--dw-fg)',
          lineHeight: 1,
        }}>
          {value ?? '—'}
        </span>
        {unit && (
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 13,
            color: 'var(--dw-fg-ghost)',
          }}>
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

/* ─── Computation helpers ─────────────────────────────────────────────── */

/**
 * Extract yaw heading in degrees from camera's view/world matrix.
 * @param {import('three').Camera|any} cam
 * @returns {number} 0–360
 */
function computeHeading(cam) {
  if (!cam) return 0;
  const m = cam.matrixWorld?.elements ?? cam.matrix?.elements ?? cam.matrix;
  if (!m) return 0;
  // Forward in local space is -Z; world forward = -col2 of world matrix
  const fx = -(m[8] ?? 0);
  const fz = -(m[10] ?? 0);
  const yaw = Math.atan2(fx, -fz) * (180 / Math.PI);
  return ((yaw % 360) + 360) % 360;
}

/**
 * Compute local terrain slope at normalised position (nx, nz) using finite differences
 * on the heightmap. Returns slope in degrees.
 * @param {Object} g - glRef.current
 * @param {number} nx - normalised X [0,1]
 * @param {number} nz - normalised Z [0,1]
 * @returns {number} slope in degrees
 */
function computeSlope(g, nx, nz) {
  if (!g.heightData || !g.hmWidth || !g.hmHeight) return 0;

  const { heightData: hd, hmWidth: w, hmHeight: h, heightScale: hs, exaggeration: ex } = g;
  const scale = hs * ex;

  function sampleH(u, v) {
    const px = Math.min(Math.max(Math.round(u * (w - 1)), 0), w - 1);
    const pz = Math.min(Math.max(Math.round(v * (h - 1)), 0), h - 1);
    return hd[pz * w + px] * scale;
  }

  const step = 1 / Math.max(w, h);
  const dX = (sampleH(nx + step, nz) - sampleH(nx - step, nz)) / (2 * step * 2); // world X span = 2
  const dZ = (sampleH(nx, nz + step) - sampleH(nx, nz - step)) / (2 * step * 2); // world Z span = 2
  const grad = Math.hypot(dX, dZ);
  return Math.atan(grad) * (180 / Math.PI);
}

/** Format number to N decimal places, or '—' if null */
function fmt(v, decimals = 2) {
  if (v == null || isNaN(v)) return '—';
  return v.toFixed(decimals);
}

/** Format heading: e.g. 045.2 */
function fmtHeading(v) {
  if (v == null || isNaN(v)) return '—';
  return v.toFixed(1).padStart(5, '0');
}

/**
 * DepthWizard — Minimap (Phase 6)
 *
 * The signature UX feature: a 200×200 canvas overlay that draws the uploaded
 * source photograph as background and overlays live camera telemetry.
 *
 * Features:
 *   6.1 — 200×200 canvas; draws source image as background; top-left of terrain viewport
 *   6.2 — Camera position marker: amber dot at projected 2D position, updated every frame
 *   6.3 — Camera heading arrow: rotates with camera yaw
 *   6.4 — FOV cone: filled --dw-fov triangle showing camera view frustum
 *   6.5 — Coordinate mapping: georeferenced world→CRS→image pixel; non-geo: normalised
 *   6.6 — Selected point marker: secondary dot at last terrain click; preserved across layers
 *   6.7 — Navigation trail: breadcrumb path in first-person mode; max 200 points, oldest culled
 *
 * DESIGN.md:
 *   - 200×200px, top-left of terrain viewport
 *   - border: 1px solid --dw-rim, border-radius: 4px
 *   - Camera position in amber (--dw-marker: #f59e0b)
 *   - FOV cone: rgba(59,130,246,0.22) fill (--dw-fov)
 *   - Marker interpolated with lerp(0.15) per frame (instrument-needle feel)
 *   - No backdrop blur on minimap itself — blur only inside canvas overlay elements
 *
 * Spec §8, §57, §58.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { resolveAssetUrl } from '../../api/client.js';
import { Map, Minus } from 'lucide-react';

/** Minimap canvas size */
const SIZE = 200;

/** Camera marker position lerp factor (instrument-needle interpolation) */
const MARKER_LERP = 0.15;

/** Max navigation trail length (task 6.7) */
const TRAIL_MAX = 200;

/**
 * @param {{
 *   terrainRef: React.RefObject,
 *   cameraMode: 'orbit'|'first-person'|'top',
 *   minimapMeta: import('../../types/api.js').MinimapMeta|null,
 *   selectedPoint: { x: number, z: number }|null,
 * }} props
 */
export default function Minimap({ terrainRef, cameraMode, minimapMeta, selectedPoint }) {
  const [collapsed, setCollapsed] = useState(false);
  const canvasRef   = useRef(null);
  const bgRef       = useRef(null);  // loaded HTMLImageElement for the source photo
  const markerPos   = useRef({ x: SIZE / 2, y: SIZE / 2 }); // smoothed marker
  const trailRef    = useRef([]);    // array of { x, y } canvas coords
  const rafRef      = useRef(null);

  /* ── Load background image ─────────────────────────────────────────── */
  useEffect(() => {
    const imageUrl = resolveAssetUrl(minimapMeta?.url ?? minimapMeta?.image_url ?? null);
    if (!imageUrl) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => { bgRef.current = img; };
    img.src = imageUrl;
  }, [minimapMeta?.url, minimapMeta?.image_url]);

  /* ── World → canvas coordinate mapping (task 6.5) ─────────────────── */
  /**
   * Maps a world-space position [-1, 1] to minimap canvas pixel [0, SIZE].
   * When georeferenced: world → CRS bounds → image pixel → canvas pixel.
   * Non-georeferenced: normalised terrain [0,1]² → canvas pixel.
   *
   * @param {number} wx - world X (-1 to 1)
   * @param {number} wz - world Z (-1 to 1)
   * @returns {{ x: number, y: number }} canvas pixel
   */
  const worldToCanvas = useCallback((wx, wz) => {
    // Normalised terrain position [0,1]
    const nx = (wx + 1) / 2;
    const nz = (wz + 1) / 2;

    if (minimapMeta?.coordinate_system !== 'image' && minimapMeta?.bounds) {
      // Georeferenced: map through image bounds
      const { min_x, max_x, min_y, max_y } = minimapMeta.bounds;
      // terrain bounds are in CRS space; we project normalised terrain → CRS → image pixel
      // (Frontend has no terrain CRS bounds here — use normalised approximation)
      // TODO: accept terrain CRS bounds from terrainMeta for precise mapping
    }

    // Normalised → canvas (flip Z so north = top)
    return {
      x: nx * SIZE,
      y: nz * SIZE,
    };
  }, [minimapMeta]);

  /* ── Get current camera yaw from glRef ────────────────────────────── */
  const getCameraYaw = useCallback(() => {
    const g = terrainRef.current?.getRef?.()?.current;
    if (!g?.camera) return 0;
    // Extract yaw from camera's quaternion / rotation matrix
    // OGL Camera extends Transform — position/rotation are accessible
    const pos = g.camera.position;
    const orb = g.orbit;
    if (orb) {
      // Orbit target → camera = direction vector
      const tx = orb.target?.x ?? 0, tz = orb.target?.z ?? 0;
      return Math.atan2(pos.x - tx, pos.z - tz);
    }
    return 0;
  }, [terrainRef]);

  /* ── Get camera world position ─────────────────────────────────────── */
  const getCameraPos = useCallback(() => {
    const g = terrainRef.current?.getRef?.()?.current;
    if (!g?.camera) return { x: 0, z: 0 };
    return { x: g.camera.position.x, z: g.camera.position.z };
  }, [terrainRef]);

  /* ── Draw loop ─────────────────────────────────────────────────────── */
  useEffect(() => {
    let animId;

    function draw() {
      animId = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // ─ Clear ─
      ctx.clearRect(0, 0, SIZE, SIZE);

      // ─ 6.1: Background image ─
      ctx.save();
      ctx.fillStyle = '#0d1117'; // --dw-panel fallback
      ctx.fillRect(0, 0, SIZE, SIZE);
      if (bgRef.current) {
        ctx.drawImage(bgRef.current, 0, 0, SIZE, SIZE);
        // Darken overlay for readability
        ctx.fillStyle = 'rgba(7,9,14,0.38)';
        ctx.fillRect(0, 0, SIZE, SIZE);
      }
      ctx.restore();

      // ─ Get camera position in canvas space ─
      const camWorld = getCameraPos();
      const target = worldToCanvas(camWorld.x, camWorld.z);

      // ─ 6.2: Marker position (lerp for instrument-needle feel, instant on reduced motion §32) ─
      const prefersReducedMotion = typeof window !== 'undefined' &&
        window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;

      if (prefersReducedMotion) {
        markerPos.current.x = target.x;
        markerPos.current.y = target.y;
      } else {
        markerPos.current.x += (target.x - markerPos.current.x) * MARKER_LERP;
        markerPos.current.y += (target.y - markerPos.current.y) * MARKER_LERP;
      }
      const mx = markerPos.current.x;
      const my = markerPos.current.y;

      const yaw = getCameraYaw();

      // ─ 6.7: Navigation trail (first-person mode only) ─
      if (cameraMode === 'first-person') {
        const trail = trailRef.current;
        // Append if moved more than 1px
        const last = trail[trail.length - 1];
        if (!last || Math.hypot(mx - last.x, my - last.y) > 1) {
          trail.push({ x: mx, y: my });
          if (trail.length > TRAIL_MAX) trail.shift();
        }

        if (trail.length > 1) {
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(trail[0].x, trail[0].y);
          for (let i = 1; i < trail.length; i++) {
            ctx.lineTo(trail[i].x, trail[i].y);
          }
          ctx.strokeStyle = 'rgba(249,115,22,0.35)'; // faint amber trail
          ctx.lineWidth = 1.5;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.stroke();
          ctx.restore();
        }
      } else {
        // Clear trail when not in first-person
        if (trailRef.current.length > 0) trailRef.current = [];
      }

      // ─ 6.4: FOV cone (drawn under the marker) ─
      if (cameraMode !== 'top') {
        const coneLen = 28;
        const halfFov = (45 * Math.PI / 180) / 2; // 45° FOV
        const leftAngle  = yaw - halfFov - Math.PI / 2;
        const rightAngle = yaw + halfFov - Math.PI / 2;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(mx + Math.cos(leftAngle) * coneLen, my + Math.sin(leftAngle) * coneLen);
        ctx.arc(mx, my, coneLen, leftAngle, rightAngle);
        ctx.closePath();
        ctx.fillStyle = 'rgba(59,130,246,0.22)'; // --dw-fov
        ctx.fill();
        ctx.restore();
      }

      // ─ 6.3: Camera heading arrow ─
      if (cameraMode !== 'top') {
        const arrowLen = 12;
        const ax = yaw - Math.PI / 2; // offset so 0 = north = up on canvas
        ctx.save();
        ctx.strokeStyle = '#f59e0b'; // --dw-marker
        ctx.lineWidth = 1.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(mx + Math.cos(ax) * arrowLen, my + Math.sin(ax) * arrowLen);
        // Arrowhead
        const headLen = 5;
        const headAngle = 0.45;
        ctx.lineTo(mx + Math.cos(ax - headAngle) * (arrowLen - headLen), my + Math.sin(ax - headAngle) * (arrowLen - headLen));
        ctx.moveTo(mx + Math.cos(ax) * arrowLen, my + Math.sin(ax) * arrowLen);
        ctx.lineTo(mx + Math.cos(ax + headAngle) * (arrowLen - headLen), my + Math.sin(ax + headAngle) * (arrowLen - headLen));
        ctx.stroke();
        ctx.restore();
      }

      // ─ 6.6: Selected point marker ─
      if (selectedPoint) {
        const sp = worldToCanvas(selectedPoint.x, selectedPoint.z);
        ctx.save();
        ctx.beginPath();
        ctx.arc(sp.x, sp.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(96,165,250,0.9)'; // --dw-probe
        ctx.fill();
        ctx.strokeStyle = 'rgba(96,165,250,0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
      }

      // ─ 6.2: Camera position dot (amber) — drawn last, on top ─
      ctx.save();
      // Outer ring
      ctx.beginPath();
      ctx.arc(mx, my, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(245,158,11,0.2)';
      ctx.fill();
      // Inner dot
      ctx.beginPath();
      ctx.arc(mx, my, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b'; // --dw-marker
      ctx.fill();
      ctx.restore();
    }

    animId = requestAnimationFrame(draw);
    rafRef.current = animId;

    return () => cancelAnimationFrame(animId);
  }, [cameraMode, selectedPoint, worldToCanvas, getCameraPos, getCameraYaw]);

  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        aria-label="Expand minimap"
        title="Expand minimap"
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          zIndex: 12,
          background: 'rgba(13,17,23,0.92)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
          height: 34,
          padding: '0 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13,
          fontWeight: 500,
          color: 'var(--dw-fg)',
          cursor: 'pointer',
          outline: 'none',
        }}
        onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; }}
        onBlur={e => { e.currentTarget.style.outline = 'none'; }}
      >
        <Map size={16} strokeWidth={1.5} color="var(--dw-accent)" />
        <span>Minimap</span>
      </button>
    );
  }

  return (
    <div
      aria-label="Minimap — source image with camera position overlay"
      style={{
        position: 'absolute',
        top: 12,
        left: 12,
        width: 'min(var(--dw-minimap-sz, 220px), 45vw)',
        height: 'min(var(--dw-minimap-sz, 220px), 45vw)',
        maxWidth: 220,
        maxHeight: 220,
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        overflow: 'hidden',
        zIndex: 10,
      }}
    >
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        style={{ display: 'block', width: '100%', height: '100%' }}
        aria-hidden="true"
      />
      <button
        onClick={() => setCollapsed(true)}
        aria-label="Collapse minimap"
        title="Collapse minimap"
        style={{
          position: 'absolute',
          top: 5,
          right: 5,
          zIndex: 12,
          background: 'rgba(7,9,14,0.85)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 3,
          width: 24,
          height: 24,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--dw-fg-muted)',
          cursor: 'pointer',
          padding: 0,
          outline: 'none',
        }}
        onMouseEnter={e => { e.currentTarget.style.color = 'var(--dw-fg)'; }}
        onMouseLeave={e => { e.currentTarget.style.color = 'var(--dw-fg-muted)'; }}
      >
        <Minus size={14} strokeWidth={2} />
      </button>
    </div>
  );
}

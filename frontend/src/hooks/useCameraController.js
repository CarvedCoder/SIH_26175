/**
 * DepthWizard — useCameraController hook (React Three Fiber / Three.js)
 *
 * Manages three camera modes for the terrain viewer:
 *   5.1 — Orbit: mouse drag rotates, scroll zooms, right-drag pans
 *   5.2 — Walkthrough (internal id 'first-person'): creative-style free
 *         flight — WASD strafe/advance, Space/Ctrl altitude, Shift boost,
 *         pointer-lock mouse look. The camera is never clamped to the
 *         terrain; only a small safety floor keeps it near the scene.
 *   5.3 — Top-view: overhead view
 *
 * Returns { mode, setMode, resetCamera, attachOrbit, orbitRef, tickWalkthrough }
 * and handles all event listeners for the canvas.
 *
 * Design decisions (§D06):
 *   - All camera logic is frontend-only — no API calls per frame
 *   - Movement and mouse-look only run while the pointer is locked, so keys
 *     can never stay stuck after Escape, window blur, or a mode switch
 *   - Entry placement samples the *visual* terrain scale, never the physical
 *     heightScale (see TerrainCanvas §BASE_VISUAL_HEIGHT_SCALE)
 */
import { useState, useRef, useEffect, useCallback } from 'react';
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/** @typedef {'orbit'|'first-person'|'top'} CameraMode */

/* ─── Walkthrough tuning — all speeds in world units / second ─────────────
 * The terrain slab spans [-1, 1] on X/Z, so these feel fast on purpose:
 * normal flight crosses the scene in about a second, boost is for gaining
 * overview altitude quickly. Vertical is slightly slower for precision. */
const WALKTHROUGH_SPEED = 2.0;
const WALKTHROUGH_BOOST_SPEED = 6.0;
const VERTICAL_SPEED = 1.6;
const MOUSE_SENSITIVITY = 0.0022;

/** Pitch clamp (±) — just short of straight up/down so the view never flips. */
const PITCH_LIMIT = Math.PI / 2 - 0.08;

/** Entry placement: altitude above the terrain centre, slight south offset,
 * and a gentle downward gaze so the surrounding relief reads immediately. */
const WALKTHROUGH_START_CLEARANCE = 0.3;
const WALKTHROUGH_START_OFFSET_Z = 0.35;
const WALKTHROUGH_START_PITCH = -0.32;

/** Soft bounds — allow leaving the terrain slab briefly to look back at it,
 * without letting the camera drift far into the empty void. */
const WALKTHROUGH_HORIZONTAL_BOUND = 1.6;
const WALKTHROUGH_MIN_ALTITUDE = -0.2;

/** Keys that drive the walkthrough, mapped to intent (Ctrl OR C descend). */
const WALKTHROUGH_KEYS = {
  KeyW: 'forward',  ArrowUp: 'forward',
  KeyS: 'backward', ArrowDown: 'backward',
  KeyA: 'left',     ArrowLeft: 'left',
  KeyD: 'right',    ArrowRight: 'right',
  Space: 'up',
  ControlLeft: 'down', ControlRight: 'down', KeyC: 'down',
  ShiftLeft: 'boost',  ShiftRight: 'boost',
};

/**
 * @param {{
 *   canvasRef: React.RefObject<HTMLCanvasElement>,
 *   glRef: React.MutableRefObject<{
 *     renderer: any,
 *     camera: any,
 *     scene: any,
 *     orbit: any,
 *     heightData: Float32Array|null,
 *     hmWidth: number,
 *     hmHeight: number,
 *     visualHeightScale: number,
 *     exaggeration: number,
 *   }>,
 * }} options
 */
export function useCameraController({ canvasRef, glRef }) {
  const [mode, setModeState] = useState(/** @type {CameraMode} */ ('orbit'));
  const orbitRef = useRef(null);
  const fpState  = useRef({
    keys: new Set(),
    yaw: 0,
    pitch: WALKTHROUGH_START_PITCH,
    locked: false,
    active: false,
  });

  /* ── Resolve the actual <canvas> DOM element ─────────────────────────
   * R3F wraps the WebGL canvas inside its own container div, so
   * canvasRef.current may point to the wrapper rather than the true
   * <canvas>. This helper always returns the real canvas element. */
  const resolveCanvas = useCallback(() => {
    const ref = canvasRef?.current;
    if (!ref) return null;
    // If the ref itself is a canvas, use it directly
    if (ref instanceof HTMLCanvasElement) return ref;
    // If it's a DOM element (R3F wrapper div), find the canvas inside
    if (ref instanceof HTMLElement) return ref.querySelector('canvas') ?? ref;
    // If it's a ref object { current: ... }
    if (ref.current instanceof HTMLCanvasElement) return ref.current;
    if (ref.current instanceof HTMLElement) return ref.current.querySelector('canvas') ?? ref.current;
    return null;
  }, [canvasRef]);

  /** Sample *visual* terrain height (rendered world Y) at normalised [0,1] coords */
  const sampleVisualHeight = useCallback((nx, nz) => {
    const g = glRef.current;
    if (!g.heightData || !g.hmWidth || !g.hmHeight) return 0;
    const px = Math.min(Math.max(Math.round(nx * (g.hmWidth - 1)), 0), g.hmWidth - 1);
    const pz = Math.min(Math.max(Math.round(nz * (g.hmHeight - 1)), 0), g.hmHeight - 1);
    // Visual scale keeps the mesh proportional to its fixed [-1,1] footprint —
    // never use the physical heightScale for camera placement here.
    const visualScale = (g.visualHeightScale ?? 0.22) * (g.exaggeration ?? 1);
    return g.heightData[pz * g.hmWidth + px] * visualScale;
  }, [glRef]);

  /** Aim the walkthrough camera along its current yaw/pitch */
  const applyLook = useCallback((cam, fp) => {
    const cosPitch = Math.cos(fp.pitch);
    cam.lookAt(
      cam.position.x - Math.sin(fp.yaw) * cosPitch,
      cam.position.y + Math.sin(fp.pitch),
      cam.position.z - Math.cos(fp.yaw) * cosPitch
    );
  }, []);

  /** Tear down walkthrough interaction: keys, lock flag, browser pointer lock */
  const exitWalkthrough = useCallback(() => {
    const fp = fpState.current;
    fp.active = false;
    fp.locked = false;
    fp.keys.clear();
    if (document.pointerLockElement) {
      try { document.exitPointerLock(); } catch { /* ignore */ }
    }
  }, []);

  /** Switch camera mode */
  const setMode = useCallback((newMode) => {
    const g = glRef.current;
    if (!g.camera) return;

    // Disable orbit in non-orbit modes
    const orbit = g.orbit || orbitRef.current;
    if (orbit) {
      orbit.enabled = (newMode === 'orbit');
    }

    setModeState(newMode);

    if (newMode === 'orbit') {
      exitWalkthrough();
      g.camera.position.set(0, 1.2, 2.5);
      if (orbit) {
        orbit.target?.set(0, 0, 0);
        orbit.update?.();
      }
    }

    if (newMode === 'top') {
      exitWalkthrough();
      g.camera.position.set(0, 5, 0);
      g.camera.lookAt(0, 0, 0);
      if (orbit) {
        orbit.target?.set(0, 0, 0);
        orbit.update?.();
      }
    }

    if (newMode === 'first-person') {
      // Enter Walkthrough: near the centre, slightly above the visual terrain
      // surface with enough clearance to read the surrounding relief.
      const startH = sampleVisualHeight(0.5, 0.5) + WALKTHROUGH_START_CLEARANCE;
      const fp = fpState.current;
      fp.yaw = 0;
      fp.pitch = WALKTHROUGH_START_PITCH;
      fp.locked = false;
      fp.active = true;
      fp.keys.clear();
      g.camera.position.set(0, startH, WALKTHROUGH_START_OFFSET_Z);
      applyLook(g.camera, fp);
    }
  }, [glRef, sampleVisualHeight, applyLook, exitWalkthrough]);

  /** Reset camera to default orbit */
  const resetCamera = useCallback(() => {
    const g = glRef.current;
    if (!g.camera) return;
    exitWalkthrough();
    g.camera.position.set(0, 1.2, 2.5);
    const orbit = g.orbit || orbitRef.current;
    if (orbit) {
      orbit.target?.set(0, 0, 0);
      orbit.enabled = true;
      orbit.update?.();
    }
    setModeState('orbit');
  }, [glRef, exitWalkthrough]);

  /** Attach orbit controller to a canvas */
  const attachOrbit = useCallback((canvas, camera) => {
    if (!canvas || !camera) return null;
    const orbit = new OrbitControls(camera, canvas);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    orbit.enablePan = true;
    orbit.panSpeed = 0.5;
    orbit.minDistance = 0.3;
    orbit.maxDistance = 8;
    orbit.minPolarAngle = 0.05;
    orbit.maxPolarAngle = Math.PI * 0.48;
    orbitRef.current = orbit;
    return orbit;
  }, []);

  /** Walkthrough keyboard / mouse-look / pointer-lock handlers */
  useEffect(() => {
    const fp = fpState.current;

    function onKeyDown(e) {
      if (!fp.active || !fp.locked) return;
      if (e.code in WALKTHROUGH_KEYS) {
        fp.keys.add(e.code);
        e.preventDefault();
      }
    }
    function onKeyUp(e) {
      fp.keys.delete(e.code);
    }

    /* Escape key exits walkthrough cleanly even when pointer lock
     * is already released (browser auto-releases on Escape). */
    function onKeyDownGlobal(e) {
      if (e.code === 'Escape' && fp.active) {
        fp.locked = false;
        fp.keys.clear();
      }
    }

    function onMouseMove(e) {
      // Only process mouse look while pointer is locked — prevents
      // erratic camera rotation from normal mouse movement before
      // the user has clicked to capture the cursor.
      if (!fp.active || !fp.locked) return;
      fp.yaw   -= e.movementX * MOUSE_SENSITIVITY;
      fp.pitch -= e.movementY * MOUSE_SENSITIVITY;
      fp.pitch  = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, fp.pitch));
    }

    function onCanvasClick(e) {
      if (!fp.active || fp.locked) return;
      // Resolve the actual canvas element — R3F nests it inside a wrapper
      const canvas = resolveCanvas();
      if (!canvas) return;
      // Accept clicks on the canvas itself or any element inside
      // the terrain viewport container
      if (e.target !== canvas && !canvas.contains(e.target) && !e.target?.closest?.('[aria-label="3D terrain viewer"]')) {
        return;
      }
      const request = canvas.requestPointerLock?.();
      if (request && typeof request.catch === 'function') request.catch(() => {});
    }

    function onPointerLockChange() {
      const canvas = resolveCanvas();
      // Check if any element in our canvas container has the lock
      const lockEl = document.pointerLockElement;
      fp.locked = !!(lockEl && (lockEl === canvas || canvas?.contains(lockEl) || lockEl.closest?.('[aria-label="3D terrain viewer"]')));
      if (!fp.locked) {
        fp.keys.clear();
      }
    }
    function onPointerLockError() {
      fp.locked = false;
      fp.keys.clear();
    }
    function onWindowBlur() {
      fp.keys.clear();
    }

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keydown', onKeyDownGlobal);
    document.addEventListener('keyup', onKeyUp);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('click', onCanvasClick);
    document.addEventListener('pointerlockchange', onPointerLockChange);
    document.addEventListener('pointerlockerror', onPointerLockError);
    window.addEventListener('blur', onWindowBlur);

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keydown', onKeyDownGlobal);
      document.removeEventListener('keyup', onKeyUp);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('click', onCanvasClick);
      document.removeEventListener('pointerlockchange', onPointerLockChange);
      document.removeEventListener('pointerlockerror', onPointerLockError);
      window.removeEventListener('blur', onWindowBlur);
    };
  }, [canvasRef, resolveCanvas]);

  /**
   * Walkthrough camera tick — called every frame from the render loop in
   * TerrainCanvas (delta-time based, so speed is frame-rate independent).
   */
  const tickWalkthrough = useCallback((dt) => {
    const g  = glRef.current;
    const fp = fpState.current;
    if (!fp.active || !g.camera) return;

    const cam = g.camera;

    // Always apply look (so the camera orientation stays correct even
    // if the user just moved the mouse without pressing any movement keys)
    applyLook(cam, fp);

    // Movement only while pointer is locked
    if (!fp.locked) return;

    const keys = fp.keys;
    if (keys.size === 0) return;

    const boost = keys.has('ShiftLeft') || keys.has('ShiftRight');
    const speed = (boost ? WALKTHROUGH_BOOST_SPEED : WALKTHROUGH_SPEED) * dt;

    // Horizontal movement is yaw-relative (walking-style, independent of
    // pitch) — looking down while pressing W overflies, it does not dive.
    const sinYaw = Math.sin(fp.yaw);
    const cosYaw = Math.cos(fp.yaw);
    const fwdX = -sinYaw;
    const fwdZ = -cosYaw;

    let dx = 0, dz = 0;
    if (keys.has('KeyW') || keys.has('ArrowUp'))    { dx += fwdX; dz += fwdZ; }
    if (keys.has('KeyS') || keys.has('ArrowDown'))  { dx -= fwdX; dz -= fwdZ; }
    if (keys.has('KeyA') || keys.has('ArrowLeft'))  { dx -= cosYaw; dz += sinYaw; }
    if (keys.has('KeyD') || keys.has('ArrowRight')) { dx += cosYaw; dz -= sinYaw; }

    // Normalise so diagonal flight isn't faster than cardinal flight
    const hLen = Math.hypot(dx, dz);
    if (hLen > 0) { dx /= hLen; dz /= hLen; }

    let dy = 0;
    if (keys.has('Space')) dy += 1;
    if (keys.has('ControlLeft') || keys.has('ControlRight') || keys.has('KeyC')) dy -= 1;

    // Free flight: no terrain clamping — only the soft bounds below.
    const pos = cam.position;
    pos.x = Math.max(-WALKTHROUGH_HORIZONTAL_BOUND,
              Math.min(WALKTHROUGH_HORIZONTAL_BOUND, pos.x + dx * speed));
    pos.z = Math.max(-WALKTHROUGH_HORIZONTAL_BOUND,
              Math.min(WALKTHROUGH_HORIZONTAL_BOUND, pos.z + dz * speed));
    pos.y = Math.max(WALKTHROUGH_MIN_ALTITUDE, pos.y + dy * VERTICAL_SPEED * dt);

    applyLook(cam, fp);
  }, [glRef, applyLook]);

  return {
    mode,
    setMode,
    resetCamera,
    attachOrbit,
    orbitRef,
    tickWalkthrough,
  };
}

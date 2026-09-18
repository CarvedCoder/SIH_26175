/**
 * DepthWizard — useCameraController hook (React Three Fiber / Three.js)
 *
 * Manages three camera modes for the terrain viewer:
 *   5.1 — Orbit: mouse drag rotates, scroll zooms, right-drag pans
 *   5.2 — Walkthrough (internal id 'first-person'): human-scale free flight
 *         over the PHYSICAL terrain footprint — WASD strafe/advance at
 *         metres/second, Space/Ctrl altitude, Shift boost, pointer-lock
 *         mouse look, plus a touch joystick (no pointer lock needed). Entry
 *         stands at a 1.7 m eye height above the terrain surface. The
 *         camera is never clamped to the terrain; only soft, scene-scaled
 *         bounds keep it near the slab.
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

/* ─── Walkthrough tuning — all speeds in METRES / second ──────────────────
 * The terrain plane is sized to its real-world footprint (world_width_m ×
 * world_depth_m from the backend; see TerrainCanvas), so speeds are physical:
 * 5 m/s is a brisk inspection pace, boost ~4x for crossing large scenes.
 * Vertical is slightly slower for precision. */
const WALKTHROUGH_SPEED = 8.0;
const WALKTHROUGH_BOOST_SPEED = 35.0;
const VERTICAL_SPEED = 6.0;
const MOUSE_SENSITIVITY = 0.0022;

/* ─── Motion smoothing (Google-Maps-style damped movement) ─────────────────
 * The walkthrough never starts or stops at full speed: actual velocity is
 * exponentially eased toward the input's target velocity every frame, so
 * tapping a key glides in and releasing it coasts to a stop. Time constants
 * are in seconds (smaller = snappier); braking uses a longer constant than
 * accelerating so stopping has a gentle settle instead of a hard stop. */
const WALK_ACCEL_TAU  = 0.14;
const WALK_DECEL_TAU  = 0.38;
const VERT_ACCEL_TAU  = 0.18;
const VERT_DECEL_TAU  = 0.45;

/** Human eye height (metres) — the walkthrough camera stands this far above
 * the terrain surface when entering the scene. */
const EYE_HEIGHT_M = 1.7;

/** Pitch clamp (±) — just short of straight up/down so the view never flips. */
const PITCH_LIMIT = Math.PI / 2 - 0.08;

/** Entry placement: altitude = terrain surface + EYE_HEIGHT_M, a slight
 * south offset scaled to the scene size, and a gentle downward gaze so the
 * surrounding relief reads immediately. */
const WALKTHROUGH_START_PITCH = -0.25;

/** Soft bounds — the walkthrough may leave the terrain slab briefly to look
 * back at it, without drifting far into the void. Scaled to the physical
 * footprint at runtime (85% beyond the half-extent — generous enough that
 * the space feels open, not boxed-in). */
const WALKTHROUGH_BOUND_FRACTION = 0.85;
/** Lowest camera altitude (metres, relative to the terrain base plane y=0):
 * a small margin below the slab for inspecting gullies, never far under. */
const WALKTHROUGH_MIN_ALTITUDE_M = -2.0;

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

/** Scene half-diagonal helper — scales camera poses and bounds to the
 * physical footprint reported by the backend (g.worldWidth/worldDepth).
 * Falls back to the legacy 2-unit footprint for legacy responses. */
function sceneScale(g) {
  const w = typeof g?.worldWidth === 'number' && g.worldWidth > 0 ? g.worldWidth : 2;
  const d = typeof g?.worldDepth === 'number' && g.worldDepth > 0 ? g.worldDepth : 2;
  return Math.max(w, d, 2);
}

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
 *     worldWidth: number,
 *     worldDepth: number,
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
    // Joystick input (touch devices): horizontal {x: right+, y: forward+}
    // in [-1,1], vertical {-1|0|1} for descend/none/ascend. Works WITHOUT
    // pointer lock — touch users have no cursor to capture.
    joy: { x: 0, y: 0 },
    joyVertical: 0,
    // Smoothed velocity (m/s) — eased toward the input target each frame so
    // movement accelerates in and coasts to a stop (see WALK_*_TAU above).
    vel: { x: 0, y: 0, z: 0 },
  });

  /** Feed joystick horizontal movement (x: right+, y: forward+, [-1,1]). */
  const setJoystickInput = useCallback((x, y) => {
    fpState.current.joy.x = Math.max(-1, Math.min(1, Number(x) || 0));
    fpState.current.joy.y = Math.max(-1, Math.min(1, Number(y) || 0));
  }, []);

  /** Feed joystick vertical intent: +1 ascend, -1 descend, 0 none. */
  const setJoystickVertical = useCallback((v) => {
    fpState.current.joyVertical = Math.sign(Number(v) || 0);
  }, []);

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
    // Visual scale is the mesh's actual Y-units per heightmap step — the
    // real elevation span (metres) on the physical-scale path, the legacy
    // 0.22 constant on the fallback footprint. Either way this returns
    // rendered world Y at the given normalised coords.
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

  /** Tear down walkthrough interaction: keys, joystick, lock flag, browser
   * pointer lock */
  const exitWalkthrough = useCallback(() => {
    const fp = fpState.current;
    fp.active = false;
    fp.locked = false;
    fp.keys.clear();
    fp.joy.x = 0;
    fp.joy.y = 0;
    fp.joyVertical = 0;
    fp.vel.x = 0;
    fp.vel.y = 0;
    fp.vel.z = 0;
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
      const d = sceneScale(g);
      g.camera.position.set(0, d * 0.35, d * 0.7);
      if (orbit) {
        orbit.target?.set(0, 0, 0);
        orbit.update?.();
      }
    }

    if (newMode === 'top') {
      exitWalkthrough();
      g.camera.position.set(0, sceneScale(g) * 1.5, 0);
      g.camera.lookAt(0, 0, 0);
      if (orbit) {
        orbit.target?.set(0, 0, 0);
        orbit.update?.();
      }
    }

    if (newMode === 'first-person') {
      // Enter Walkthrough: a human's eye height above the terrain surface,
      // starting at the south edge of the terrain looking north so the
      // full map spreads out before the viewer — this gives a dramatic,
      // immersive sense of scale rather than dropping in at the centre.
      const d = sceneScale(g);
      const edgeNz = 0.85; // 85% down from north edge → south edge
      const startH = sampleVisualHeight(0.5, edgeNz) + EYE_HEIGHT_M;
      const fp = fpState.current;
      fp.yaw = 0;                       // face north
      fp.pitch = WALKTHROUGH_START_PITCH;
      fp.locked = false;
      fp.active = true;
      fp.keys.clear();
      fp.joy.x = 0;
      fp.joy.y = 0;
      fp.joyVertical = 0;
      fp.vel.x = 0;
      fp.vel.y = 0;
      fp.vel.z = 0;
      // Place at south edge — Z positive is south in the coordinate system.
      // (edgeNz - 0.5) * worldDepth gives world Z at 35% south of centre.
      const worldZ = (edgeNz - 0.5) * (g.worldDepth ?? d);
      g.camera.position.set(0, startH, worldZ);
      applyLook(g.camera, fp);
    }
  }, [glRef, sampleVisualHeight, applyLook, exitWalkthrough]);

  /** Reset camera to default orbit */
  const resetCamera = useCallback(() => {
    const g = glRef.current;
    if (!g.camera) return;
    exitWalkthrough();
    const d = sceneScale(g);
    g.camera.position.set(0, d * 0.35, d * 0.7);
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
    orbit.maxDistance = 5;
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
   *
   * Input sources merge: keyboard (only while the pointer is locked) and
   * the joystick vector (works without pointer lock — touch devices have
   * no cursor to capture). Speeds are METRES/second over the physical
   * terrain footprint.
   *
   * Motion is smoothed (Google-Maps-style): the input defines a *target*
   * velocity and the actual velocity is exponentially eased toward it each
   * frame — accelerating in on key press, coasting to a gentle stop on
   * release, and blending smoothly through boost on/off and analog
   * joystick deflection.
   */
  const tickWalkthrough = useCallback((dt) => {
    const g  = glRef.current;
    const fp = fpState.current;
    if (!fp.active || !g.camera) return;

    const cam = g.camera;
    const vel = fp.vel;

    // Always apply look (so the camera orientation stays correct even
    // if the user just moved the mouse without pressing any movement keys)
    applyLook(cam, fp);

    const keys = fp.keys;
    const locked = fp.locked;
    const joy = fp.joy;
    const joyActive = joy.x !== 0 || joy.y !== 0 || fp.joyVertical !== 0;

    // Keyboard movement only while locked; joystick always. With no input
    // at all, keep easing velocity toward zero so an in-progress coast
    // finishes smoothly (e.g. right after Escape releases the lock).
    const boost = locked && (keys.has('ShiftLeft') || keys.has('ShiftRight'));
    const targetSpeed = boost ? WALKTHROUGH_BOOST_SPEED : WALKTHROUGH_SPEED;

    // Horizontal movement is yaw-relative (walking-style, independent of
    // pitch) — looking down while pressing W overflies, it does not dive.
    const sinYaw = Math.sin(fp.yaw);
    const cosYaw = Math.cos(fp.yaw);
    const fwdX = -sinYaw;
    const fwdZ = -cosYaw;

    let ix = 0, iz = 0;
    if (locked) {
      if (keys.has('KeyW') || keys.has('ArrowUp'))    { ix += fwdX; iz += fwdZ; }
      if (keys.has('KeyS') || keys.has('ArrowDown'))  { ix -= fwdX; iz -= fwdZ; }
      if (keys.has('KeyA') || keys.has('ArrowLeft'))  { ix -= cosYaw; iz += sinYaw; }
      if (keys.has('KeyD') || keys.has('ArrowRight')) { ix += cosYaw; iz -= sinYaw; }
    }
    // Joystick: y+ = forward, x+ = strafe right (same intents as keys)
    if (joy.y !== 0) { ix += fwdX * joy.y; iz += fwdZ * joy.y; }
    if (joy.x !== 0) { ix += cosYaw * joy.x; iz -= sinYaw * joy.x; }

    // Normalise so diagonal movement isn't faster than cardinal movement
    const hLen = Math.hypot(ix, iz);
    if (hLen > 1) { ix /= hLen; iz /= hLen; }

    let iy = 0;
    if (locked) {
      if (keys.has('Space')) iy += 1;
      if (keys.has('ControlLeft') || keys.has('ControlRight') || keys.has('KeyC')) iy -= 1;
    }
    if (fp.joyVertical !== 0) iy += fp.joyVertical;

    // Target velocity in world space (m/s)
    const tx = ix * targetSpeed;
    const tz = iz * targetSpeed;
    const ty = iy * VERTICAL_SPEED;

    // Ease actual velocity toward the target. Separate time constants for
    // speeding up vs slowing down give a soft, settle-to-stop coast; the
    // exp form is frame-rate independent for any dt.
    const hMovingIn = Math.hypot(tx, tz) > Math.hypot(vel.x, vel.z);
    const hTau = hMovingIn ? WALK_ACCEL_TAU : WALK_DECEL_TAU;
    const vMovingIn = Math.abs(ty) > Math.abs(vel.y);
    const vTau = vMovingIn ? VERT_ACCEL_TAU : VERT_DECEL_TAU;
    const hk = 1 - Math.exp(-dt / hTau);
    const vk = 1 - Math.exp(-dt / vTau);
    vel.x += (tx - vel.x) * hk;
    vel.z += (tz - vel.z) * hk;
    vel.y += (ty - vel.y) * vk;

    // Integrate. When fully at rest (input gone and velocity decayed below
    // perception threshold), snap to zero to avoid endless micro-drift.
    if (!locked && !joyActive && Math.hypot(vel.x, vel.y, vel.z) < 0.01) {
      vel.x = 0; vel.y = 0; vel.z = 0;
    }

    // Free flight: no terrain clamping — only the soft, physically-scaled
    // bounds below (they read g.worldWidth/worldDepth each tick, so they
    // follow the loaded scene's footprint). Clamping also kills the velocity
    // component pressing into the wall so the camera doesn't stick under
    // sustained input.
    const d = sceneScale(g);
    const bound = d * (0.5 + WALKTHROUGH_BOUND_FRACTION * 0.5);
    const pos = cam.position;
    pos.x += vel.x * dt;
    pos.z += vel.z * dt;
    pos.y += vel.y * dt;
    if (pos.x < -bound) { pos.x = -bound; vel.x = 0; }
    if (pos.x >  bound) { pos.x =  bound; vel.x = 0; }
    if (pos.z < -bound) { pos.z = -bound; vel.z = 0; }
    if (pos.z >  bound) { pos.z =  bound; vel.z = 0; }
    if (pos.y < WALKTHROUGH_MIN_ALTITUDE_M) {
      pos.y = WALKTHROUGH_MIN_ALTITUDE_M;
      vel.y = 0;
    }

    applyLook(cam, fp);
  }, [glRef, applyLook]);

  return {
    mode,
    setMode,
    resetCamera,
    attachOrbit,
    orbitRef,
    tickWalkthrough,
    setJoystickInput,
    setJoystickVertical,
  };
}

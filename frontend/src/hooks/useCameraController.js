/**
 * DepthWizard — useCameraController hook (React Three Fiber / Three.js)
 *
 * Manages three camera modes for the terrain viewer:
 *   5.1 — Orbit: mouse drag rotates, scroll zooms, right-drag pans
 *   5.2 — First-person: WASD movement, mouse look, terrain-collision height clamp
 *   5.3 — Top-view: overhead view
 *
 * Returns { mode, setMode, resetCamera, attachOrbit, detachOrbit, orbitRef, tickFirstPerson }
 * and handles all event listeners for the canvas.
 *
 * Design decisions (§D06):
 *   - All camera logic is frontend-only — no API calls per frame
 *   - Height clamping reads from the glRef heightData array (client-side)
 *   - First-person mouse look only activates on pointer lock
 */
import { useState, useRef, useEffect, useCallback } from 'react';
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/** @typedef {'orbit'|'first-person'|'top'} CameraMode */

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
 *     heightScale: number,
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
    pitch: -0.3,
    locked: false,
    active: false,
  });
  const rafFp = useRef(null);

  /** Sample terrain height at normalised [0,1] coords */
  const sampleHeight = useCallback((nx, nz) => {
    const g = glRef.current;
    if (!g.heightData || !g.hmWidth || !g.hmHeight) return 0;
    const px = Math.min(Math.max(Math.round(nx * (g.hmWidth - 1)), 0), g.hmWidth - 1);
    const pz = Math.min(Math.max(Math.round(nz * (g.hmHeight - 1)), 0), g.hmHeight - 1);
    return g.heightData[pz * g.hmWidth + px] * g.heightScale * g.exaggeration;
  }, [glRef]);

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
      g.camera.position.set(0, 1.2, 2.5);
      if (orbit) {
        orbit.target?.set(0, 0, 0);
        orbit.update?.();
      }
    }

    if (newMode === 'top') {
      g.camera.position.set(0, 5, 0);
      g.camera.lookAt(0, 0, 0);
      if (orbit) {
        orbit.target?.set(0, 0, 0);
        orbit.update?.();
      }
    }

    if (newMode === 'first-person') {
      // Start at a reasonable position above terrain centre
      const startH = sampleHeight(0.5, 0.5) + 0.08;
      g.camera.position.set(0, startH, 0);
      fpState.current.yaw   = 0;
      fpState.current.pitch = 0;
      fpState.current.active = true;
    } else {
      fpState.current.active = false;
      if (rafFp.current) cancelAnimationFrame(rafFp.current);
      // Exit pointer lock if locked
      if (document.pointerLockElement) document.exitPointerLock();
    }
  }, [glRef, sampleHeight]);

  /** Reset camera to default orbit */
  const resetCamera = useCallback(() => {
    const g = glRef.current;
    if (!g.camera) return;
    g.camera.position.set(0, 1.2, 2.5);
    const orbit = g.orbit || orbitRef.current;
    if (orbit) {
      orbit.target?.set(0, 0, 0);
      orbit.enabled = true;
      orbit.update?.();
    }
    setModeState('orbit');
    fpState.current.active = false;
  }, [glRef]);

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

  /** First-person keyboard handlers */
  useEffect(() => {
    const fp = fpState.current;

    function onKeyDown(e) {
      if (fp.active) fp.keys.add(e.code);
    }
    function onKeyUp(e) {
      fp.keys.delete(e.code);
    }
    function onMouseMove(e) {
      if (!fp.active || !fp.locked) return;
      const sensitivity = 0.002;
      fp.yaw   -= e.movementX * sensitivity;
      fp.pitch -= e.movementY * sensitivity;
      fp.pitch  = Math.max(-Math.PI / 3, Math.min(Math.PI / 4, fp.pitch));
    }
    function onPointerLock() {
      fp.locked = document.pointerLockElement === canvasRef.current;
    }

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('pointerlockchange', onPointerLock);

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('pointerlockchange', onPointerLock);
    };
  }, [canvasRef]);

  /**
   * First-person camera tick — called every frame from outside this hook
   * via the render loop in TerrainCanvas.
   */
  const tickFirstPerson = useCallback((dt) => {
    const g   = glRef.current;
    const fp  = fpState.current;
    if (!fp.active || !g.camera) return;

    const cam = g.camera;
    const speed = 0.8 * dt; // world units per second

    // Build local forward/right from yaw (ignore pitch for movement)
    const sinYaw = Math.sin(fp.yaw);
    const cosYaw = Math.cos(fp.yaw);
    const fwdX = -sinYaw;
    const fwdZ = -cosYaw;

    let dx = 0, dz = 0;
    if (fp.keys.has('KeyW') || fp.keys.has('ArrowUp'))    { dx += fwdX; dz += fwdZ; }
    if (fp.keys.has('KeyS') || fp.keys.has('ArrowDown'))  { dx -= fwdX; dz -= fwdZ; }
    if (fp.keys.has('KeyA') || fp.keys.has('ArrowLeft'))  { dx -= cosYaw; dz += sinYaw; }
    if (fp.keys.has('KeyD') || fp.keys.has('ArrowRight')) { dx += cosYaw; dz -= sinYaw; }

    const pos = cam.position;
    pos.x = Math.max(-1, Math.min(1, pos.x + dx * speed));
    pos.z = Math.max(-1, Math.min(1, pos.z + dz * speed));

    // Terrain collision — clamp Y to terrain height + eye height
    const nx = (pos.x + 1) / 2; // world [-1,1] → normalised [0,1]
    const nz = (pos.z + 1) / 2;
    const groundH = sampleHeight(nx, nz);
    const eyeH = groundH + 0.08; // 8cm above terrain
    pos.y = Math.max(eyeH, pos.y);

    // Apply yaw+pitch to camera rotation (lookAt from yaw/pitch angles)
    const sinPitch = Math.sin(fp.pitch);
    const cosPitch = Math.cos(fp.pitch);
    const targetX = pos.x + fwdX * cosPitch;
    const targetY = pos.y + sinPitch;
    const targetZ = pos.z + fwdZ * cosPitch;
    cam.lookAt(targetX, targetY, targetZ);
  }, [glRef, sampleHeight]);

  return {
    mode,
    setMode,
    resetCamera,
    attachOrbit,
    orbitRef,
    tickFirstPerson,
  };
}

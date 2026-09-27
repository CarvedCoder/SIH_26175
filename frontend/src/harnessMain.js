/**
 * Terrain alignment harness — NOT part of the app.
 * Loads the real TerrainEngine with real scene artifacts served from a
 * local CORS file server, and renders fixed camera views so drape/terrain
 * alignment can be judged visually and reported via window.__harness.
 */
import * as THREE from 'three';
import { TerrainEngine } from './engine/TerrainEngine.js';
import { decodeHeightPng16 } from './engine/streaming/heightDecode.js';

const BASE = 'http://127.0.0.1:8766';
const statusEl = document.getElementById('status');
const status = (m) => { statusEl.textContent = m; };

async function main() {
  status('fetching artifacts…');
  const [hmBuf, texBlob] = await Promise.all([
    fetch(`${BASE}/heightmap.png`).then(r => r.arrayBuffer()),
    fetch(`${BASE}/rgb_preview.png`).then(r => r.blob()),
  ]);
  const decoded = await decodeHeightPng16(hmBuf);
  if (!decoded) throw new Error('heightmap decode failed');
  status(`heightmap ${decoded.width}x${decoded.height}`);

  // Same upload path as the app's loadAuthTexture (sRGB, no GL flip)
  const bitmap = await createImageBitmap(texBlob);
  const tex = new THREE.CanvasTexture(bitmap);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = true;
  tex.flipY = false;
  tex.anisotropy = 8;
  tex.needsUpdate = true;

  const canvas = document.getElementById('c');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070b);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.5, 60000);

  const minE = 0.0, maxE = 12.4091;
  const terrainMeta = {
    crs: null, projected_crs: null, affine_transform: null,
    gsd_x: null, gsd_y: null,
    raster_width: 1024, raster_height: 1024,
    world_width_m: 1024.0, world_depth_m: 1024.0,
    is_georeferenced_scale: false,
    min_elevation: minE, max_elevation: maxE,
    local_origin: [0, 0, 0],
  };

  const engine = new TerrainEngine({
    scene, camera,
    terrainMeta,
    heightData: decoded.data,
    hmWidth: decoded.width,
    hmHeight: decoded.height,
    diffuseTexture: tex,
  });
  engine.setTexture(tex);
  engine.setExaggeration(5.0);

  window.__harness = { engine, camera, renderer, scene };
  window.THREE = THREE;

  let view = 'oblique';
  function place() {
    if (view === 'oblique') {
      // Same framing as the app's default: SE-ish orbit standoff
      camera.position.set(389, 480, 840);
      camera.lookAt(0, 6, 0);
    } else if (view === 'top') {
      camera.up.set(0, 0, -1); // screen-up = north (-Z)
      camera.position.set(0, 1500, 0);
      camera.lookAt(0, 0, 0);
    }
  }
  place();

  function tick() {
    requestAnimationFrame(tick);
    engine.update(camera);
    renderer.render(scene, camera);
  }
  tick();

  window.__setView = (v) => { view = v; place(); };
  status('ready — views: oblique | top');
}

main().catch(e => { status('ERROR ' + e.message); console.error(e); });

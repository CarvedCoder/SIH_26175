/**
 * DepthWizard Geospatial Engine — TerrainDebugHUD
 *
 * Developer & mission-control telemetry HUD overlay displaying:
 *   - Real-time FPS & frame duration
 *   - Visible & active chunk count
 *   - Active triangle count
 *   - LOD distribution across the frustum
 *   - Camera altitude (MSL and AGL above terrain)
 *   - Terrain slope under camera
 *   - Real GSD and physical footprint in meters
 *   - CRS and georeferencing confirmation
 */

import { useState, useEffect, useRef } from 'react';

/**
 * @param {{
 *   engine: import('../TerrainEngine.js').TerrainEngine|null,
 *   camera: THREE.Camera|null,
 *   visible?: boolean,
 * }} props
 */
export default function TerrainDebugHUD({ engine, camera, visible = true }) {
  const [telemetry, setTelemetry] = useState(null);
  const rafRef = useRef(null);
  const frameCount = useRef(0);
  const lastTime = useRef(null);
  const fpsRef = useRef(60);

  useEffect(() => {
    if (!visible) return;

    let isRunning = true;

    function tick() {
      if (!isRunning) return;

      if (lastTime.current === null) lastTime.current = performance.now();
      frameCount.current++;
      const now = performance.now();
      const elapsed = now - lastTime.current;

      if (elapsed >= 500) {
        fpsRef.current = Math.round((frameCount.current * 1000) / elapsed);
        frameCount.current = 0;
        lastTime.current = now;

        if (engine && camera) {
          const stats = engine.getTelemetry();
          const camPos = camera.position;
          const terrainH = engine.collision.getTerrainHeight(camPos.x, camPos.z);
          const agl = camPos.y - terrainH;
          const slope = engine.spatial.sampleSlope(camPos.x, camPos.z);

          setTelemetry({
            fps: fpsRef.current,
            camX: camPos.x,
            camY: camPos.y,
            camZ: camPos.z,
            terrainH,
            agl,
            slope,
            visibleTiles: stats.visibleTileCount,
            activeTiles: stats.activeTileCount,
            pendingTiles: stats.pendingTileCount ?? 0,
            heightStreaming: stats.heightStreaming,
            triangles: stats.triangleCount,
            lods: stats.lodDistribution,
            width_m: stats.worldWidth_m,
            depth_m: stats.worldDepth_m,
            gsdX: stats.gsd_x,
            gsdY: stats.gsd_y,
            isGeoreferenced: stats.isGeoreferenced,
            crs: stats.crs || 'Local / None',
          });
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      isRunning = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [engine, camera, visible]);

  if (!visible || !telemetry) return null;

  const lodStr = Object.entries(telemetry.lods || {})
    .map(([lvl, cnt]) => `L${lvl}:${cnt}`)
    .join('  ') || 'L0:1';

  return (
    <div
      data-testid="terrain-debug-hud"
      style={{
        position: 'absolute',
        top: 60,
        left: 16,
        zIndex: 50,
        background: 'rgba(8, 12, 20, 0.88)',
        border: '1px solid rgba(79, 140, 255, 0.3)',
        borderRadius: 8,
        padding: '10px 14px',
        color: '#d2e1ff',
        fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
        fontSize: 11,
        lineHeight: 1.5,
        pointerEvents: 'none',
        backdropFilter: 'blur(6px)',
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.5)',
        minWidth: 260,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(79,140,255,0.2)', paddingBottom: 4, marginBottom: 6 }}>
        <span style={{ fontWeight: 700, color: '#4f8cff', letterSpacing: 0.8 }}>GEOSPATIAL TERRAIN ENGINE</span>
        <span style={{ color: telemetry.fps >= 50 ? '#2ecc71' : '#f39c12', fontWeight: 700 }}>{telemetry.fps} FPS</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '90px 1fr', gap: '2px 8px' }}>
        <span style={{ color: '#8aa0c4' }}>Scale Status:</span>
        <span style={{ color: telemetry.isGeoreferenced ? '#2ecc71' : '#f1c40f', fontWeight: 600 }}>
          {telemetry.isGeoreferenced ? 'CONFIRMED METRIC' : 'ESTIMATED SCALE'}
        </span>

        <span style={{ color: '#8aa0c4' }}>CRS:</span>
        <span>{telemetry.crs}</span>

        <span style={{ color: '#8aa0c4' }}>GSD:</span>
        <span>{telemetry.gsdX ? `${telemetry.gsdX.toFixed(2)}m × ${telemetry.gsdY.toFixed(2)}m` : '1.0m (Assumed)'}</span>

        <span style={{ color: '#8aa0c4' }}>Footprint:</span>
        <span>{Math.round(telemetry.width_m)}m × {Math.round(telemetry.depth_m)}m</span>

        <span style={{ color: '#8aa0c4' }}>Triangles:</span>
        <span style={{ color: '#fff' }}>{telemetry.triangles.toLocaleString()}</span>

        <span style={{ color: '#8aa0c4' }}>Chunks:</span>
        <span>{telemetry.visibleTiles} visible ({lodStr})</span>

        {telemetry.heightStreaming != null && (
          <>
            <span style={{ color: '#8aa0c4' }}>Streaming:</span>
            <span>
              {telemetry.heightStreaming ? 'H✓' : 'H—'}
              {telemetry.pendingTiles > 0 ? ` · ${telemetry.pendingTiles} loading` : ''}
            </span>
          </>
        )}

        <span style={{ color: '#8aa0c4' }}>Camera MSL:</span>
        <span>{telemetry.camY.toFixed(1)} m</span>

        <span style={{ color: '#8aa0c4' }}>Camera AGL:</span>
        <span style={{ color: telemetry.agl < 3 ? '#e74c3c' : '#4f8cff', fontWeight: 600 }}>
          {telemetry.agl.toFixed(1)} m
        </span>

        <span style={{ color: '#8aa0c4' }}>Ground Slope:</span>
        <span>{telemetry.slope.toFixed(1)}°</span>

        <span style={{ color: '#8aa0c4' }}>Coords (m):</span>
        <span>X:{Math.round(telemetry.camX)} Z:{Math.round(telemetry.camZ)}</span>
      </div>
    </div>
  );
}

/**
 * DepthWizard Geospatial Engine — TerrainCollision
 *
 * Implements terrain-aware collision detection, ground clamping, and terrain following.
 *
 * CANONICAL BEHAVIOR:
 *   - Walking mode clamps camera eye-height to 1.7 meters above terrain surface.
 *   - Flythrough mode prevents camera from tunneling through terrain (minimum clearance).
 *   - All distances and clearances operate in physical METERS.
 */

export class TerrainCollision {
  /**
   * @param {Object} options
   * @param {import('../geo/GeoReference.js').GeoReference} options.geoRef
   * @param {import('../terrain/TerrainDataset.js').TerrainDataset} options.dataset
   * @param {number} [options.eyeHeight=1.7] - Eye height in meters
   * @param {number} [options.minClearance=1.5] - Min flythrough clearance in meters
   */
  constructor(options) {
    this.geoRef = options.geoRef;
    this.dataset = options.dataset;
    this.eyeHeight = options.eyeHeight || 1.7;
    this.minClearance = options.minClearance || 1.5;
  }

  /**
   * Sample the exact physical terrain elevation in meters at a world coordinate (x, z).
   *
   * @param {number} worldX - Local world X (meters)
   * @param {number} worldZ - Local world Z (meters)
   * @returns {number} Elevation in meters
   */
  getTerrainHeight(worldX, worldZ) {
    if (!this.geoRef || !this.dataset) return 0.0;
    const { u, v } = this.geoRef.localToUv(worldX, worldZ);
    return this.dataset.sampleElevation(u, v);
  }

  /**
   * Clamp a 3D position to remain on or above the terrain.
   *
   * @param {THREE.Vector3} position - Camera position in meters
   * @param {Object} [options]
   * @param {'walk'|'fly'|'none'} [options.mode='walk']
   * @param {number} [options.exaggeration=1.0]
   * @returns {THREE.Vector3}
   */
  clampPosition(position, options = {}) {
    const mode = options.mode || 'walk';
    const exaggeration = options.exaggeration || 1.0;

    const terrainH = this.getTerrainHeight(position.x, position.z);
    // Visual elevation adjusted for vertical exaggeration around datum
    const minElev = this.geoRef.minElevation;
    const visualH = minElev + (terrainH - minElev) * exaggeration;

    if (mode === 'walk') {
      // Clamped to terrain surface + human eye height (1.7m)
      position.y = visualH + this.eyeHeight;
    } else if (mode === 'fly') {
      // Free flight, but cannot tunnel underground
      const floor = visualH + this.minClearance;
      if (position.y < floor) {
        position.y = floor;
      }
    }

    // Soft horizontal bounds to prevent drifting infinitely into the void
    const maxExtentX = this.geoRef.worldWidth * 0.85;
    const maxExtentZ = this.geoRef.worldDepth * 0.85;

    position.x = Math.max(-maxExtentX, Math.min(maxExtentX, position.x));
    position.z = Math.max(-maxExtentZ, Math.min(maxExtentZ, position.z));

    return position;
  }
}

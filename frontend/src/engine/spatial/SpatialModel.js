/**
 * DepthWizard Geospatial Engine — SpatialModel
 *
 * Centralized spatial query interface. Powers inspection, measurements,
 * and routing queries without coupling UI components to rendering internals.
 *
 * ALL DISTANCES AND ELEVATIONS ARE EXPRESSED IN REAL-WORLD METERS.
 */


export class SpatialModel {
  /**
   * @param {Object} options
   * @param {import('../geo/GeoReference.js').GeoReference} options.geoRef
   * @param {import('../terrain/TerrainDataset.js').TerrainDataset} options.dataset
   * @param {import('../camera/TerrainCollision.js').TerrainCollision} options.collision
   */
  constructor(options) {
    this.geoRef = options.geoRef;
    this.dataset = options.dataset;
    this.collision = options.collision;
    this.semanticSampler = null; // optional external semantic sampler callback
  }

  /**
   * Set semantic sampler callback: (u, v) => { classId, className, color }
   */
  setSemanticSampler(sampler) {
    this.semanticSampler = sampler;
  }

  /**
   * Sample elevation in meters at a local world position.
   *
   * @param {number} worldX - Local world X (meters)
   * @param {number} worldZ - Local world Z (meters)
   * @returns {number} Elevation in meters
   */
  sampleElevation(worldX, worldZ) {
    const { u, v } = this.geoRef.localToUv(worldX, worldZ);
    return this.dataset.sampleElevation(u, v);
  }

  /**
   * Sample terrain slope in degrees [0, 90] at a local world position.
   *
   * @param {number} worldX - Local world X (meters)
   * @param {number} worldZ - Local world Z (meters)
   * @returns {number} Slope in degrees
   */
  sampleSlope(worldX, worldZ) {
    const { u, v } = this.geoRef.localToUv(worldX, worldZ);
    return this.dataset.sampleSlope(u, v);
  }

  /**
   * Sample the surface normal vector at a local world position.
   *
   * @param {number} worldX - Local world X (meters)
   * @param {number} worldZ - Local world Z (meters)
   * @returns {THREE.Vector3}
   */
  sampleNormal(worldX, worldZ) {
    const { u, v } = this.geoRef.localToUv(worldX, worldZ);
    return this.dataset.sampleNormal(u, v);
  }

  /**
   * Query surface elevation for ground clamping and collision.
   *
   * @param {number} worldX - Local world X (meters)
   * @param {number} worldZ - Local world Z (meters)
   * @returns {number} Elevation in meters
   */
  getTerrainHeight(worldX, worldZ) {
    return this.sampleElevation(worldX, worldZ);
  }

  /**
   * Sample semantic class information at a local world position.
   *
   * @param {number} worldX - Local world X (meters)
   * @param {number} worldZ - Local world Z (meters)
   * @returns {Object|null}
   */
  sampleSemantic(worldX, worldZ) {
    if (!this.semanticSampler) return null;
    const { u, v } = this.geoRef.localToUv(worldX, worldZ);
    return this.semanticSampler(u, v);
  }

  /**
   * Convert local world coordinates to source raster pixel coordinates.
   */
  worldToPixel(worldX, worldZ) {
    return this.geoRef.localToPixel(worldX, worldZ);
  }

  /**
   * Convert source raster pixel coordinates to local world coordinates.
   */
  pixelToWorld(px, py) {
    const pt = this.geoRef.pixelToLocal(px, py);
    pt.y = this.sampleElevation(pt.x, pt.z);
    return pt;
  }

  /**
   * Convert local world coordinates to global projected CRS coordinates.
   */
  worldToProjected(worldX, worldZ) {
    const elev = this.sampleElevation(worldX, worldZ);
    return this.geoRef.localToProjected(worldX, worldZ, elev);
  }

  /**
   * Physical 3D measurement between two world-space points.
   *
   * @param {{ x: number, z: number, elevation?: number }} ptA
   * @param {{ x: number, z: number, elevation?: number }} ptB
   * @returns {{ horizontalDistance_m: number, elevationDelta_m: number, distance3D_m: number }}
   */
  measureDistance(ptA, ptB) {
    const elevA = ptA.elevation ?? this.sampleElevation(ptA.x, ptA.z);
    const elevB = ptB.elevation ?? this.sampleElevation(ptB.x, ptB.z);

    const dx = ptB.x - ptA.x;
    const dz = ptB.z - ptA.z;
    const dy = Math.abs(elevB - elevA);

    const horizontalDistance_m = Math.hypot(dx, dz);
    const distance3D_m = Math.hypot(horizontalDistance_m, dy);

    return {
      horizontalDistance_m,
      elevationDelta_m: dy,
      distance3D_m,
    };
  }

  /**
   * Query the physical bounding box and elevation limits of the world.
   */
  getWorldBounds() {
    const halfW = this.geoRef.worldWidth * 0.5;
    const halfD = this.geoRef.worldDepth * 0.5;
    return {
      minX: -halfW,
      maxX: halfW,
      minZ: -halfD,
      maxZ: halfD,
      width_m: this.geoRef.worldWidth,
      depth_m: this.geoRef.worldDepth,
      minElevation_m: this.geoRef.minElevation,
      maxElevation_m: this.geoRef.maxElevation,
      elevationSpan_m: this.geoRef.elevationSpan,
    };
  }
}

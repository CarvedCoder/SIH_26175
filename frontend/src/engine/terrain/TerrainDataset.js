/**
 * DepthWizard Geospatial Engine — TerrainDataset
 *
 * Encapsulates the underlying numerical elevation surface and texture assets.
 * Provides high-precision bilinear elevation sampling, slope calculation in degrees,
 * and metric normal evaluation.
 */

import * as THREE from 'three';

export class TerrainDataset {
  /**
   * @param {Object} options
   * @param {Float32Array} options.heightData - Normalized [0, 1] elevation array or metric floats
   * @param {number} options.width - Heightmap resolution width
   * @param {number} options.height - Heightmap resolution height
   * @param {number} options.minElevation - Minimum elevation in meters
   * @param {number} options.maxElevation - Maximum elevation in meters
   * @param {boolean} [options.isNormalized=true] - True if heightData is [0, 1] normalized
   * @param {import('../geo/GeoReference.js').GeoReference} options.geoRef
   */
  constructor(options) {
    this.heightData = options.heightData;
    this.width = options.width;
    this.height = options.height;
    this.minElevation = options.minElevation ?? 0.0;
    this.maxElevation = options.maxElevation ?? 100.0;
    this.elevationSpan = Math.max(0.001, this.maxElevation - this.minElevation);
    this.isNormalized = options.isNormalized !== false;
    this.geoRef = options.geoRef;

    // Cache texture / asset references
    this.baseTexture = null;
    this.semanticTexture = null;
    this.colormapMode = 0;
  }

  /**
   * Sample elevation in meters using high-precision bilinear interpolation at UV [0, 1].
   *
   * @param {number} u - Normalized horizontal coord [0, 1]
   * @param {number} v - Normalized vertical coord [0, 1]
   * @returns {number} Elevation in meters
   */
  sampleElevation(u, v) {
    if (!this.heightData || !this.width || !this.height) return this.minElevation;

    const fx = Math.min(Math.max(u, 0.0), 1.0) * (this.width - 1);
    const fy = Math.min(Math.max(v, 0.0), 1.0) * (this.height - 1);

    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(x0 + 1, this.width - 1);
    const y1 = Math.min(y0 + 1, this.height - 1);

    const tx = fx - x0;
    const ty = fy - y0;

    const w = this.width;
    const a = this.heightData[y0 * w + x0];
    const b = this.heightData[y0 * w + x1];
    const c = this.heightData[y1 * w + x0];
    const d = this.heightData[y1 * w + x1];

    // Bilinear blend
    const raw = (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty;

    if (this.isNormalized) {
      return this.minElevation + raw * this.elevationSpan;
    }
    return raw;
  }

  /**
   * Sample slope in degrees at UV [0, 1].
   * Computed using real horizontal physical distance in meters.
   *
   * @param {number} u - Normalized coord [0, 1]
   * @param {number} v - Normalized coord [0, 1]
   * @returns {number} Slope angle in degrees [0, 90]
   */
  sampleSlope(u, v) {
    if (!this.geoRef) return 0.0;

    const texelU = 1.0 / Math.max(1, this.width);
    const texelV = 1.0 / Math.max(1, this.height);

    const hL = this.sampleElevation(u - texelU, v);
    const hR = this.sampleElevation(u + texelU, v);
    const hU = this.sampleElevation(u, v - texelV);
    const hD = this.sampleElevation(u, v + texelV);

    const stepX_m = Math.max(2.0 * this.geoRef.worldWidth * texelU, 1e-4);
    const stepZ_m = Math.max(2.0 * this.geoRef.worldDepth * texelV, 1e-4);

    const dhdx = (hR - hL) / stepX_m;
    const dhdz = (hD - hU) / stepZ_m;

    const grad = Math.hypot(dhdx, dhdz);
    return (Math.atan(grad) * 180.0) / Math.PI;
  }

  /**
   * Sample surface normal vector at UV [0, 1].
   *
   * @param {number} u - Normalized coord [0, 1]
   * @param {number} v - Normalized coord [0, 1]
   * @returns {THREE.Vector3} Normalized surface normal in world space
   */
  sampleNormal(u, v) {
    const texelU = 1.0 / Math.max(1, this.width);
    const texelV = 1.0 / Math.max(1, this.height);

    const hL = this.sampleElevation(u - texelU, v);
    const hR = this.sampleElevation(u + texelU, v);
    const hU = this.sampleElevation(u, v - texelV);
    const hD = this.sampleElevation(u, v + texelV);

    const stepX_m = Math.max(2.0 * this.geoRef.worldWidth * texelU, 1e-4);
    const stepZ_m = Math.max(2.0 * this.geoRef.worldDepth * texelV, 1e-4);

    const dhdx = (hR - hL) / stepX_m;
    const dhdz = (hD - hU) / stepZ_m;

    const normal = new THREE.Vector3(-dhdx, 1.0, -dhdz);
    return normal.normalize();
  }

  /**
   * Direct sampling from raster pixel coordinates (integer).
   *
   * @param {number} px - Pixel column [0, width - 1]
   * @param {number} py - Pixel row [0, height - 1]
   * @returns {number} Elevation in meters
   */
  samplePixel(px, py) {
    const cx = Math.min(this.width - 1, Math.max(0, px));
    const cy = Math.min(this.height - 1, Math.max(0, py));
    const raw = this.heightData[cy * this.width + cx];
    if (this.isNormalized) {
      return this.minElevation + raw * this.elevationSpan;
    }
    return raw;
  }
}

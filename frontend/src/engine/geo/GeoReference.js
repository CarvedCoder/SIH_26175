/**
 * DepthWizard Geospatial Engine — GeoReference
 *
 * Centralized geospatial metadata, coordinate frame definitions, and scale derivations.
 *
 * CANONICAL RULES:
 *   1. 1 renderer world unit = 1 meter.
 *   2. Never assume 1 pixel = 1 meter unless actual metadata specifies GSD = 1.0 m/px.
 *   3. World-space axes convention:
 *        +X = East  (meters)
 *        +Y = Up    (elevation in meters)
 *        +Z = South (meters) / -Z = North (meters, Three.js camera forward default)
 *   4. Origin Rebasing: Local coordinates are centered around the local origin (0, 0, 0)
 *      to eliminate 32-bit floating point precision loss in WebGL shaders.
 */

export class GeoReference {
  /**
   * @param {Object} params
   * @param {string|null} [params.crs] - e.g. "EPSG:32617"
   * @param {string|null} [params.projectedCrs] - Projected metric CRS
   * @param {number[]|null} [params.affineTransform] - [a, b, c, d, e, f]
   * @param {number|null} [params.gsdX] - Easting meters per pixel
   * @param {number|null} [params.gsdY] - Northing meters per pixel
   * @param {number} params.rasterWidth - Width in pixels
   * @param {number} params.rasterHeight - Height in pixels
   * @param {number} [params.worldWidth] - Physical width in meters
   * @param {number} [params.worldDepth] - Physical depth in meters
   * @param {boolean} [params.isGeoreferenced] - True if genuine CRS and GSD exist
   * @param {number[]} [params.localOrigin] - [globalX, globalY, globalZ] for floating origin
   * @param {number} [params.minElevation] - Minimum elevation in meters
   * @param {number} [params.maxElevation] - Maximum elevation in meters
   * @param {string} [params.elevationMode] - "absolute" | "relative"
   */
  constructor(params = {}) {
    this.crs = params.crs ?? null;
    this.projectedCrs = params.projectedCrs ?? this.crs;
    this.affineTransform = params.affineTransform ?? null;

    this.rasterWidth = Math.max(1, params.rasterWidth || 1024);
    this.rasterHeight = Math.max(1, params.rasterHeight || 1024);

    this.isGeoreferenced = !!(params.isGeoreferenced || (this.crs && params.gsdX));

    // Ground Sampling Distance (GSD) in meters / pixel
    if (this.isGeoreferenced && params.gsdX && params.gsdY) {
      this.gsdX = Math.abs(Number(params.gsdX));
      this.gsdY = Math.abs(Number(params.gsdY));
      this.isRelativeMode = false;
    } else {
      // Non-georeferenced fallback: explicitly documented as assumed / relative
      this.gsdX = params.gsdX ? Math.abs(Number(params.gsdX)) : 1.0;
      this.gsdY = params.gsdY ? Math.abs(Number(params.gsdY)) : 1.0;
      this.isRelativeMode = true;
    }

    // Physical Footprint in meters
    this.worldWidth = params.worldWidth && params.worldWidth > 0
      ? Number(params.worldWidth)
      : this.rasterWidth * this.gsdX;

    this.worldDepth = params.worldDepth && params.worldDepth > 0
      ? Number(params.worldDepth)
      : this.rasterHeight * this.gsdY;

    // Elevation range in meters
    this.minElevation = typeof params.minElevation === 'number' ? params.minElevation : 0.0;
    this.maxElevation = typeof params.maxElevation === 'number' ? params.maxElevation : 100.0;
    this.elevationSpan = Math.max(0.001, this.maxElevation - this.minElevation);
    this.elevationMode = params.elevationMode || (this.isGeoreferenced ? 'absolute' : 'relative');

    // Local Origin for Origin Rebasing (projected coordinates mapped to (0,0,0) local)
    if (params.localOrigin && Array.isArray(params.localOrigin) && params.localOrigin.length >= 2) {
      this.localOrigin = [
        Number(params.localOrigin[0]),
        Number(params.localOrigin[1]),
        Number(params.localOrigin[2] || 0.0),
      ];
    } else {
      this.localOrigin = [0.0, 0.0, 0.0];
    }
  }

  /**
   * Convert normalized terrain UV coords [0, 1] x [0, 1] to local world coordinates (meters).
   * Local world (0, 0) is at the center of the terrain footprint.
   *
   * @param {number} u - Horizontal normalized coord [0, 1] (0 = West, 1 = East)
   * @param {number} v - Vertical normalized coord [0, 1] (0 = North, 1 = South)
   * @param {number} [elevation] - Elevation in meters
   * @returns {{ x: number, y: number, z: number }} Coordinates in meters
   */
  uvToLocal(u, v, elevation = 0) {
    const clampedU = Math.max(0, Math.min(1, u));
    const clampedV = Math.max(0, Math.min(1, v));
    const x = (clampedU - 0.5) * this.worldWidth;
    const z = (clampedV - 0.5) * this.worldDepth;
    return { x, y: elevation, z };
  }

  /**
   * Convert local world coordinates (meters) to normalized terrain UV coords [0, 1] x [0, 1].
   *
   * @param {number} x - Easting offset from center (meters)
   * @param {number} z - Southing offset from center (meters)
   * @returns {{ u: number, v: number, inBounds: boolean }}
   */
  localToUv(x, z) {
    const u = x / this.worldWidth + 0.5;
    const v = z / this.worldDepth + 0.5;
    const inBounds = u >= 0 && u <= 1 && v >= 0 && v <= 1;
    return {
      u: Math.max(0, Math.min(1, u)),
      v: Math.max(0, Math.min(1, v)),
      inBounds,
    };
  }

  /**
   * Convert source raster pixel coordinates (integer) to local world coordinates (meters).
   *
   * @param {number} px - Pixel X [0, rasterWidth - 1]
   * @param {number} py - Pixel Y [0, rasterHeight - 1]
   * @param {number} [elevation] - Elevation in meters
   * @returns {{ x: number, y: number, z: number }}
   */
  pixelToLocal(px, py, elevation = 0) {
    const u = (px + 0.5) / this.rasterWidth;
    const v = (py + 0.5) / this.rasterHeight;
    return this.uvToLocal(u, v, elevation);
  }

  /**
   * Convert local world coordinates (meters) to source raster pixel coordinates.
   *
   * @param {number} x - Local world X (meters)
   * @param {number} z - Local world Z (meters)
   * @returns {{ px: number, py: number, inBounds: boolean }}
   */
  localToPixel(x, z) {
    const { u, v, inBounds } = this.localToUv(x, z);
    const px = Math.min(this.rasterWidth - 1, Math.max(0, Math.round(u * this.rasterWidth - 0.5)));
    const py = Math.min(this.rasterHeight - 1, Math.max(0, Math.round(v * this.rasterHeight - 0.5)));
    return { px, py, inBounds };
  }

  /**
   * Rebase a global projected CRS coordinate (e.g. UTM Easting/Northing in meters)
   * into a local renderer coordinate (meters from origin).
   *
   * @param {number} globalX - Easting in projected CRS
   * @param {number} globalY - Northing in projected CRS
   * @param {number} [elevation] - Elevation in meters
   * @returns {{ x: number, y: number, z: number }}
   */
  projectedToLocal(globalX, globalY, elevation = 0) {
    const x = globalX - this.localOrigin[0];
    // In Projected CRS, Northing increases Northwards; in Three.js, -Z is Northwards.
    const z = -(globalY - this.localOrigin[1]);
    const y = elevation - (this.localOrigin[2] || 0.0);
    return { x, y, z };
  }

  /**
   * Convert a local renderer coordinate (meters) back to global projected CRS coordinates.
   *
   * @param {number} x - Local world X (meters)
   * @param {number} z - Local world Z (meters)
   * @param {number} [y] - Local world Y (meters)
   * @returns {{ globalX: number, globalY: number, elevation: number }}
   */
  localToProjected(x, z, y = 0) {
    const globalX = x + this.localOrigin[0];
    const globalY = -z + this.localOrigin[1];
    const elevation = y + (this.localOrigin[2] || 0.0);
    return { globalX, globalY, elevation };
  }

  /**
   * Returns a human-readable scale description for UI display.
   */
  getScaleDescription() {
    if (this.isGeoreferenced) {
      return {
        label: 'METRIC (GEOREFERENCED)',
        badge: 'CONFIRMED',
        gsdText: `${this.gsdX.toFixed(2)} m/px`,
        crsText: this.crs || 'Projected',
        footprintText: `${Math.round(this.worldWidth)}m × ${Math.round(this.worldDepth)}m`,
        isEstimated: false,
      };
    }
    return {
      label: 'RELATIVE / ESTIMATED SCALE',
      badge: 'ESTIMATED',
      gsdText: `Assumed ${this.gsdX.toFixed(1)} m/px`,
      crsText: 'Local Frame',
      footprintText: `~${Math.round(this.worldWidth)}m × ${Math.round(this.worldDepth)}m`,
      isEstimated: true,
    };
  }
}

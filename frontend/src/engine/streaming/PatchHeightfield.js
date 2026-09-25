/**
 * DepthWizard Geospatial Engine — PatchHeightfield
 *
 * Maps a streamed height-tile patch (a size×size normalized [0,1] PNG decoded
 * to Float32Array) back onto the global raster frame, so chunk geometry and
 * spatial queries can sample real elevations in METERS from tile-local data.
 *
 * The window arithmetic here MUST mirror the backend
 * `TerrainService._quadtree_window`: tile (x, y) at level z covers
 *   px [round(x*W/2^z), round((x+1)*W/2^z)) — columns over raster width,
 *   px [round(y*H/2^z), round((y+1)*H/2^z)) — rows over raster height,
 * with row 0 = north. Per-axis windows make non-square rasters tile cleanly.
 */

/**
 * Bilinear sample of a square patch at floating-point output coordinates.
 *
 * @param {Float32Array} data - size*size normalized values, row-major
 * @param {number} size - Patch edge length
 * @param {number} c - Column coordinate in output-pixel space [0, size-1]
 * @param {number} r - Row coordinate in output-pixel space [0, size-1]
 * @returns {number} Normalized [0, 1] value
 */
export function samplePatchBilinear(data, size, c, r) {
  const maxIdx = size - 1;
  const c0 = Math.floor(Math.min(Math.max(c, 0), maxIdx));
  const r0 = Math.floor(Math.min(Math.max(r, 0), maxIdx));
  const c1 = Math.min(c0 + 1, maxIdx);
  const r1 = Math.min(r0 + 1, maxIdx);
  const tc = Math.min(Math.max(c - c0, 0), 1);
  const tr = Math.min(Math.max(r - r0, 0), 1);

  const a = data[r0 * size + c0];
  const b = data[r0 * size + c1];
  const d = data[r1 * size + c0];
  const e = data[r1 * size + c1];

  return (a * (1 - tc) + b * tc) * (1 - tr) + (d * (1 - tc) + e * tc) * tr;
}

/** Integer pixel window of tile (x, y) at level z over an axis of `full` px —
 * mirrors backend TerrainService._quadtree_window exactly. */
export function quadtreeWindow(full, x, y, z) {
  const grid = 2 ** z;
  let x0 = Math.round((x * full) / grid);
  let x1 = Math.round(((x + 1) * full) / grid);
  let y0 = Math.round((y * full) / grid);
  let y1 = Math.round(((y + 1) * full) / grid);
  if (x1 <= x0) x1 = Math.min(full, x0 + 1);
  if (y1 <= y0) y1 = Math.min(full, y0 + 1);
  return [x0, x1, y0, y1];
}

export class PatchHeightfield {
  /**
   * @param {Object} options
   * @param {Float32Array} options.data - size*size normalized [0,1] heights
   * @param {number} options.size - Patch edge length in pixels
   * @param {number} options.z - Quadtree level the patch was served for
   * @param {number} options.tx - Tile column
   * @param {number} options.ty - Tile row
   * @param {number} options.rasterWidth - Full raster width in px
   * @param {number} options.rasterHeight - Full raster height in px
   * @param {number} options.minElevation - Scene-global minimum elevation (meters)
   * @param {number} options.maxElevation - Scene-global maximum elevation (meters)
   */
  constructor(options) {
    this.data = options.data;
    this.size = options.size;
    this.z = options.z;
    this.rasterWidth = options.rasterWidth;
    this.rasterHeight = options.rasterHeight;
    this.minElevation = options.minElevation;
    this.elevationSpan = Math.max(0.001, options.maxElevation - options.minElevation);

    const [wx0, wx1] = quadtreeWindow(options.rasterWidth, options.tx, options.ty, options.z);
    const [, , wy0, wy1] = quadtreeWindow(options.rasterHeight, options.tx, options.ty, options.z);
    this.winX0 = wx0;
    this.winW = Math.max(1, wx1 - wx0);
    this.winY0 = wy0;
    this.winH = Math.max(1, wy1 - wy0);
  }

  /**
   * Sample elevation in METERS at global raster UV [0, 1] (u: west→east, v: north→south).
   *
   * @param {number} u
   * @param {number} v
   * @returns {number} Elevation in meters
   */
  sampleGlobal(u, v) {
    // Global UV -> raster pixel coordinates (matches TerrainDataset: fx = u*(W-1))
    const pxCol = Math.min(Math.max(u, 0), 1) * (this.rasterWidth - 1);
    const pxRow = Math.min(Math.max(v, 0), 1) * (this.rasterHeight - 1);

    // Raster pixel -> patch output-pixel space. PIL resize maps output pixel i's
    // center to input coordinate (i + 0.5) * win / size - 0.5; inverting gives
    // the output coordinate that reproduces a given input pixel.
    const c = ((pxCol - this.winX0 + 0.5) * this.size) / this.winW - 0.5;
    const r = ((pxRow - this.winY0 + 0.5) * this.size) / this.winH - 0.5;

    const raw = samplePatchBilinear(this.data, this.size, c, r);
    return this.minElevation + raw * this.elevationSpan;
  }
}

/**
 * Convert a z=0 overview patch (one tile covering the whole raster, square
 * size×size, edge-padded for non-square rasters) into a rectangular normalized
 * heightfield that covers exactly the raster's pixel extents. The result feeds
 * TerrainDataset for global queries (collision, inspection, routing) without
 * loading a full-resolution heightmap (spec §29).
 *
 * @param {Object} options
 * @param {Float32Array} options.data - size*size normalized patch
 * @param {number} options.size - Patch edge length
 * @param {number} options.rasterWidth
 * @param {number} options.rasterHeight
 * @param {number} [options.maxEdge=512] - Longest edge of the returned array
 * @returns {{ data: Float32Array, width: number, height: number }}
 */
export function overviewToHeightfield(options) {
  const { data, size, rasterWidth, rasterHeight } = options;
  const maxEdge = options.maxEdge ?? 512;
  const scale = maxEdge / Math.max(rasterWidth, rasterHeight);
  const width = Math.max(2, Math.round(rasterWidth * scale));
  const height = Math.max(2, Math.round(rasterHeight * scale));

  const out = new Float32Array(width * height);
  // The overview patch is a z=0 tile: window == the full raster axis.
  for (let j = 0; j < height; j++) {
    const pxRow = (j / (height - 1)) * (rasterHeight - 1);
    const r = ((pxRow + 0.5) * size) / rasterHeight - 0.5;
    for (let i = 0; i < width; i++) {
      const pxCol = (i / (width - 1)) * (rasterWidth - 1);
      const c = ((pxCol + 0.5) * size) / rasterWidth - 0.5;
      out[j * width + i] = samplePatchBilinear(data, size, c, r);
    }
  }
  return { data: out, width, height };
}

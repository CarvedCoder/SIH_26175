/**
 * DepthWizard Geospatial Engine — heightDecode
 *
 * Decoding helpers for backend height tiles and heightmaps:
 *   - Primary: non-interlaced 16-bit grayscale PNG (color type 0, bit depth 16)
 *     decoded with full numerical precision into a Float32Array of normalized
 *     [0, 1] elevation values.
 *   - Fallback: 8-bit canvas decode for legacy artifacts.
 *
 * Moved out of TerrainCanvas so both the inline heightmap path and the tile
 * streaming path share one implementation.
 */

import { assetFetch } from '../../api/client.js';

/**
 * Decode a heightmap/height-tile resource into { data, width, height } with
 * normalized [0, 1] elevation values.
 *
 * This is a storage-asset fetch, not an authenticated API call: resolvedUrl
 * may resolve to a presigned MinIO URL or the legacy same-origin backend
 * route, and assetFetch() is what tells those apart safely. There is
 * deliberately no way to pass arbitrary auth headers into this function —
 * that was the leak: a caller could (and did) hand it the Supabase JWT
 * provider, which then rode along to MinIO and got rejected.
 *
 * @param {string} resolvedUrl - Already-resolved absolute/relative URL
 * @returns {Promise<{data: Float32Array, width: number, height: number}>}
 */
export async function decodeHeightmap(resolvedUrl) {
  const res = await assetFetch(resolvedUrl);
  if (!res.ok) throw new Error(`heightmap fetch failed: ${res.status}`);

  const buf = await res.arrayBuffer();
  try {
    const parsed = await decodeHeightPng16(buf);
    if (parsed) return parsed;
  } catch (err) {
    console.warn('[terrain] 16-bit heightmap decode failed, falling back to 8-bit canvas decode', err);
  }
  // Non-PNG bodies (e.g. an auth error page) must fail fast with a clear
  // error — never enter the canvas fallback, which can hang on bad input.
  if (!isPngBuffer(buf)) {
    throw new Error(
      `heightmap is not a PNG image (got ${buf.byteLength} bytes — ` +
      'check authentication/storage delivery)'
    );
  }
  return decodeHeightViaCanvasBytes(buf);
}

/** Cheap signature check to reject JSON/error bodies before decoding. */
export function isPngBuffer(buf) {
  if (buf.byteLength < 8) return false;
  const b = new Uint8Array(buf, 0, 8);
  return b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47;
}

/**
 * Parse a non-interlaced 16-bit grayscale PNG (color type 0, bit depth 16).
 * Returns null when the buffer is a different PNG flavor.
 *
 * @param {ArrayBuffer} buf
 * @returns {Promise<{data: Float32Array, width: number, height: number}|null>}
 */
export async function decodeHeightPng16(buf) {
  const bytes = new Uint8Array(buf);
  const SIG = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) {
    if (bytes[i] !== SIG[i]) throw new Error('not a PNG');
  }
  const dv = new DataView(buf);
  let p = 8;
  let width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  const idat = [];
  while (p + 8 <= bytes.length) {
    const len = dv.getUint32(p);
    const type = String.fromCharCode(bytes[p + 4], bytes[p + 5], bytes[p + 6], bytes[p + 7]);
    const data = bytes.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(p + 8);
      height = dv.getUint32(p + 12);
      bitDepth = bytes[p + 16];
      colorType = bytes[p + 17];
      interlace = bytes[p + 20];
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    p += 12 + len;
  }
  if (colorType !== 0 || bitDepth !== 16 || interlace !== 0) return null;

  const stream = new Blob(idat).stream().pipeThrough(new DecompressionStream('deflate'));
  const raw = new Uint8Array(await new Response(stream).arrayBuffer());

  const bpp = 2;
  const stride = width * bpp;
  const out = new Uint8Array(height * stride);
  let src = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[src++];
    const row = out.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? row[x - bpp] : 0;
      const b = prev ? prev[x] : 0;
      const c = (prev && x >= bpp) ? prev[x - bpp] : 0;
      let v = raw[src + x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      row[x] = v & 0xff;
    }
    src += stride;
  }

  const data = new Float32Array(width * height);
  for (let i = 0; i < data.length; i++) {
    data[i] = ((out[i * 2] << 8) | out[i * 2 + 1]) / 65535;
  }
  return { data, width, height };
}

/** 8-bit fallback decode from raw image bytes. */
export async function decodeHeightViaCanvasBytes(buf) {
  const blob = new Blob([buf]);
  const bitmap = await createImageBitmap(blob);
  return decodeBitmapToHeightfield(bitmap);
}

/** 8-bit fallback decode from a URL (kept for callers holding a URL only). */
export async function decodeHeightViaCanvas(resolvedUrl) {
  const res = await fetch(resolvedUrl);
  const blob = await res.blob();
  const bitmap = await createImageBitmap(blob);
  return decodeBitmapToHeightfield(bitmap);
}

async function decodeBitmapToHeightfield(bitmap) {
  const { width, height } = bitmap;
  let ctx;
  if (typeof OffscreenCanvas !== 'undefined') {
    const oc = new OffscreenCanvas(width, height);
    ctx = oc.getContext('2d');
  } else {
    const c = document.createElement('canvas');
    c.width = width;
    c.height = height;
    ctx = c.getContext('2d');
  }
  ctx.drawImage(bitmap, 0, 0);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const data = new Float32Array(width * height);
  for (let i = 0; i < data.length; i++) data[i] = pixels[i * 4] / 255;
  bitmap.close();
  return { data, width, height };
}

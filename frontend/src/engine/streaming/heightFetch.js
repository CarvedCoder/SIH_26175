/**
 * DepthWizard Geospatial Engine — heightFetch
 *
 * The auth-aware transport for the full-scene heightmap. Separated from
 * heightDecode.js (the pure decoders) so the engine's Node-runnable code
 * paths never pull in api/client.js and its Vite path aliases.
 *
 * decodeHeightmap is a storage-asset fetch, not an authenticated API call:
 * resolvedUrl may resolve to a presigned MinIO URL or the legacy same-origin
 * backend route, and assetFetch() is what tells those apart safely. There is
 * deliberately no way to pass arbitrary auth headers into this function —
 * that was the leak: a caller could (and did) hand it the Supabase JWT
 * provider, which then rode along to MinIO and got rejected.
 */

import { assetFetch } from '../../api/client.js';
import { decodeHeightmapBytes } from './heightDecode.js';

/**
 * @param {string} resolvedUrl - Already-resolved absolute/relative URL
 * @returns {Promise<{data: Float32Array, width: number, height: number}>}
 */
export async function decodeHeightmap(resolvedUrl) {
  const res = await assetFetch(resolvedUrl);
  if (!res.ok) throw new Error(`heightmap fetch failed: ${res.status}`);
  return decodeHeightmapBytes(await res.arrayBuffer());
}

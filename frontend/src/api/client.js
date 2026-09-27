/**
 * DepthWizard — API Client
 *
 * One base URL. Consistent error shape. Never call fetch() directly from components.
 * See DECISIONS.md §D03 and spec §69, §72.
 */

import { getAccessToken, supabase } from '@/lib/supabase';

// Relative by default: the vite dev server proxies /api to the backend
// (see vite.config.js), so a stale VITE_API_BASE_URL can never point the
// browser at a dead port. Absolute overrides still win when explicitly set.
export const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '');

export function getBaseUrl() {
  return BASE_URL;
}

/**
 * Attach the Supabase access token (JWT) to every API request. The
 * backend verifies the token and derives the user id from its `sub`
 * claim; we never send a user id in payloads.
 */
export async function authHeaders() {
  const token = await getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Normalise any backend error into our standard error shape.
 * Backend contract: { error: { code, message, details, recoverable } }
 *
 * Every failure is logged to the console (with the backend request id
 * when present) so "the UI shows nothing" can always be debugged from
 * devtools.
 * @param {Response} res
 * @returns {Promise<never>}
 */
async function throwApiError(res) {
  let body;
  try { body = await res.json(); } catch { body = {}; }

  const raw = body?.error ?? {};
  console.warn(
    `[api] ${res.status} ${res.url}`,
    JSON.stringify({
      code: raw.code ?? 'INTERNAL_ERROR',
      message: raw.message ?? `HTTP ${res.status}`,
      request_id: res.headers?.get?.('X-Request-ID') ?? null,
    }),
  );
  throw {
    code:        raw.code        ?? 'INTERNAL_ERROR',
    message:     raw.message     ?? `HTTP ${res.status}`,
    recoverable: raw.recoverable ?? false,
    details:     raw.details     ?? {},
    status:      res.status,
  };
}

/**
 * Core fetch wrapper. All API modules go through this.
 *
 * @param {string} path   - path after BASE_URL, e.g. '/scenes'
 * @param {RequestInit} [options]
 * @returns {Promise<any>}
 */
export async function apiFetch(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  let res;
  // One refresh-retry on 401: the Supabase access token may have expired
  // between requests; refresh the session once and try again.
  for (let attempt = 0; ; attempt++) {
    const auth = await authHeaders();
    try {
      res = await fetch(url, {
        ...options,
        headers: { 'Accept': 'application/json', ...auth, ...options.headers },
      });
    } catch (netErr) {
      throw {
        code: 'NETWORK_ERROR',
        message: `Backend unreachable at ${BASE_URL}. Ensure the FastAPI server is running.`,
        recoverable: true,
        details: { error: netErr?.message },
        status: 0,
      };
    }
    if (res.status !== 401 || attempt > 0 || !supabase) break;
    const { data } = await supabase.auth.refreshSession();
    if (!data?.session) break;
  }

  if (!res.ok) await throwApiError(res);

  // 204 No Content
  if (res.status === 204) return null;

  return res.json();
}

/**
 * Authorized download: fetches a protected artifact as a blob (with the
 * Supabase token attached; signed-URL redirects are followed) and hands
 * it to the browser as a download. Used by the export module.
 * @param {string} url
 * @param {string} [filename]
 */
export async function downloadArtifact(url, filename) {
  const auth = await authHeaders();
  const res = await fetch(url, { headers: auth });
  if (!res.ok) {
    await throwApiError(res);
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = objectUrl;
  if (filename) a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objectUrl);
}

/**
 * Fetch a "result asset" URL — heightmaps, textures, previews, downloads —
 * as opposed to a JSON API call (use apiFetch for those).
 *
 * These URLs come from resolveAssetUrl() and can resolve to one of two
 * fundamentally different things, decided at RUNTIME by the backend's
 * storage config (see backend `storage/service.py::presign_artifact`):
 *
 *   1. A short-lived S3 (RustFS) presigned URL (absolute, a DIFFERENT origin
 *      from our API). It already carries its own signature in the query
 *      string. It MUST NOT receive the Supabase JWT: the store rejects the
 *      request (400) when an unexpected Authorization header is present,
 *      and Chrome reports that as a CORS failure because the 400 response
 *      has no CORS headers on it.
 *   2. The legacy same-origin FastAPI file route (e.g.
 *      `/api/v1/scenes/{id}/results/heightmap`), used only when no object
 *      store is configured. This route enforces the exact same ownership
 *      guard as every other backend endpoint, so it DOES need the JWT —
 *      it just isn't reached through apiFetch because it returns raw
 *      bytes, not JSON.
 *
 * assetFetch tells these apart from the URL itself — never from which
 * call site invoked it — so application auth can never leak into a
 * cross-origin storage request no matter which code path resolves it.
 *
 * @param {string} url - Usually the output of resolveAssetUrl().
 * @param {RequestInit} [options] - Safe transport options (signal, cache, ...).
 */
export async function assetFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (isSameOriginAsApi(url)) {
    // Same-origin backend route standing in for a storage asset — treat
    // it like any other authenticated backend request.
    Object.assign(headers, await authHeaders());
  }
  // Different origin: a presigned storage URL. Never attach application
  // auth here — its only "auth" is the signature already in its query
  // string, and it must travel alone.
  return fetch(url, { ...options, headers });
}

function isSameOriginAsApi(url) {
  if (!/^https?:\/\//i.test(url)) return true; // relative path => same origin
  try {
    const target = new URL(url);
    const apiOrigin = /^https?:\/\//i.test(BASE_URL)
      ? new URL(BASE_URL).origin
      : window.location.origin;
    return target.origin === apiOrigin;
  } catch {
    // Unparseable URL: fail closed — never attach the JWT to it.
    return false;
  }
}

/**
 * Resolve an asset URL returned by the backend against the API base.
 *
 * Backend asset URLs are relative paths (e.g. /api/v1/scenes/x/results/preview).
 * Components that load images directly (new Image(), fetch(), <a href>) must
 * resolve them against VITE_API_BASE_URL — a bare relative path would hit
 * the frontend origin instead of the API.
 * @param {string|null} url
 * @returns {string|null}
 */
export function resolveAssetUrl(url) {
  if (!url) return null;
  if (/^https?:\/\//i.test(url) || url.startsWith('data:')) return url;
  // Asset paths are API-rooted (they already include /api/v1), so they
  // resolve against the API ORIGIN only — never against BASE_URL's path.
  const path = url.startsWith('/') ? url : `/${url}`;
  if (/^https?:\/\//i.test(BASE_URL)) {
    return `${new URL(BASE_URL).origin}/${path.replace(/^\//, '')}`;
  }
  // Relative API base (vite dev proxy): same-origin, path already API-rooted.
  return path;
}

/**
 * Lightweight health probe — used by the status indicator.
 * Does NOT throw on failure — returns null so callers can show "offline" state.
 * @returns {Promise<{ status: string, version: string, model_loaded: boolean }|null>}
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

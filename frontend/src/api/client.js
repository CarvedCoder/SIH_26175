/**
 * DepthWizard — API Client
 *
 * One base URL. Consistent error shape. Never call fetch() directly from components.
 * See DECISIONS.md §D03 and spec §69, §72.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1').replace(/\/$/, '');

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
  try {
    res = await fetch(url, {
      headers: { 'Accept': 'application/json', ...options.headers },
      ...options,
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

  if (!res.ok) await throwApiError(res);

  // 204 No Content
  if (res.status === 204) return null;

  return res.json();
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
  return `${new URL(BASE_URL).origin}/${path.replace(/^\//, '')}`;
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

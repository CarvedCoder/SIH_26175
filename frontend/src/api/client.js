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
 * @param {Response} res
 * @returns {Promise<never>}
 */
async function throwApiError(res) {
  let body;
  try { body = await res.json(); } catch { body = {}; }

  const raw = body?.error ?? {};
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

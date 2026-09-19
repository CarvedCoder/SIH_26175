/**
 * DepthWizard — Supabase client
 *
 * Supabase OWNS authentication: signup, login, session persistence and
 * token refresh all happen here. The backend verifies the JWT we attach
 * (see src/api/client.js) and derives the user id from its `sub` claim —
 * user ids are never taken from request payloads.
 *
 * Configuration (Vite env):
 *   VITE_SUPABASE_URL      — project URL
 *   VITE_SUPABASE_ANON_KEY — public anon key (safe to expose; RLS still
 *                            applies server-side)
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabaseConfigured = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

export const supabase = supabaseConfigured
  ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        // Refresh access tokens automatically before expiry and persist
        // the session so a reload keeps the user signed in.
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true,
      },
    })
  : null;

/**
 * The current Supabase access token (JWT) for the Authorization header,
 * or null when Supabase is not configured / no session exists.
 * @returns {Promise<string|null>}
 */
export async function getAccessToken() {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

import { createContext, useContext, useState, useEffect } from 'react';
import { supabase, supabaseConfigured } from '@/lib/supabase';

const AuthContext = createContext(null);

const STORAGE_KEY = 'depthwizard_auth_user';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  // Default view is 'landing'
  const [view, setView] = useState('landing'); // 'landing' | 'auth' | 'app' | 'recent' | 'help'

  // Sync user state to localStorage
  useEffect(() => {
    try {
      if (user) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch (e) {
      console.warn('Failed to sync auth user to localStorage:', e);
    }
  }, [user]);

  // Supabase session restore + live auth-state sync. The backend derives
  // the user id from the JWT sub claim; the profile here is display-only.
  useEffect(() => {
    if (!supabase) return undefined;

    let mounted = true;

    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      const profile = data.session?.user
        ? profileFromSupabase(data.session.user)
        : null;
      if (profile) {
        setUser(profile);
        setView('app');
      }
    });

    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!mounted) return;
      if (session?.user) {
        setUser(profileFromSupabase(session.user));
        setView('app');
      } else {
        setUser(null);
      }
    });

    return () => {
      mounted = false;
      sub.subscription?.unsubscribe?.();
    };
  }, []);

  const login = (profile) => {
    setUser(profile);
    setView('app');
  };

  const logout = async () => {
    if (supabase) await supabase.auth.signOut();
    setUser(null);
    setView('landing');
  };

  /**
   * Real Supabase email/password sign-in. Throws { message } on failure.
   * LOCAL MODE (Supabase unconfigured — e.g. plain local dev): the profile
   * is derived from the email and kept on this device only; there is no
   * server-side account and the backend runs with auth disabled.
   */
  const signInWithPassword = async (email, password) => {
    if (!supabase) return login(localProfile(email));
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) throw { message: error.message, code: error.code ?? 'AUTH_FAILED' };
    const profile = profileFromSupabase(data.user);
    login(profile);
    return profile;
  };

  /**
   * Real Supabase email/password sign-up. May return a user with no
   * session when email confirmation is enabled — the UI should say so.
   * LOCAL MODE: no server round-trip; the profile is created locally.
   */
  const signUpWithPassword = async (email, password) => {
    if (!supabase) return { profile: login(localProfile(email)), needsConfirmation: false };
    const { data, error } = await supabase.auth.signUp({ email, password });
    if (error) throw { message: error.message, code: error.code ?? 'AUTH_FAILED' };
    if (data.session && data.user) login(profileFromSupabase(data.user));
    return {
      user: data.user,
      profile: data.user ? profileFromSupabase(data.user) : null,
      needsConfirmation: !data.session,
    };
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        view,
        setView,
        login,
        logout,
        signInWithPassword,
        signUpWithPassword,
        supabaseConfigured,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

/**
 * LOCAL MODE profile: stable pseudo-id per email so scene ownership stays
 * consistent across reloads of the same browser profile.
 */
function localProfile(email) {
  const clean = (email ?? '').trim().toLowerCase() || 'guest@depthwizard.io';
  let hash = 0;
  for (let i = 0; i < clean.length; i += 1) {
    hash = (hash * 31 + clean.charCodeAt(i)) >>> 0;
  }
  const namePart = clean.split('@')[0] || 'Explorer';
  const name =
    namePart
      .split(/[.\-_]/)
      .filter(Boolean)
      .map((p) => p.charAt(0).toUpperCase() + p.slice(1).toLowerCase())
      .join(' ') || 'Explorer';
  return { id: `local_${hash.toString(16)}`, name, email: clean };
}

function profileFromSupabase(supabaseUser) {
  const email = supabaseUser.email ?? '';
  const namePart = email.split('@')[0] || 'Explorer';
  const name =
    namePart
      .split(/[.\-_]/)
      .filter(Boolean)
      .map((p) => p.charAt(0).toUpperCase() + p.slice(1).toLowerCase())
      .join(' ') || 'Explorer';
  return {
    id: supabaseUser.id,
    name,
    email,
  };
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return ctx;
}

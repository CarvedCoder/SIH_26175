import { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { LoginForm } from './login-form';
import { SignupForm } from './signup-form';
import { useAuth } from '@/store/authContext';

export default function AuthPage({ onAuthenticate, onBackToHome, initialView = 'login' }) {
  const [view, setView] = useState(initialView);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const { signInWithPassword, signUpWithPassword, supabaseConfigured } = useAuth();

  // Real Supabase authentication: email/password via the auth context.
  // The backend later derives the user id from the verified JWT sub —
  // this profile is display-only.
  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    const form = e.currentTarget;
    const emailInput = form.querySelector('input[type="email"]');
    const passwordInput = form.querySelector('input[type="password"]');
    const email = emailInput?.value?.trim() || '';
    const password = passwordInput?.value || '';
    if (!email || !password) {
      setError('Email and password are required.');
      return;
    }
    // Local mode has no server-side policy — enforce the advertised rule.
    if (!supabaseConfigured && password.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }

    setBusy(true);
    try {
      if (view === 'login') {
        const profile = await signInWithPassword(email, password);
        onAuthenticate?.(profile);
      } else {
        const { profile, needsConfirmation } = await signUpWithPassword(email, password);
        if (needsConfirmation) {
          setError('Check your inbox to confirm your email address, then sign in.');
          setView('login');
          setBusy(false);
          return;
        }
        onAuthenticate?.(profile);
      }
    } catch (err) {
      setError(err?.message || 'Authentication failed.');
    } finally {
      setBusy(false);
    }
  };

  const handleSocialLogin = (provider) => {
    setError(`${provider} sign-in is not configured yet — use email and password.`);
  };

  return (
    <div className="dark min-h-screen flex flex-col justify-center items-center p-4 sm:p-6 bg-zinc-950 relative selection:bg-zinc-800 selection:text-zinc-100">
      {/* Top Bar with Back Button and Brand */}
      <header className="absolute top-0 left-0 right-0 h-20 px-6 sm:px-12 flex items-center justify-between z-10">
        <button
          onClick={onBackToHome}
          className="flex items-center gap-2.5 text-sm font-medium text-zinc-400 hover:text-zinc-100 transition-colors duration-200 cursor-pointer focus:outline-none group"
        >
          <div className="p-1.5 rounded-full bg-zinc-900 border border-zinc-800 group-hover:border-zinc-700 transition-colors">
            <ArrowLeft className="w-4 h-4 transition-transform duration-200 group-hover:-translate-x-0.5" />
          </div>
          <span>Back to overview</span>
        </button>

        <button
          onClick={onBackToHome}
          className="font-display font-medium text-lg tracking-tight text-zinc-200 hover:text-white transition-colors cursor-pointer focus:outline-none"
        >
          DepthWizard
        </button>
      </header>

      {/* Auth Card Container */}
      <div className="w-full max-w-4xl relative z-10 pt-16 sm:pt-0">
        {!supabaseConfigured && (
          <div
            role="status"
            className="mb-4 px-4 py-3 rounded-lg bg-amber-950/50 border border-amber-800/80 text-amber-200 text-sm text-center"
          >
            Local mode — Supabase is not configured. Accounts stay on this
            device only; enter any email and a password (8+ characters) to
            continue.
          </div>
        )}
        {error && (
          <div role="alert" className="mb-4 px-4 py-3 rounded-lg bg-red-950/60 border border-red-800/80 text-red-200 text-sm text-center">
            {error}
          </div>
        )}
        {view === 'login' ? (
          <div
            onClick={(e) => {
              const link = e.target.closest('a');
              if (link && (link.textContent?.includes('Sign up') || link.getAttribute('href') === '#signup')) {
                e.preventDefault();
                setError(null);
                setView('signup');
              }
            }}
          >
            <LoginForm onSocialLogin={handleSocialLogin} onSubmit={handleAuthSubmit} busy={busy} />
          </div>
        ) : (
          <div
            onClick={(e) => {
              const link = e.target.closest('a');
              if (link && (link.textContent?.includes('Sign in') || link.getAttribute('href') === '#signin')) {
                e.preventDefault();
                setError(null);
                setView('login');
              }
            }}
          >
            <SignupForm onSocialLogin={handleSocialLogin} onSubmit={handleAuthSubmit} busy={busy} />
          </div>
        )}
      </div>
    </div>
  );
}

import { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { LoginForm } from './login-form';
import { SignupForm } from './signup-form';
import { useAuth } from '@/store/authContext';

export default function AuthPage({ onAuthenticate, onBackToHome, initialView = 'login' }) {
  const [view, setView] = useState(initialView);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const { signInWithPassword, signUpWithPassword } = useAuth();

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
    <div className="dark min-h-screen flex flex-col justify-center items-center p-4 sm:p-6 bg-slate-950 relative selection:bg-blue-600 selection:text-white">
      {/* Subtle background glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Top Bar with Back Button and Brand */}
      <header className="absolute top-0 left-0 right-0 h-20 px-6 sm:px-12 flex items-center justify-between z-10">
        <button
          onClick={onBackToHome}
          className="flex items-center gap-2.5 text-sm font-medium text-slate-400 hover:text-slate-100 transition-colors duration-200 cursor-pointer focus:outline-none group"
        >
          <div className="p-1.5 rounded-full bg-slate-900 border border-slate-800 group-hover:border-slate-700 transition-colors">
            <ArrowLeft className="w-4 h-4 transition-transform duration-200 group-hover:-translate-x-0.5" />
          </div>
          <span>Back to overview</span>
        </button>

        <button
          onClick={onBackToHome}
          className="font-display font-medium text-lg tracking-tight text-slate-200 hover:text-white transition-colors cursor-pointer focus:outline-none"
        >
          TerraMesh
        </button>
      </header>

      {/* Auth Card Container */}
      <div className="w-full max-w-4xl relative z-10 pt-16 sm:pt-0">
        {error && (
          <div role="alert" className="mb-4 px-4 py-3 rounded-lg bg-red-950/60 border border-red-800 text-red-200 text-sm text-center">
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

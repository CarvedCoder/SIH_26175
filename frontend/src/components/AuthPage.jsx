import { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { LoginForm } from './login-form';
import { SignupForm } from './signup-form';

export default function AuthPage({ onAuthenticate, onBackToHome, initialView = 'login' }) {
  const [view, setView] = useState(initialView);

  // Authentication handler to extract name from email
  const handleAuthSubmit = (e) => {
    e.preventDefault();
    const emailInput = e.target.querySelector('input[type="email"]');
    const email = emailInput?.value?.trim() || 'user@terramesh.io';
    
    // Extract name from email (e.g., john.doe@example.com -> John Doe)
    const namePart = email.split('@')[0];
    const name = namePart
      .split(/[\.\-\_]/)
      .filter(Boolean)
      .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(' ');

    onAuthenticate?.({
      name: name || 'Explorer',
      email: email
    });
  };

  const handleSocialLogin = (provider) => {
    onAuthenticate?.({
      name: `${provider} User`,
      email: `user@${provider.toLowerCase()}.com`
    });
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
      <div className="w-full max-w-4xl relative z-10 pt-16 sm:pt-0" onSubmit={handleAuthSubmit}>
        {view === 'login' ? (
          <div
            onClick={(e) => {
              const link = e.target.closest('a');
              if (link && (link.textContent?.includes('Sign up') || link.getAttribute('href') === '#signup')) {
                e.preventDefault();
                setView('signup');
              }
            }}
          >
            <LoginForm onSocialLogin={handleSocialLogin} />
          </div>
        ) : (
          <div
            onClick={(e) => {
              const link = e.target.closest('a');
              if (link && (link.textContent?.includes('Sign in') || link.getAttribute('href') === '#signin')) {
                e.preventDefault();
                setView('login');
              }
            }}
          >
            <SignupForm onSocialLogin={handleSocialLogin} />
          </div>
        )}
      </div>
    </div>
  );
}

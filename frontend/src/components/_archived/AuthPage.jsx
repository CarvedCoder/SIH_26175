import { useState } from 'react';
import { LoginForm } from './login-form';
import { SignupForm } from './signup-form';

export default function AuthPage({ onAuthenticate, initialView = 'login' }) {
  const [view, setView] = useState(initialView);

  // Authentication handler to extract name from email
  const handleAuthSubmit = (e) => {
    e.preventDefault();
    const emailInput = e.target.querySelector('input[type="email"]');
    const email = emailInput ? emailInput.value : 'user@example.com';
    
    // Extract name from email (e.g., john.doe@example.com -> John Doe)
    const namePart = email.split('@')[0];
    const name = namePart
      .split(/[\.\-\_]/)
      .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(' ');

    onAuthenticate({
      name: name || 'User',
      email: email
    });
  };

  return (
    <div className="dark min-h-screen flex items-center justify-center p-4 bg-slate-950">
      <div className="w-full max-w-4xl" onSubmit={handleAuthSubmit}>
        {view === 'login' ? (
          <div onClick={(e) => {
            if (e.target.tagName === 'A' && e.target.textContent === 'Sign up') {
              e.preventDefault();
              setView('signup');
            }
          }}>
            <LoginForm />
          </div>
        ) : (
          <div onClick={(e) => {
            if (e.target.tagName === 'A' && e.target.textContent === 'Sign in') {
              e.preventDefault();
              setView('login');
            }
          }}>
            <SignupForm />
          </div>
        )}
      </div>
    </div>
  );
}

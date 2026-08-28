import { useState } from 'react';
import Navbar from './components/Navbar';
import LandingHero from './components/LandingHero';
import AuthPage from './components/AuthPage';
import UploadProcessingView from './components/UploadProcessingView';

function App() {
  const [view, setView] = useState('landing'); // 'landing' | 'auth' | 'upload'
  const [user, setUser] = useState(null);

  const handleOpenAuth = () => {
    setView('auth');
  };

  const handleAuthenticate = (profile) => {
    setUser(profile);
    setView('upload');
  };

  const handleSignOut = () => {
    setUser(null);
    setView('landing');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      
      {/* Minimal Top Navigation */}
      {view !== 'auth' && (
        <Navbar
          user={user}
          onOpenAuth={handleOpenAuth}
          onSignOut={handleSignOut}
          onNavigateHome={() => setView('landing')}
        />
      )}

      {/* Main View Transition */}
      <main>
        {view === 'landing' ? (
          <LandingHero
            onOpenAuth={handleOpenAuth}
          />
        ) : view === 'auth' ? (
          <AuthPage 
            onAuthenticate={handleAuthenticate}
          />
        ) : (
          <UploadProcessingView
            onBackToHome={() => setView('landing')}
          />
        )}
      </main>

    </div>
  );
}

export default App;

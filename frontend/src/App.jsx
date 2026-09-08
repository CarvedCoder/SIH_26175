/**
 * DepthWizard — Root app shell
 *
 * Routing is driven by:
 * 1. Auth & Landing state machine (landing → auth → app) in authContext.
 * 2. In-app workflow state machine (NO_SCENE → UPLOAD → PROCESSING → RESULTS → TERRAIN) in appStore.
 * See DECISIONS.md §D02.
 */
import { AppProvider, useApp, AppState } from './store/appStore.jsx';
import { AuthProvider, useAuth } from './store/authContext.jsx';
import ErrorBoundary from './components/common/ErrorBoundary.jsx';
import Header from './components/common/Header.jsx';
import ApiErrorAlert from './components/common/ApiErrorAlert.jsx';

/* Landing & Auth Components */
import LandingHero from './components/LandingHero.jsx';
import Navbar from './components/Navbar.jsx';
import AuthPage from './components/AuthPage.jsx';

/* Application Workflow Pages */
import Home from './pages/Home.jsx';
import Processing from './pages/Processing.jsx';
import ResultDashboard from './pages/ResultDashboard.jsx';
import TerrainWorkspace from './pages/TerrainWorkspace.jsx';

function FailedPage() {
  const { state } = useApp();
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />
      <main style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}>
        <ApiErrorAlert error={state.error} />
      </main>
    </div>
  );
}

function AppRoutes() {
  const { state } = useApp();

  switch (state.status) {
    case AppState.NO_SCENE:
    case AppState.UPLOADING:
    case AppState.SCENE_READY:
      return <Home />;

    case AppState.PROCESSING:
      return <Processing />;

    case AppState.RESULTS_READY:
      return <ResultDashboard />;

    case AppState.TERRAIN_LOADING:
    case AppState.TERRAIN_READY:
    case AppState.ANALYSIS:
      return <TerrainWorkspace />;

    case AppState.FAILED:
      return <FailedPage />;

    default:
      return <Home />;
  }
}

function AppContent() {
  const { view, user, login, logout, setView } = useAuth();

  /* 1. Landing View: Hero with WebGL Halftone visualizer & Top Nav */
  if (view === 'landing') {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
        <Navbar
          user={user}
          onOpenAuth={() => setView('auth')}
          onSignOut={logout}
          onNavigateHome={() => setView('landing')}
          onOpenApp={() => setView('app')}
        />
        <main>
          <LandingHero
            user={user}
            onOpenAuth={() => setView('auth')}
            onOpenApp={() => setView('app')}
          />
        </main>
      </div>
    );
  }

  /* 2. Auth View: Login and Signup forms */
  if (view === 'auth') {
    return (
      <AuthPage
        onAuthenticate={(profile) => {
          login(profile);
          setView('app');
        }}
        onBackToHome={() => setView('landing')}
      />
    );
  }

  /* 3. Main Workspace: How it is now */
  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'var(--dw-void)',
        color: 'var(--dw-fg)',
        fontFamily: 'var(--dw-font-ui)',
      }}
    >
      <AppRoutes />
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <AppProvider>
          <AppContent />
        </AppProvider>
      </AuthProvider>
    </ErrorBoundary>
  );
}

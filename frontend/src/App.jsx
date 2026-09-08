/**
 * DepthWizard — Root app shell
 *
 * Routing is driven entirely by the state machine in appStore.
 * No React Router — pages are determined by `state.status`.
 * See DECISIONS.md §D02.
 */
import { AppProvider, useApp, AppState } from './store/appStore.jsx';
import ErrorBoundary from './components/common/ErrorBoundary.jsx';
import Header from './components/common/Header.jsx';
import ApiErrorAlert from './components/common/ApiErrorAlert.jsx';

/* Pages */
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

export default function App() {
  return (
    <ErrorBoundary>
      <AppProvider>
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
      </AppProvider>
    </ErrorBoundary>
  );
}

/**
 * DepthWizard — Root app shell
 *
 * Routing is driven entirely by the state machine in appStore.
 * No React Router — pages are determined by `state.status`.
 * See DECISIONS.md §D02.
 */
import { AppProvider, useApp, AppState } from './store/appStore.jsx';

/* Pages */
import Home from './pages/Home.jsx';
import Processing from './pages/Processing.jsx';
import ResultDashboard from './pages/ResultDashboard.jsx';
function TerrainWorkspace()  { return <PageStub label="3D Terrain Workspace" />; }
function FailedPage()        { return <PageStub label="Error" />; }

function PageStub({ label }) {
  return (
    <div className="flex items-center justify-center h-screen" style={{ color: 'var(--dw-fg-muted)', fontFamily: 'var(--dw-font-ui)' }}>
      <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 13 }}>{label} — coming soon</span>
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
  );
}

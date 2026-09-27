/**
 * DepthWizard — Recent Projects page
 *
 * Standalone session-history page (view = 'recent').
 * Lists every persisted scene with resume / terrain-jump / remove actions,
 * and hosts the primary "New reconstruction" entry point.
 *
 * DESIGN.md: Operate mode. Instrument panel aesthetic — labelled list, no
 * card chrome beyond thin --dw-rim borders. Spec §28.
 */
import { Mountain, Upload } from 'lucide-react';
import Header from '../components/common/Header.jsx';
import RecentProjects from '../components/common/RecentProjects.jsx';
import { useApp } from '../store/appStore.jsx';
import { useAuth } from '../store/authContext.jsx';

export default function RecentProjectsPage() {
  const { actions } = useApp();
  const { setView } = useAuth();

  const startNew = () => {
    actions.reset();
    setView('app');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: 28,
        width: '100%',
        maxWidth: 880,
        margin: '0 auto',
        padding: '36px 24px 64px',
      }}>

        {/* Page heading + primary action */}
        <section style={{
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          gap: 16,
          flexWrap: 'wrap',
          width: '100%',
        }} aria-labelledby="recent-heading">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0 }}>
            <h1
              id="recent-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 'clamp(1.45rem, 2.2vw, 1.75rem)',
                fontWeight: 600,
                letterSpacing: '-0.02em',
                color: 'var(--dw-fg)',
                lineHeight: 1.2,
                margin: 0,
              }}
            >
              Recent projects
            </h1>
            <p style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 14.5,
              color: 'var(--dw-fg-muted)',
              margin: 0,
              lineHeight: 1.6,
            }}>
              Every scene you upload or process is saved on this device — resume
              results, jump back into 3D terrain, or clear old entries.
            </p>
          </div>

          <button
            onClick={startNew}
            style={{
              height: 38,
              padding: '0 16px',
              flexShrink: 0,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              background: 'var(--dw-accent)',
              border: '1px solid var(--dw-accent)',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 14,
              fontWeight: 600,
              color: 'var(--dw-fg-invert)',
              cursor: 'pointer',
              outline: 'none',
              transition: 'background 120ms ease, border-color 120ms ease',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = '#ffffff';
              e.currentTarget.style.borderColor = '#ffffff';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = 'var(--dw-accent)';
              e.currentTarget.style.borderColor = 'var(--dw-accent)';
            }}
            onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            aria-label="Start a new reconstruction — go to upload"
          >
            <Upload size={15} strokeWidth={1.5} />
            New reconstruction
          </button>
        </section>

        {/* Session history list */}
        <RecentProjects onSelectProject={() => setView('app')} />
      </main>
    </div>
  );
}

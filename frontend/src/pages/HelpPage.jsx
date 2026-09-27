/**
 * DepthWizard — Help / About page
 *
 * Standalone documentation page (view = 'help'). Absorbs the former
 * Help modal: product overview, the two processing paths, camera
 * navigation controls, and a shortcut into the demo scene.
 *
 * DESIGN.md: Operate mode. Data face for control names, thin --dw-rim
 * borders, no decorative statistics. No fake accuracy claims (D10).
 */
import { Play } from 'lucide-react';
import Header from '../components/common/Header.jsx';
import { STEPS } from './Home.jsx';
import { useApp, AppState } from '../store/appStore.jsx';
import { useAuth } from '../store/authContext.jsx';

const CONTROLS = [
  { name: 'Orbit Mode:',    usage: 'Left-drag rotate · Right-drag pan · Scroll zoom' },
  { name: 'Walkthrough:',   usage: 'W/A/S/D fly · Space/Ctrl altitude · Shift boost · Mouse look' },
  { name: 'Top View:',      usage: 'Orthographic 2D/3D nadir view' },
  { name: 'Elevation Probe:', usage: 'Hover cursor over terrain to sample elevation' },
];

export default function HelpPage() {
  const { state, actions } = useApp();
  const { setView } = useAuth();

  const demoProject = state.recentScenes?.[0];

  const openDemo = () => {
    if (!demoProject) return;
    actions.resumeSession(demoProject, AppState.RESULTS_READY);
    setView('app');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: 24,
        width: '100%',
        maxWidth: 720,
        margin: '0 auto',
        padding: '36px 24px 64px',
      }}>

        {/* Page heading */}
        <section style={{ width: '100%' }} aria-labelledby="help-heading">
          <h1
            id="help-heading"
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
            Help &amp; About
          </h1>
          <p style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 14.5,
            color: 'var(--dw-fg-muted)',
            margin: '8px 0 0',
            lineHeight: 1.65,
          }}>
            DepthWizard transforms a single optical remote-sensing image into a fully
            interactive, measurable, and validatable 3D digital surface model (DSM).
          </p>
        </section>

        {/* How it works — pipeline steps */}
        <section
          aria-labelledby="help-pipeline-heading"
          style={{
            background: 'var(--dw-panel)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius)',
            padding: '18px 24px 20px',
            width: '100%',
          }}
        >
          <h2
            id="help-pipeline-heading"
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12,
              fontWeight: 600,
              letterSpacing: '0.07em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-muted)',
              margin: 0,
            }}
          >
            How it works
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 14 }}>
            {STEPS.map((step) => (
              <div key={step.id} style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                <span style={{
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 12.5,
                  fontWeight: 600,
                  letterSpacing: '0.06em',
                  color: 'var(--dw-fg)',
                  minWidth: 110,
                  flexShrink: 0,
                }}>
                  {step.label}
                </span>
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 13.5,
                  color: 'var(--dw-fg-muted)',
                  lineHeight: 1.5,
                }}>
                  {step.sub}
                </span>
              </div>
            ))}
          </div>

          {/* Two processing paths — capability truth from D10 */}
          <div style={{
            borderTop: '1px solid var(--dw-rim)',
            marginTop: 16,
            paddingTop: 14,
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}>
            <p style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              color: 'var(--dw-fg-muted)',
              margin: 0,
              lineHeight: 1.6,
            }}>
              A <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12.5, color: 'var(--dw-fg)' }}>GeoTIFF</span> with
              georeferencing produces an <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12.5, color: 'var(--dw-fg)' }}>Absolute DSM</span> —
              metric elevations, validatable against reference DEMs.
            </p>
            <p style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              color: 'var(--dw-fg-muted)',
              margin: 0,
              lineHeight: 1.6,
            }}>
              A plain <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12.5, color: 'var(--dw-fg)' }}>PNG / JPG</span> produces
              a <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12.5, color: 'var(--dw-fg)' }}>Relative DSM</span> in
              scene units — no metric claims.
            </p>
          </div>
        </section>

        {/* Navigation controls */}
        <section
          aria-labelledby="help-controls-heading"
          style={{
            background: 'var(--dw-panel)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius)',
            padding: '18px 24px 20px',
            width: '100%',
          }}
        >
          <h2
            id="help-controls-heading"
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12,
              fontWeight: 600,
              letterSpacing: '0.07em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-muted)',
              margin: 0,
            }}
          >
            Navigation controls
          </h2>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(120px, 150px) 1fr',
            gap: '8px 12px',
            fontFamily: 'var(--dw-font-data)',
            fontSize: 13,
            marginTop: 14,
          }}>
            {CONTROLS.map((c) => (
              <div key={c.name} style={{ display: 'contents' }}>
                <span style={{ color: 'var(--dw-fg)', fontWeight: 500 }}>{c.name}</span>
                <span style={{ color: 'var(--dw-fg-muted)' }}>{c.usage}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Demo shortcut */}
        {demoProject && (
          <section style={{ width: '100%' }} aria-label="Demo scene">
            <button
              onClick={openDemo}
              style={{
                height: 38,
                padding: '0 16px',
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
              aria-label="Open the demo scene results"
            >
              <Play size={14} strokeWidth={1.5} />
              Open the demo scene (Scene_042 · Absolute DSM)
            </button>
          </section>
        )}

        {/* Scope disclaimer */}
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 12.5,
          color: 'var(--dw-fg-ghost)',
          margin: 0,
          lineHeight: 1.6,
        }}>
          Intended for preliminary terrain assessment and geospatial reconnaissance
          support. Measured values are never presented as certified accuracy.
        </p>
      </main>
    </div>
  );
}

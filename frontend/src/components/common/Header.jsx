/**
 * DepthWizard — Header
 *
 * 48px instrument-panel bar.
 * - Wordmark left
 * - Nav links centre
 * - Backend status dot + version right
 *
 * Design tokens: DESIGN.md §Palette, §Component Character
 * Spec: §4 Home/Landing navigation structure
 */
import { useEffect, useState, useCallback } from 'react';
import { X, HelpCircle, Info, Keyboard } from 'lucide-react';
import { checkHealth } from '../../api/client.js';
import { useApp, AppState } from '../../store/appStore.jsx';
import RecentProjects from './RecentProjects.jsx';

/** Maps to DESIGN.md status dot semantics */
const STATUS = { checking: 'checking', online: 'online', offline: 'offline' };

function StatusDot({ status }) {
  const colors = {
    checking: 'var(--dw-fg-ghost)',
    online:   'var(--dw-confirm)',
    offline:  'var(--dw-fault)',
  };
  return (
    <span
      aria-hidden="true"
      style={{
        display: 'inline-block',
        width: 6,
        height: 6,
        borderRadius: '50%',
        background: colors[status],
        flexShrink: 0,
        ...(status === 'online' && {
          boxShadow: `0 0 0 2px rgba(34,197,94,0.2)`,
        }),
      }}
    />
  );
}

/** @param {{ label: string, active?: boolean, onClick: () => void, disabled?: boolean }} props */
function NavItem({ label, active, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: 'none',
        border: 'none',
        padding: '0 10px',
        height: 28,
        borderRadius: 'var(--dw-radius-sm)',
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 12,
        fontWeight: active ? 500 : 400,
        color: active ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        letterSpacing: '0em',
        transition: 'color 150ms ease, background 150ms ease',
        outline: 'none',
      }}
      onMouseEnter={e => { if (!disabled && !active) e.currentTarget.style.color = 'var(--dw-fg)'; }}
      onMouseLeave={e => { if (!active) e.currentTarget.style.color = active ? 'var(--dw-fg)' : 'var(--dw-fg-muted)'; }}
      onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
      onBlur={e => { e.currentTarget.style.outline = 'none'; }}
      aria-current={active ? 'page' : undefined}
    >
      {label}
    </button>
  );
}

export default function Header({ onNavigate }) {
  const { state, actions } = useApp();
  const [health, setHealth] = useState({ status: STATUS.checking, version: null });

  const probe = useCallback(async () => {
    const result = await checkHealth();
    setHealth({
      status: result ? STATUS.online : STATUS.offline,
      version: result?.version ?? null,
    });
  }, []);

  useEffect(() => {
    probe();
    const id = setInterval(probe, 30_000);
    return () => clearInterval(id);
  }, [probe]);

  const isHome      = state.status === AppState.NO_SCENE || state.status === AppState.SCENE_READY;
  const hasScene    = state.status !== AppState.NO_SCENE;
  const inWorkspace = state.status === AppState.TERRAIN_READY || state.status === AppState.ANALYSIS;

  const [recentOpen, setRecentOpen] = useState(false);
  const [aboutOpen, setAboutOpen]   = useState(false);

  return (
    <>
      <header
        role="banner"
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 100,
          height: 'var(--dw-header-h)',
          background: 'var(--dw-panel)',
          borderBottom: '1px solid var(--dw-rim)',
          display: 'flex',
          alignItems: 'center',
          padding: '0 16px',
          gap: 0,
          userSelect: 'none',
        }}
      >
        {/* Wordmark */}
        <button
          onClick={() => actions.reset()}
          aria-label="DepthWizard — go to home"
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: 0,
            outline: 'none',
          }}
          onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        >
          {/* Logo mark — a simple elevation indicator glyph */}
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <polyline
              points="2,14 6,7 10,10 14,4 16,4"
              stroke="var(--dw-accent)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <line x1="14" y1="2" x2="14" y2="6" stroke="var(--dw-accent)" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="12" y1="4" x2="16" y2="4" stroke="var(--dw-accent)" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          <span
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontWeight: 600,
              fontSize: 14,
              letterSpacing: '-0.02em',
              color: 'var(--dw-fg)',
            }}
          >
            DepthWizard
          </span>
        </button>

        {/* Centre nav */}
        <nav
          aria-label="Main navigation"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            marginLeft: 'auto',
            marginRight: 'auto',
          }}
        >
          <NavItem
            label="Home"
            active={isHome && !recentOpen && !aboutOpen}
            onClick={() => {
              setRecentOpen(false);
              setAboutOpen(false);
              actions.reset();
            }}
          />
          <NavItem
            label="New Reconstruction"
            active={false}
            onClick={() => {
              setRecentOpen(false);
              setAboutOpen(false);
              if (!isHome) actions.reset();
            }}
          />
          <NavItem
            label="Recent Projects"
            active={recentOpen}
            onClick={() => {
              setRecentOpen(v => !v);
              setAboutOpen(false);
              onNavigate?.('recent');
            }}
          />
          <NavItem
            label="Help / About"
            active={aboutOpen}
            onClick={() => {
              setAboutOpen(v => !v);
              setRecentOpen(false);
              onNavigate?.('help');
            }}
          />
        </nav>

        {/* Right: status */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            marginLeft: 'auto',
          }}
        >
          <StatusDot status={health.status} />
          <span
            style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 11,
              color: 'var(--dw-fg-ghost)',
              letterSpacing: '0.03em',
            }}
            aria-label={`Backend status: ${health.status}`}
          >
            {health.status === 'online'
              ? health.version ? `API ${health.version}` : 'API online'
              : health.status === 'offline'
              ? 'API offline'
              : '…'}
          </span>
        </div>
      </header>

      {/* Slide-out Recent Projects Drawer (§28) */}
      {recentOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Recent Projects"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(7,9,14,0.65)',
            zIndex: 150,
            display: 'flex',
            justifyContent: 'flex-end',
          }}
          onClick={() => setRecentOpen(false)}
        >
          <div
            style={{
              width: '100%',
              maxWidth: 420,
              height: '100%',
              background: 'var(--dw-panel)',
              borderLeft: '1px solid var(--dw-rim)',
              padding: '20px 16px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 16,
              boxSizing: 'border-box',
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: 8,
              borderBottom: '1px solid var(--dw-rim)',
            }}>
              <span style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 13,
                fontWeight: 600,
                color: 'var(--dw-fg)',
              }}>
                Session Persistence & Projects
              </span>
              <button
                onClick={() => setRecentOpen(false)}
                aria-label="Close recent projects"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--dw-fg-muted)',
                  cursor: 'pointer',
                  padding: 4,
                  display: 'flex',
                  alignItems: 'center',
                  outline: 'none',
                }}
              >
                <X size={15} strokeWidth={1.5} />
              </button>
            </div>

            <RecentProjects
              compact={false}
              onClose={() => setRecentOpen(false)}
            />
          </div>
        </div>
      )}

      {/* Help / About Modal */}
      {aboutOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Help and documentation"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(7,9,14,0.7)',
            zIndex: 150,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16,
          }}
          onClick={() => setAboutOpen(false)}
        >
          <div
            style={{
              width: '100%',
              maxWidth: 520,
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-md)',
              padding: 20,
              display: 'flex',
              flexDirection: 'column',
              gap: 14,
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: 8,
              borderBottom: '1px solid var(--dw-rim)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Info size={15} strokeWidth={1.5} color="var(--dw-accent)" />
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 13,
                  fontWeight: 600,
                  color: 'var(--dw-fg)',
                }}>
                  DepthWizard Terrain Intelligence
                </span>
              </div>
              <button
                onClick={() => setAboutOpen(false)}
                aria-label="Close help modal"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--dw-fg-muted)',
                  cursor: 'pointer',
                  padding: 4,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <X size={15} strokeWidth={1.5} />
              </button>
            </div>

            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12,
              color: 'var(--dw-fg-muted)',
              lineHeight: 1.6,
            }}>
              <p style={{ margin: 0 }}>
                DepthWizard transforms a single optical remote-sensing image into a fully interactive, measurable, and validatable 3D digital surface model (DSM).
              </p>

              <div style={{
                background: 'var(--dw-surface)',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                padding: '10px 12px',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}>
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 10,
                  fontWeight: 500,
                  color: 'var(--dw-fg)',
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                }}>
                  Navigation Controls
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '4px 8px', fontFamily: 'var(--dw-font-data)', fontSize: 11 }}>
                  <span style={{ color: 'var(--dw-accent)' }}>Orbit Mode:</span>
                  <span>Left-drag rotate · Right-drag pan · Scroll zoom</span>
                  <span style={{ color: 'var(--dw-accent)' }}>First Person:</span>
                  <span>W/A/S/D walk · Mouse look · Terrain-clamped</span>
                  <span style={{ color: 'var(--dw-accent)' }}>Top View:</span>
                  <span>Orthographic 2D/3D nadir view</span>
                  <span style={{ color: 'var(--dw-accent)' }}>Elevation Probe:</span>
                  <span>Hover cursor over terrain to sample elevation</span>
                </div>
              </div>

              <p style={{ margin: 0, fontSize: 11, color: 'var(--dw-fg-ghost)' }}>
                Intended for preliminary terrain assessment and geospatial reconnaissance support.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

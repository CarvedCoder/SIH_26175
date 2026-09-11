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
import { X, HelpCircle, Info, Keyboard, Menu, LogOut } from 'lucide-react';
import { checkHealth } from '../../api/client.js';
import { useApp, AppState } from '../../store/appStore.jsx';
import { useAuth } from '../../store/authContext.jsx';
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
        background: active ? 'var(--dw-surface)' : 'none',
        border: active ? '1px solid var(--dw-accent)' : '1px solid transparent',
        padding: '0 14px',
        height: 34,
        borderRadius: 'var(--dw-radius-sm)',
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 14,
        fontWeight: active ? 600 : 450,
        color: active ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        letterSpacing: '0em',
        transition: 'color 150ms ease, background 150ms ease, border-color 150ms ease',
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
  const auth = useAuth();
  const user = auth?.user;
  const logout = auth?.logout;
  const setView = auth?.setView;

  const getInitials = (name) => {
    if (!name) return 'U';
    return name
      .split(' ')
      .filter(Boolean)
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

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

  const [recentOpen, setRecentOpen]       = useState(false);
  const [aboutOpen, setAboutOpen]         = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

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
          display: 'grid',
          gridTemplateColumns: '1fr auto 1fr',
          alignItems: 'center',
          padding: '0 20px',
          userSelect: 'none',
        }}
      >
        {/* Wordmark (Column 1 — left aligned) */}
        <button
          onClick={() => {
            setMobileMenuOpen(false);
            actions.reset();
          }}
          aria-label="DepthWizard — go to home"
          style={{
            justifySelf: 'start',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: 0,
            outline: 'none',
          }}
          onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        >
          {/* Logo mark — elevation indicator glyph */}
          <svg width="22" height="22" viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <polyline
              points="2,14 6,7 10,10 14,4 16,4"
              stroke="var(--dw-accent)"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <line x1="14" y1="2" x2="14" y2="6" stroke="var(--dw-accent)" strokeWidth="1.8" strokeLinecap="round" />
            <line x1="12" y1="4" x2="16" y2="4" stroke="var(--dw-accent)" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
          <span
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontWeight: 600,
              fontSize: 16,
              letterSpacing: '-0.02em',
              color: 'var(--dw-fg)',
            }}
          >
            DepthWizard
          </span>
        </button>

        {/* Centre nav (Column 2 — mathematical center of header) */}
        <nav
          className="dw-desktop-only"
          aria-label="Main navigation"
          style={{
            justifySelf: 'center',
            alignItems: 'center',
            gap: 4,
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
          <NavItem
            label="Overview"
            active={false}
            onClick={() => {
              setRecentOpen(false);
              setAboutOpen(false);
              setView?.('landing');
            }}
          />
        </nav>

        {/* Right: status & mobile toggle (Column 3 — right aligned) */}
        <div
          style={{
            justifySelf: 'end',
            display: 'flex',
            alignItems: 'center',
            gap: 12,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <StatusDot status={health.status} />
            <span
              style={{
                fontFamily: 'var(--dw-font-data)',
                fontSize: 12.5,
                color: 'var(--dw-fg-muted)',
                letterSpacing: '0.02em',
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

          {/* User profile & sign out */}
          {user && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingLeft: 10, borderLeft: '1px solid var(--dw-rim)' }}>
              <div
                title={user.email || user.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 7,
                  padding: '3px 8px',
                  borderRadius: 'var(--dw-radius-sm)',
                  background: 'var(--dw-surface)',
                  border: '1px solid var(--dw-rim)',
                }}
              >
                <div style={{
                  width: 22,
                  height: 22,
                  borderRadius: '50%',
                  background: 'var(--dw-accent)',
                  color: '#fff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 11,
                  fontWeight: 600,
                }}>
                  {getInitials(user.name)}
                </div>
                <span className="dw-desktop-only" style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 13,
                  fontWeight: 500,
                  color: 'var(--dw-fg)',
                  maxWidth: 110,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {user.name}
                </span>
              </div>
              <button
                onClick={logout}
                title="Sign out & return to landing"
                aria-label="Sign out"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--dw-fg-muted)',
                  cursor: 'pointer',
                  padding: 6,
                  display: 'flex',
                  alignItems: 'center',
                  borderRadius: 'var(--dw-radius-sm)',
                  transition: 'color 150ms ease',
                }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--dw-fg)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--dw-fg-muted)'}
              >
                <LogOut size={16} strokeWidth={1.5} />
              </button>
            </div>
          )}

          {/* Mobile hamburger menu toggle */}
          <button
            className="dw-mobile-only"
            onClick={() => setMobileMenuOpen(v => !v)}
            aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={mobileMenuOpen}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--dw-fg)',
              cursor: 'pointer',
              padding: 6,
              outline: 'none',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {mobileMenuOpen ? <X size={22} strokeWidth={1.5} /> : <Menu size={22} strokeWidth={1.5} />}
          </button>
        </div>
      </header>

      {/* Mobile navigation drop-down */}
      {mobileMenuOpen && (
        <div
          className="dw-mobile-only"
          style={{
            position: 'fixed',
            top: 'var(--dw-header-h)',
            left: 0,
            right: 0,
            background: 'var(--dw-panel)',
            borderBottom: '1px solid var(--dw-rim)',
            zIndex: 99,
            flexDirection: 'column',
            padding: '12px 16px',
            gap: 8,
          }}
        >
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              actions.reset();
            }}
            style={{
              background: 'none',
              border: 'none',
              textAlign: 'left',
              padding: '10px 0',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 15,
              fontWeight: isHome ? 600 : 400,
              color: isHome ? 'var(--dw-accent)' : 'var(--dw-fg)',
              cursor: 'pointer',
            }}
          >
            Home
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              actions.reset();
            }}
            style={{
              background: 'none',
              border: 'none',
              textAlign: 'left',
              padding: '10px 0',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 15,
              color: 'var(--dw-fg-muted)',
              cursor: 'pointer',
            }}
          >
            New Reconstruction
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              setRecentOpen(true);
            }}
            style={{
              background: 'none',
              border: 'none',
              textAlign: 'left',
              padding: '10px 0',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 15,
              color: recentOpen ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
              cursor: 'pointer',
            }}
          >
            Recent Projects
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              setAboutOpen(true);
            }}
            style={{
              background: 'none',
              border: 'none',
              textAlign: 'left',
              padding: '10px 0',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 15,
              color: aboutOpen ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
              cursor: 'pointer',
            }}
          >
            Help / About
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              setView?.('landing');
            }}
            style={{
              background: 'none',
              border: 'none',
              textAlign: 'left',
              padding: '10px 0',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 15,
              color: 'var(--dw-fg-muted)',
              cursor: 'pointer',
            }}
          >
            Overview / Landing
          </button>
          {user && (
            <button
              onClick={() => {
                setMobileMenuOpen(false);
                logout?.();
              }}
              style={{
                background: 'none',
                border: 'none',
                textAlign: 'left',
                padding: '10px 0',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 15,
                color: 'var(--dw-fault)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <LogOut size={16} strokeWidth={1.5} />
              Sign Out ({user.name})
            </button>
          )}
        </div>
      )}

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
              maxWidth: 440,
              height: '100%',
              background: 'var(--dw-panel)',
              borderLeft: '1px solid var(--dw-rim)',
              padding: '24px 20px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 18,
              boxSizing: 'border-box',
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: 10,
              borderBottom: '1px solid var(--dw-rim)',
            }}>
              <span style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 15,
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
                  padding: 6,
                  display: 'flex',
                  alignItems: 'center',
                  outline: 'none',
                }}
              >
                <X size={18} strokeWidth={1.5} />
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
              maxWidth: 540,
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-md)',
              padding: 24,
              display: 'flex',
              flexDirection: 'column',
              gap: 16,
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: 10,
              borderBottom: '1px solid var(--dw-rim)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Info size={18} strokeWidth={1.5} color="var(--dw-accent)" />
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 15,
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
                  padding: 6,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <X size={18} strokeWidth={1.5} />
              </button>
            </div>

            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 14,
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 14,
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
                padding: '12px 14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
              }}>
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 12,
                  fontWeight: 600,
                  color: 'var(--dw-fg)',
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                }}>
                  Navigation Controls
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: '130px 1fr', gap: '6px 10px', fontFamily: 'var(--dw-font-data)', fontSize: 13 }}>
                  <span style={{ color: 'var(--dw-accent)' }}>Orbit Mode:</span>
                  <span>Left-drag rotate · Right-drag pan · Scroll zoom</span>
                  <span style={{ color: 'var(--dw-accent)' }}>Walkthrough:</span>
                  <span>W/A/S/D fly · Space/Ctrl altitude · Shift boost · Mouse look</span>
                  <span style={{ color: 'var(--dw-accent)' }}>Top View:</span>
                  <span>Orthographic 2D/3D nadir view</span>
                  <span style={{ color: 'var(--dw-accent)' }}>Elevation Probe:</span>
                  <span>Hover cursor over terrain to sample elevation</span>
                </div>
              </div>

              <p style={{ margin: 0, fontSize: 12.5, color: 'var(--dw-fg-ghost)' }}>
                Intended for preliminary terrain assessment and geospatial reconnaissance support.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

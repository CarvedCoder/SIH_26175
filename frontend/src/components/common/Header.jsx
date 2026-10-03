/**
 * DepthWizard — Header
 *
 * 56px instrument-panel bar.
 * - Wordmark left
 * - Page nav centre (Home / Recent Projects / Help-About / Overview)
 * - Backend status dot + version + user chip right
 *
 * Navigation is view-based (authContext view machine): Home returns to the
 * workspace workflow, Recent Projects and Help / About are standalone pages.
 *
 * Design tokens: DESIGN.md §Palette, §Component Character
 * Spec: §4 Home/Landing navigation structure
 */
import { useEffect, useState, useCallback } from 'react';
import { X, Menu, LogOut } from 'lucide-react';
import { checkHealth } from '../../api/client.js';
import { useApp, AppState } from '../../store/appStore.jsx';
import { useAuth } from '../../store/authContext.jsx';

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
          boxShadow: `0 0 0 2px rgba(74,222,128,0.2)`,
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
        background: active ? 'var(--dw-accent-soft)' : 'none',
        border: active ? '1px solid var(--dw-accent-dim)' : '1px solid transparent',
        padding: '0 14px',
        height: 34,
        borderRadius: 'var(--dw-radius-sm)',
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 14,
        fontWeight: active ? 600 : 450,
        color: active ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        letterSpacing: '0em',
        transition: 'color 150ms ease, background 150ms ease, border-color 150ms ease',
        outline: 'none',
      }}
      onMouseEnter={e => { if (!disabled && !active) e.currentTarget.style.color = 'var(--dw-fg)'; }}
      onMouseLeave={e => { if (!active) e.currentTarget.style.color = 'var(--dw-fg-muted)'; }}
      onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
      onBlur={e => { e.currentTarget.style.outline = 'none'; }}
      aria-current={active ? 'page' : undefined}
    >
      {label}
    </button>
  );
}

export default function Header() {
  const { state, actions } = useApp();
  const auth = useAuth();
  const user = auth?.user;
  const logout = auth?.logout;
  const setView = auth?.setView;
  const view = auth?.view;

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
    let cancelled = false;
    let timer = null;
    // Probe immediately, then poll. While the backend is still starting up
    // (or unreachable) retry every 3 s so the indicator flips to online as
    // soon as the API answers instead of sitting on "offline" for 30 s.
    const run = async () => {
      if (cancelled) return;
      const result = await checkHealth();
      if (cancelled) return;
      setHealth({
        status: result ? STATUS.online : STATUS.offline,
        version: result?.version ?? null,
      });
      timer = setTimeout(run, result ? 30_000 : 3_000);
    };
    run();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [probe]);

  const isHome      = view === 'app'
    && (state.status === AppState.NO_SCENE || state.status === AppState.SCENE_READY);
  const inRecent    = view === 'recent';
  const inHelp      = view === 'help';

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const goHome = () => {
    setMobileMenuOpen(false);
    actions.reset();
    setView?.('app');
  };

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
          onClick={goHome}
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
            active={isHome}
            onClick={goHome}
          />
          <NavItem
            label="Recent Projects"
            active={inRecent}
            onClick={() => setView?.('recent')}
          />
          <NavItem
            label="Help / About"
            active={inHelp}
            onClick={() => setView?.('help')}
          />
          <NavItem
            label="Overview"
            active={false}
            onClick={() => setView?.('landing')}
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
                  color: 'var(--dw-fg-invert)',
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
            onClick={goHome}
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
              setView?.('recent');
            }}
            style={{
              background: 'none',
              border: 'none',
              textAlign: 'left',
              padding: '10px 0',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 15,
              color: inRecent ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
              cursor: 'pointer',
            }}
          >
            Recent Projects
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              setView?.('help');
            }}
            style={{
              background: 'none',
              border: 'none',
              textAlign: 'left',
              padding: '10px 0',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 15,
              color: inHelp ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
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
    </>
  );
}

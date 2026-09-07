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
import { checkHealth } from '../../api/client.js';
import { useApp, AppState } from '../../store/appStore.jsx';

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

  return (
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
          active={isHome}
          onClick={() => actions.reset()}
        />
        <NavItem
          label="New Reconstruction"
          active={false}
          onClick={() => { if (!isHome) actions.reset(); }}
        />
        <NavItem
          label="Recent Projects"
          active={false}
          onClick={() => onNavigate?.('recent')}
        />
        <NavItem
          label="Help / About"
          active={false}
          onClick={() => onNavigate?.('help')}
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
  );
}

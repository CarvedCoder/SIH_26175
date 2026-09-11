/**
 * DepthWizard — ControlsHint (3D Viewport Controls Overlay)
 *
 * Sleek, non-intrusive instrument-panel overlay showing live controls
 * based on current camera mode:
 *   - Orbit: Left Drag (Rotate), Right Drag / Shift (Pan), Scroll (Zoom)
 *   - Top-view: Right Drag (Pan), Scroll (Zoom)
 *   - Walkthrough (internal mode id 'first-person'): Click (Lock Cursor),
 *     WASD (Move), Space / Ctrl (Altitude), Shift (Boost), Mouse (Look), Esc (Unlock)
 *
 * Design matches DESIGN.md:
 *   - Dark translucent background (rgba(13, 17, 23, 0.92)) with 1px var(--dw-rim)
 *   - Geist monospace for keycaps (<kbd>) and UI sans for descriptions
 *   - Collapsible with persistent preference in localStorage
 */

import { useState } from 'react';
import { Keyboard, ChevronDown, ChevronUp, Move3d, Compass, Eye } from 'lucide-react';

const STORAGE_KEY = 'dw_controls_hint_collapsed';

/**
 * @param {{
 *   cameraMode: 'orbit'|'first-person'|'top',
 * }} props
 */
export default function ControlsHint({ cameraMode = 'orbit' }) {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === 'true';
    } catch {
      return false;
    }
  });

  const toggleCollapsed = () => {
    setCollapsed(prev => {
      const next = !prev;
      try {
        localStorage.setItem(STORAGE_KEY, String(next));
      } catch {
        // Ignore storage errors
      }
      return next;
    });
  };

  const modeConfig = {
    orbit: {
      label: 'Orbit Mode',
      icon: Compass,
      controls: [
        { key: 'Left Drag', label: 'Rotate' },
        { key: 'Right Drag', label: 'Pan' },
        { key: 'Scroll', label: 'Zoom' },
      ],
    },
    top: {
      label: 'Top View',
      icon: Eye,
      controls: [
        { key: 'Right Drag', label: 'Pan' },
        { key: 'Scroll', label: 'Zoom' },
      ],
    },
    'first-person': {
      label: 'Walkthrough',
      icon: Move3d,
      controls: [
        { key: 'Click Canvas', label: 'Lock Cursor' },
        { key: 'W A S D', label: 'Move' },
        { key: 'Space', label: 'Rise' },
        { key: 'Ctrl / C', label: 'Descend' },
        { key: 'Shift', label: 'Boost' },
        { key: 'Mouse', label: 'Look' },
        { key: 'Esc', label: 'Release' },
      ],
    },
  }[cameraMode] || {
    label: 'Camera',
    icon: Compass,
    controls: [
      { key: 'Left Drag', label: 'Rotate' },
      { key: 'Right Drag', label: 'Pan' },
      { key: 'Scroll', label: 'Zoom' },
    ],
  };

  const ModeIcon = modeConfig.icon;

  return (
    <aside
      aria-label="3D Viewport Controls Guide"
      style={{
        position: 'absolute',
        bottom: 60,
        left: 16,
        zIndex: 11,
        maxWidth: 'calc(100vw - 32px)',
        pointerEvents: 'auto',
      }}
    >
      <div
        style={{
          background: 'rgba(13, 17, 23, 0.90)',
          backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)',
          border: '1px solid var(--dw-rim, #222c3e)',
          borderRadius: 6,
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
          overflow: 'hidden',
          transition: 'all 200ms ease-out',
        }}
      >
        {/* Header / Toggle bar */}
        <div
          onClick={toggleCollapsed}
          role="button"
          tabIndex={0}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') toggleCollapsed(); }}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 10,
            padding: '6px 10px',
            cursor: 'pointer',
            userSelect: 'none',
            borderBottom: collapsed ? 'none' : '1px solid var(--dw-rim, #222c3e)',
            background: 'rgba(255, 255, 255, 0.02)',
          }}
          title={collapsed ? 'Expand Controls' : 'Collapse Controls'}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Keyboard size={13} color="var(--dw-accent, #3b82f6)" />
            <span
              style={{
                fontFamily: 'var(--dw-font-ui, sans-serif)',
                fontSize: 11,
                fontWeight: 600,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                color: 'var(--dw-fg, #e2e8f0)',
              }}
            >
              Controls
            </span>
            <span
              style={{
                fontSize: 10,
                fontFamily: 'var(--dw-font-data, monospace)',
                color: 'var(--dw-accent, #3b82f6)',
                background: 'rgba(59, 130, 246, 0.12)',
                padding: '1px 6px',
                borderRadius: 3,
                letterSpacing: '0.04em',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <ModeIcon size={10} />
              {modeConfig.label}
            </span>
          </div>

          <button
            type="button"
            aria-label={collapsed ? 'Expand controls panel' : 'Collapse controls panel'}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--dw-fg-ghost, #94a3b8)',
              display: 'flex',
              alignItems: 'center',
              cursor: 'pointer',
              padding: 0,
            }}
          >
            {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {/* Expanded Controls List */}
        {!collapsed && (
          <div
            style={{
              padding: '8px 12px',
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              gap: 12,
            }}
          >
            {modeConfig.controls.map((item, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <kbd
                  style={{
                    fontFamily: 'var(--dw-font-data, monospace)',
                    fontSize: 11,
                    fontWeight: 600,
                    color: '#f8fafc',
                    background: 'rgba(30, 41, 59, 0.85)',
                    border: '1px solid rgba(148, 163, 184, 0.25)',
                    borderRadius: 3,
                    padding: '2px 6px',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.3)',
                    lineHeight: 1.2,
                  }}
                >
                  {item.key}
                </kbd>
                <span
                  style={{
                    fontFamily: 'var(--dw-font-ui, sans-serif)',
                    fontSize: 11,
                    color: 'var(--dw-fg-muted, #94a3b8)',
                    letterSpacing: '0.02em',
                  }}
                >
                  {item.label}
                </span>
                {idx < modeConfig.controls.length - 1 && (
                  <span
                    style={{
                      color: 'rgba(148, 163, 184, 0.3)',
                      fontSize: 10,
                      marginLeft: 4,
                    }}
                  >
                    •
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}

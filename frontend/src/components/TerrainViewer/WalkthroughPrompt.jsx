/**
 * DepthWizard — WalkthroughPrompt
 *
 * Small, unobtrusive instrument-panel cue shown while Walkthrough mode is
 * active but the cursor has not been captured yet. Disappears entirely once
 * the pointer is locked so it never obstructs the terrain; reappears after
 * Escape releases the lock.
 *
 * Matches the ControlsHint / CameraHUD panel idiom:
 *   - rgba(13,17,23,0.9) panel, 1px var(--dw-rim), blur backdrop
 *   - Data face for the key summary, UI face for the instruction
 */

import { useEffect, useState } from 'react';
import { MousePointerClick } from 'lucide-react';

/**
 * @param {{
 *   cameraMode: 'orbit'|'first-person'|'top',
 * }} props
 */
export default function WalkthroughPrompt({ cameraMode = 'orbit' }) {
  const [locked, setLocked] = useState(() => !!document.pointerLockElement);

  useEffect(() => {
    function onPointerLockChange() {
      setLocked(!!document.pointerLockElement);
    }
    document.addEventListener('pointerlockchange', onPointerLockChange);
    return () => {
      document.removeEventListener('pointerlockchange', onPointerLockChange);
    };
  }, []);

  if (cameraMode !== 'first-person' || locked) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      aria-label="Walkthrough mode ready — click the terrain to capture the cursor"
      style={{
        position: 'absolute',
        top: 14,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 11,
        maxWidth: 'calc(100vw - 32px)',
        pointerEvents: 'none',
      }}
    >
      <div style={{
        background: 'rgba(13, 17, 23, 0.90)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        border: '1px solid var(--dw-rim, #222c3e)',
        borderRadius: 6,
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
        padding: '8px 14px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 5,
      }}>
        <span style={{
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          fontFamily: 'var(--dw-font-ui, sans-serif)',
          fontSize: 11.5,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg, #e2e8f0)',
        }}>
          <MousePointerClick size={13} color="var(--dw-accent, #3b82f6)" aria-hidden="true" />
          Walkthrough
          <span style={{ fontWeight: 500, textTransform: 'none', letterSpacing: '0.02em', color: 'var(--dw-fg-muted, #94a3b8)' }}>
            Click the terrain to capture the cursor
          </span>
        </span>
        <span style={{
          fontFamily: 'var(--dw-font-data, monospace)',
          fontSize: 10.5,
          color: 'var(--dw-fg-ghost, #94a3b8)',
          letterSpacing: '0.04em',
          whiteSpace: 'nowrap',
        }}>
          W A S D move · Space / Ctrl altitude · Shift boost · Esc release
        </span>
      </div>
    </div>
  );
}

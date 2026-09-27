/**
 * DepthWizard — Joystick (touch walkthrough control)
 *
 * Google-Maps-style on-screen movement pad for the Walkthrough mode. Two
 * parts, both pointer-event based (works for touch AND mouse-drag):
 *
 *   1. Circular pad  — drag from centre to move; the offset vector is fed
 *      to the camera controller as (x: right+, y: forward+) in [-1, 1].
 *   2. Ascend/descend buttons — secondary control for vertical movement
 *      (the walkthrough keeps Space/Ctrl on desktop; touch users get these).
 *
 * Integration (no parallel camera system): the pad feeds
 * useCameraController's setJoystickInput/setJoystickVertical, which
 * tickWalkthrough merges with keyboard input every frame — identical
 * movement code path for both input methods.
 *
 * DESIGN.md — dark instrument-panel styling; no neon.
 */

import { useCallback, useRef, useState } from 'react';

/** Pad radius in px (the drag range maps to [-1, 1]). */
const PAD_RADIUS = 56;
/** Dead zone fraction — tiny accidental offsets don't creep the camera. */
const DEAD_ZONE = 0.12;

export default function Joystick({ onMove, onVertical }) {
  const padRef = useRef(null);
  const pointerId = useRef(null);
  const [knob, setKnob] = useState({ x: 0, y: 0 });
  const [vertical, setVertical] = useState(0);

  const handlePointerDown = useCallback((e) => {
    e.preventDefault();
    pointerId.current = e.pointerId;
    padRef.current?.setPointerCapture?.(e.pointerId);
    updateFromPointer(e);
  }, []);

  const updateFromPointer = useCallback((e) => {
    const pad = padRef.current;
    if (!pad || pointerId.current !== e.pointerId) return;
    const rect = pad.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    let dx = (e.clientX - cx) / PAD_RADIUS;
    let dy = (e.clientY - cy) / PAD_RADIUS;
    const len = Math.hypot(dx, dy);
    if (len > 1) { dx /= len; dy /= len; }

    // Dead zone → 0; beyond it, rescale so the ramp starts at 0
    const mag = Math.hypot(dx, dy);
    let x = 0, y = 0;
    if (mag > DEAD_ZONE) {
      const scaled = (mag - DEAD_ZONE) / (1 - DEAD_ZONE);
      x = (dx / mag) * scaled;
      y = (dy / mag) * scaled;
    }

    setKnob({ x: dx * PAD_RADIUS, y: dy * PAD_RADIUS });
    // Screen coords: y grows downward → forward is NEGATIVE dy
    onMove?.(x, -y);
  }, [onMove]);

  const releasePointer = useCallback((e) => {
    if (pointerId.current !== e.pointerId) return;
    pointerId.current = null;
    setKnob({ x: 0, y: 0 });
    onMove?.(0, 0);
  }, [onMove]);

  const pressVertical = useCallback((dir) => {
    setVertical(dir);
    onVertical?.(dir);
  }, [onVertical]);

  const controlStyle = {
    touchAction: 'none',
    userSelect: 'none',
    WebkitUserSelect: 'none',
  };

  return (
    <div
      style={{
        position: 'absolute',
        left: 16,
        bottom: 96,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 8,
        zIndex: 14,
      }}
      aria-label="Walkthrough movement joystick"
    >
      {/* Ascend / descend — secondary vertical control (touch) */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          type="button"
          aria-label="Ascend"
          onPointerDown={(e) => { e.preventDefault(); pressVertical(1); }}
          onPointerUp={() => pressVertical(0)}
          onPointerLeave={() => vertical === 1 && pressVertical(0)}
          onPointerCancel={() => pressVertical(0)}
          style={{
            width: 44, height: 36, borderRadius: 8,
            background: vertical === 1 ? 'rgba(245, 158, 11, 0.25)' : 'rgba(24, 24, 27, 0.75)',
            border: '1px solid rgba(161, 161, 170, 0.35)',
            color: '#d4d4d8', fontSize: 16, cursor: 'pointer',
            ...controlStyle,
          }}
        >▲</button>
        <button
          type="button"
          aria-label="Descend"
          onPointerDown={(e) => { e.preventDefault(); pressVertical(-1); }}
          onPointerUp={() => pressVertical(0)}
          onPointerLeave={() => vertical === -1 && pressVertical(0)}
          onPointerCancel={() => pressVertical(0)}
          style={{
            width: 44, height: 36, borderRadius: 8,
            background: vertical === -1 ? 'rgba(245, 158, 11, 0.25)' : 'rgba(24, 24, 27, 0.75)',
            border: '1px solid rgba(161, 161, 170, 0.35)',
            color: '#d4d4d8', fontSize: 16, cursor: 'pointer',
            ...controlStyle,
          }}
        >▼</button>
      </div>

      {/* Movement pad */}
      <div
        ref={padRef}
        role="application"
        aria-label="Movement pad"
        onPointerDown={handlePointerDown}
        onPointerMove={updateFromPointer}
        onPointerUp={releasePointer}
        onPointerCancel={releasePointer}
        style={{
          width: PAD_RADIUS * 2 + 24,
          height: PAD_RADIUS * 2 + 24,
          borderRadius: '50%',
          background: 'rgba(24, 24, 27, 0.75)',
          border: '1px solid rgba(161, 161, 170, 0.35)',
          position: 'relative',
          ...controlStyle,
        }}
      >
        {/* cross-hair guides */}
        <div style={{
          position: 'absolute', left: '50%', top: 8, bottom: 8, width: 1,
          background: 'rgba(161, 161, 170, 0.2)', transform: 'translateX(-0.5px)',
        }} />
        <div style={{
          position: 'absolute', top: '50%', left: 8, right: 8, height: 1,
          background: 'rgba(161, 161, 170, 0.2)', transform: 'translateY(-0.5px)',
        }} />
        {/* knob */}
        <div style={{
          position: 'absolute',
          left: `calc(50% + ${knob.x}px)`,
          top: `calc(50% + ${knob.y}px)`,
          width: 44, height: 44,
          transform: 'translate(-50%, -50%)',
          borderRadius: '50%',
          background: 'rgba(245, 158, 11, 0.85)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.45)',
          pointerEvents: 'none',
        }} />
      </div>
    </div>
  );
}

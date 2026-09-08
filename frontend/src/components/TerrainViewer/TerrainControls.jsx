/**
 * DepthWizard — TerrainControls (Phase 4, tasks 4.5 + 4.6)
 *
 * Compact control strip for the terrain viewer:
 *   - Terrain exaggeration slider 1×–5× (changes uExaggeration uniform only;
 *     measured elevation values remain unchanged — label mandated by §D06)
 *   - Wireframe toggle
 *
 * Calls imperative methods on the TerrainCanvas ref.
 *
 * DESIGN.md: Operate mode.
 * - Slider: 2px track, --dw-rim unfilled, --dw-accent filled, 12px thumb
 * - Toggle: same button pattern as Processing page cancel/confirm buttons
 * - No card chrome, no shadows
 */
import { useState } from 'react';
import { Grid3x3 } from 'lucide-react';

/**
 * @param {{
 *   terrainRef: React.RefObject,
 *   disabled?: boolean,
 * }} props
 */
export default function TerrainControls({ terrainRef, disabled = false }) {
  const [exaggeration, setExaggeration] = useState(1.5);
  const [wireframe, setWireframe]       = useState(false);

  function handleExaggeration(e) {
    const v = parseFloat(e.target.value);
    setExaggeration(v);
    terrainRef.current?.setExaggeration(v);
  }

  function handleWireframe() {
    const next = !wireframe;
    setWireframe(next);
    terrainRef.current?.setWireframe(next);
  }

  const opacity = disabled ? 0.4 : 1;

  return (
    <div
      aria-label="Terrain viewer controls"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 20,
        padding: '0 16px',
        height: '100%',
        opacity,
        pointerEvents: disabled ? 'none' : 'auto',
      }}
    >
      {/* Exaggeration slider */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 160 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            color: 'var(--dw-fg-muted)',
            letterSpacing: '0.02em',
          }}>
            Exaggeration
          </span>
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 12,
            color: 'var(--dw-fg)',
          }}>
            {exaggeration.toFixed(1)}×
          </span>
        </div>

        {/* Custom-styled range input */}
        <div style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center' }}>
          {/* Track background */}
          <div style={{
            position: 'absolute',
            left: 0, right: 0,
            height: 2,
            background: 'var(--dw-rim)',
            borderRadius: 1,
            pointerEvents: 'none',
          }} />
          {/* Filled track */}
          <div style={{
            position: 'absolute',
            left: 0,
            width: `${((exaggeration - 1) / 4) * 100}%`,
            height: 2,
            background: 'var(--dw-accent)',
            borderRadius: 1,
            pointerEvents: 'none',
          }} />
          <input
            type="range"
            min={1}
            max={5}
            step={0.1}
            value={exaggeration}
            onChange={handleExaggeration}
            aria-label="Terrain vertical exaggeration"
            aria-valuetext={`${exaggeration.toFixed(1)} times`}
            style={{
              // Reset browser range input
              position: 'relative',
              width: '100%',
              height: 20,
              margin: 0,
              cursor: 'pointer',
              appearance: 'none',
              WebkitAppearance: 'none',
              background: 'transparent',
              outline: 'none',
            }}
          />
        </div>

        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 10,
          color: 'var(--dw-fg-ghost)',
          margin: 0,
          letterSpacing: '0.01em',
        }}>
          Visual only — elevation values unchanged
        </p>
      </div>

      {/* Divider */}
      <div style={{ width: 1, height: 32, background: 'var(--dw-rim)', flexShrink: 0 }} />

      {/* Wireframe toggle */}
      <button
        onClick={handleWireframe}
        aria-pressed={wireframe}
        aria-label="Toggle wireframe"
        title="Wireframe"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          height: 32,
          padding: '0 10px',
          background: wireframe ? 'var(--dw-surface)' : 'none',
          border: wireframe ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 12,
          color: wireframe ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
          cursor: 'pointer',
          outline: 'none',
          transition: 'border-color 120ms ease, color 120ms ease, background 120ms ease',
        }}
        onFocus={e => {
          e.currentTarget.style.outline = '2px solid var(--dw-accent)';
          e.currentTarget.style.outlineOffset = '2px';
        }}
        onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        onMouseEnter={e => {
          if (!wireframe) e.currentTarget.style.borderColor = 'var(--dw-fg-ghost)';
        }}
        onMouseLeave={e => {
          if (!wireframe) e.currentTarget.style.borderColor = 'var(--dw-rim)';
        }}
      >
        <Grid3x3 size={14} strokeWidth={1.5} aria-hidden="true" />
        Wireframe
      </button>
    </div>
  );
}

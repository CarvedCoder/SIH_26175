/**
 * DepthWizard — TerrainControls (Phase 4, tasks 4.5 + 4.6, Phase 8 task 8.4)
 *
 * Compact control strip for the terrain viewer:
 *   - Terrain exaggeration slider 1×–5× (changes uExaggeration uniform only;
 *     measured elevation values remain unchanged — label mandated by §D06)
 *   - Wireframe toggle
 *   - Contour lines toggle + interval input (metric 'm' when absolute, 'scene-units' otherwise — §21)
 *
 * Calls imperative methods on the TerrainCanvas ref.
 *
 * DESIGN.md: Operate mode.
 * - Slider: 2px track, --dw-rim unfilled, --dw-accent filled, 12px thumb
 * - Toggle: same button pattern as Processing page cancel/confirm buttons
 * - No card chrome, no shadows
 */
import { useState } from 'react';
import { Grid3x3, Spline } from 'lucide-react';
import { useApp } from '../../store/appStore.jsx';

/**
 * @param {{
 *   terrainRef: React.RefObject,
 *   disabled?: boolean,
 * }} props
 */
export default function TerrainControls({ terrainRef, disabled = false }) {
  const { state } = useApp();
  const [exaggeration, setExaggeration] = useState(2.5);
  const [wireframe, setWireframe]       = useState(true);
  const [contours, setContours]         = useState(false);
  const [interval, setIntervalVal]      = useState(5);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel = 'm'; // world scale is metres (1 m/pixel documented fallback)

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

  function handleToggleContours() {
    const next = !contours;
    setContours(next);
    terrainRef.current?.setContours(next, interval);
  }

  function handleIntervalChange(e) {
    const val = parseFloat(e.target.value);
    const safeVal = isNaN(val) || val <= 0 ? 1 : val;
    setIntervalVal(safeVal);
    if (contours) {
      terrainRef.current?.setContours(true, safeVal);
    }
  }

  const opacity = disabled ? 0.4 : 1;

  return (
    <div
      aria-label="Terrain viewer controls"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        padding: '0 16px',
        height: '100%',
        opacity,
        pointerEvents: disabled ? 'none' : 'auto',
      }}
    >
      {/* Exaggeration slider */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 150 }}>
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
          whiteSpace: 'nowrap',
        }}>
          Visual only — elevation unchanged
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
          whiteSpace: 'nowrap',
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

      {/* Divider */}
      <div style={{ width: 1, height: 32, background: 'var(--dw-rim)', flexShrink: 0 }} />

      {/* Contour lines toggle + interval input (task 8.4, §21) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          onClick={handleToggleContours}
          aria-pressed={contours}
          aria-label="Toggle contour lines"
          title="Contour lines"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            height: 32,
            padding: '0 10px',
            background: contours ? 'var(--dw-surface)' : 'none',
            border: contours ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 12,
            color: contours ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
            cursor: 'pointer',
            outline: 'none',
            transition: 'border-color 120ms ease, color 120ms ease, background 120ms ease',
            whiteSpace: 'nowrap',
          }}
          onFocus={e => {
            e.currentTarget.style.outline = '2px solid var(--dw-accent)';
            e.currentTarget.style.outlineOffset = '2px';
          }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          onMouseEnter={e => {
            if (!contours) e.currentTarget.style.borderColor = 'var(--dw-fg-ghost)';
          }}
          onMouseLeave={e => {
            if (!contours) e.currentTarget.style.borderColor = 'var(--dw-rim)';
          }}
        >
          <Spline size={14} strokeWidth={1.5} aria-hidden="true" />
          Contours
        </button>

        {contours && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <label
              htmlFor="contour-interval"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 11,
                color: 'var(--dw-fg-muted)',
                whiteSpace: 'nowrap',
              }}
            >
              Interval:
            </label>
            <input
              id="contour-interval"
              type="number"
              min={1}
              max={500}
              step={1}
              value={interval}
              onChange={handleIntervalChange}
              aria-label={`Contour interval in ${unitLabel}`}
              style={{
                width: 46,
                height: 26,
                background: 'var(--dw-surface)',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                padding: '0 4px',
                fontFamily: 'var(--dw-font-data)',
                fontSize: 11,
                color: 'var(--dw-fg)',
                outline: 'none',
                textAlign: 'right',
              }}
              onFocus={e => {
                e.currentTarget.style.borderColor = 'var(--dw-accent)';
              }}
              onBlur={e => {
                e.currentTarget.style.borderColor = 'var(--dw-rim)';
              }}
            />
            <span style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 10,
              color: 'var(--dw-fg-ghost)',
              whiteSpace: 'nowrap',
            }}>
              {unitLabel}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

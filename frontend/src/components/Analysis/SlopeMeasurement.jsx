/**
 * DepthWizard — SlopeMeasurement (Phase 9, Task 9.4)
 *
 * Two-point slope calculation tool (§13).
 * Workflow:
 *   1. User clicks Point A
 *   2. User clicks Point B
 *   3. Calculates Elevation Difference, Horizontal Distance, and Slope (degrees)
 *
 * Displays:
 *   SLOPE ANALYSIS
 *   Elevation Difference:  8.6 m
 *   Horizontal Distance:  42.1 m
 *   Slope:                11.5°
 *
 * DESIGN.md:
 *   - Two-column table layout
 *   - Monospace data face for metrics
 *   - Units labelled correctly per D10
 *
 * Spec §13, §D10.
 */
import { useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { useApp } from '../../store/appStore.jsx';
import { TrendingUp, RotateCcw } from 'lucide-react';

const SlopeMeasurement = forwardRef(function SlopeMeasurement(
  { active = false, onActiveChange },
  ref
) {
  const { state } = useApp();
  const [step, setStep] = useState(0); // 0: need A, 1: need B, 2: completed
  const [pointA, setPointA] = useState(null);
  const [pointB, setPointB] = useState(null);
  const [result, setResult] = useState(null);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel = 'm'; // world scale is metres (1 m/pixel documented fallback)

  const clearMeasurement = useCallback(() => {
    setStep(0);
    setPointA(null);
    setPointB(null);
    setResult(null);
  }, []);

  const handleSelectPoint = useCallback((point) => {
    if (!active) return;

    if (step === 0) {
      setPointA(point);
      setStep(1);
    } else if (step === 1) {
      setPointB(point);
      setStep(2);

      // Terrain clicks give WORLD coordinates — already metres on the
      // physical-scale path (mesh spans world_width_m × world_depth_m),
      // scene units on the legacy fallback. Elevation is metric when absolute.
      const dx = point.x - pointA.x;
      const dz = point.z - pointA.z;
      const horizDist = Math.hypot(dx, dz);
      const elevDiff = Math.abs(point.elevation - pointA.elevation);

      const slopeDegrees = horizDist > 0.0001
        ? (Math.atan(elevDiff / horizDist) * 180.0) / Math.PI
        : 0;

      setResult({
        elevationDiff: elevDiff,
        horizontal: horizDist,
        slope: slopeDegrees,
      });
    }
  }, [active, step, pointA, state.terrain, isAbsolute]);

  useImperativeHandle(ref, () => ({
    handleSelectPoint,
    clear: clearMeasurement,
  }));

  if (!active && !result) return null;

  return (
    <div
      aria-label="Slope analysis results"
      style={{
        background: 'var(--dw-panel)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        padding: '14px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        minWidth: 260,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <TrendingUp size={15} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11.5,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
          }}>
            SLOPE ANALYSIS
          </span>
        </div>

        {result && (
          <button
            onClick={clearMeasurement}
            aria-label="Clear measurement"
            title="Clear measurement"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              height: 28,
              padding: '0 8px',
              background: 'none',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12,
              color: 'var(--dw-fg-muted)',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            <RotateCcw size={12} strokeWidth={1.5} aria-hidden="true" />
            Clear
          </button>
        )}
      </div>

      {step === 0 && !result && (
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          color: 'var(--dw-fg-muted)',
          margin: 0,
        }}>
          Click base / start point…
        </p>
      )}

      {step === 1 && !result && (
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          color: 'var(--dw-accent)',
          margin: 0,
        }}>
          Click crest / end point…
        </p>
      )}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Elevation Difference
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {result.elevationDiff.toFixed(1)} {unitLabel}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Horizontal Distance
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {result.horizontal.toFixed(1)} {unitLabel}
            </span>
          </div>

          <div style={{ width: '100%', height: 1, background: 'var(--dw-rim)', margin: '2px 0' }} />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13.5, fontWeight: 600, color: 'var(--dw-fg)' }}>
              Slope
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 16,
              fontWeight: 600,
              color: 'var(--dw-accent)',
            }}>
              {result.slope.toFixed(1)}°
            </span>
          </div>
        </div>
      )}
    </div>
  );
});

export default SlopeMeasurement;

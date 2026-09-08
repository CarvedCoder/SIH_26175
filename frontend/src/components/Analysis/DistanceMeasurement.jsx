/**
 * DepthWizard — DistanceMeasurement (Phase 9, Task 9.3)
 *
 * Two-point distance measurement tool (§12).
 * Workflow:
 *   1. User clicks Point A
 *   2. User clicks Point B
 *   3. Calculates Horizontal Distance and 3D Distance
 *
 * Displays:
 *   DISTANCE
 *   Horizontal Distance: 42.1 m
 *   3D Distance:         43.0 m
 *
 * Only displays metric units when calibrated (isAbsolute); otherwise scene units (§12).
 *
 * DESIGN.md:
 *   - Two-column table layout
 *   - Left column --dw-fg-muted (label)
 *   - Right column data face (value + unit)
 *
 * Spec §12, §D10.
 */
import { useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { useApp } from '../../store/appStore.jsx';
import { Ruler, RotateCcw } from 'lucide-react';

const DistanceMeasurement = forwardRef(function DistanceMeasurement(
  { active = false, onActiveChange },
  ref
) {
  const { state } = useApp();
  const [step, setStep] = useState(0); // 0: need A, 1: need B, 2: completed
  const [pointA, setPointA] = useState(null);
  const [pointB, setPointB] = useState(null);
  const [result, setResult] = useState(null);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel  = isAbsolute ? 'm' : 'scene units';

  const clearMeasurement = useCallback(() => {
    setStep(0);
    setPointA(null);
    setPointB(null);
    setResult(null);
  }, []);

  // Handler invoked when terrain point is selected
  const handleSelectPoint = useCallback((point) => {
    if (!active) return;

    if (step === 0) {
      setPointA(point);
      setStep(1);
    } else if (step === 1) {
      setPointB(point);
      setStep(2);

      // Compute distances
      // Point coordinates: world x, z [-1, 1], elevation in units
      // If absolute, scale world [-1, 1] by real geographic width/height from metadata
      const widthMeters = state.terrain?.width ?? (isAbsolute ? 200.0 : 100.0);
      const heightMeters = state.terrain?.height ?? (isAbsolute ? 200.0 : 100.0);

      const dx = (point.x - pointA.x) * (widthMeters / 2.0);
      const dz = (point.z - pointA.z) * (heightMeters / 2.0);
      const dy = Math.abs(point.elevation - pointA.elevation);

      const horizDist = Math.hypot(dx, dz);
      const dist3D = Math.hypot(horizDist, dy);

      setResult({
        horizontal: horizDist,
        distance3D: dist3D,
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
      aria-label="Distance measurement results"
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
          <Ruler size={15} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11.5,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
          }}>
            DISTANCE
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
          Click initial point (A)…
        </p>
      )}

      {step === 1 && !result && (
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          color: 'var(--dw-accent)',
          margin: 0,
        }}>
          Click second point (B)…
        </p>
      )}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Horizontal Distance
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {result.horizontal.toFixed(1)} {unitLabel}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13.5, fontWeight: 600, color: 'var(--dw-fg)' }}>
              3D Distance
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 16,
              fontWeight: 600,
              color: 'var(--dw-accent)',
            }}>
              {result.distance3D.toFixed(1)} {unitLabel}
            </span>
          </div>
        </div>
      )}
    </div>
  );
});

export default DistanceMeasurement;

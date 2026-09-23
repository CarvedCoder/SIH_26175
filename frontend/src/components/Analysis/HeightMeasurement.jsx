/**
 * DepthWizard — HeightMeasurement (Phase 9, Task 9.2)
 *
 * Two-point structure height measurement tool (§11).
 * Workflow:
 *   1. User clicks Ground point
 *   2. User clicks Structure Top point
 *   3. Calculates height difference
 *
 * Displays:
 *   HEIGHT MEASUREMENT
 *   Ground Elevation: 126.2 m
 *   Top Elevation:    144.6 m
 *   Estimated Height:  18.4 m
 *
 * Remains visible until cleared (§11).
 * Uses API measureHeight(id, ground, top) with client-side fallback.
 *
 * DESIGN.md:
 *   - Two-column table layout
 *   - Left column --dw-fg-muted (label)
 *   - Right column data face (value + unit)
 *   - Clear button: 32px height, --dw-rim border
 *
 * Spec §11, §D10.
 */
import { useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { useApp } from '../../store/appStore.jsx';
import { measureHeight } from '../../api/terrain.js';
import { ArrowUpDown, RotateCcw } from 'lucide-react';

const HeightMeasurement = forwardRef(function HeightMeasurement(
  { active = false, onActiveChange },
  ref
) {
  const { state } = useApp();
  const [step, setStep] = useState(0); // 0: idle/need ground, 1: need top, 2: completed
  const [groundPoint, setGroundPoint] = useState(null);
  const [topPoint, setTopPoint] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel = 'm'; // world scale is metres (1 m/pixel documented fallback)

  const clearMeasurement = useCallback(() => {
    setStep(0);
    setGroundPoint(null);
    setTopPoint(null);
    setResult(null);
  }, []);

  // Called when terrain is clicked while height measurement is active
  const handleTerrainClick = useCallback(async (point) => {
    if (!active) return;

    if (step === 0) {
      setGroundPoint(point);
      setStep(1);
    } else if (step === 1) {
      setTopPoint(point);
      setStep(2);
      setLoading(true);

      const sceneId = state.scene?.scene_id;
      const groundElev = groundPoint.elevation;
      const topElev = point.elevation;
      const calculatedHeight = Math.abs(topElev - groundElev);

      try {
        if (sceneId && groundPoint.px != null && point.px != null) {
          // Metered path: the backend samples the exact DSM pixels
          const res = await measureHeight(
            sceneId,
            { x: groundPoint.px, y: groundPoint.py },
            { x: point.px, y: point.py },
          );
          setResult({
            groundElevation: res.ground_elevation ?? groundElev,
            topElevation: res.top_elevation ?? topElev,
            estimatedHeight: res.height ?? calculatedHeight,
            metered: res.height != null,
            confidence: point.confidence ?? groundPoint.confidence ?? null,
            precision_m: Math.max(point.precision_m ?? 0, groundPoint.precision_m ?? 0),
          });
        } else {
          setResult({
            groundElevation: groundElev,
            topElevation: topElev,
            estimatedHeight: calculatedHeight,
            metered: !!point.metered && !!groundPoint.metered,
            confidence: point.confidence ?? groundPoint.confidence ?? null,
            precision_m: Math.max(point.precision_m ?? 0, groundPoint.precision_m ?? 0),
          });
        }
      } catch {
        // Fallback to client-side calculated difference
        setResult({
          groundElevation: groundElev,
          topElevation: topElev,
          estimatedHeight: calculatedHeight,
          metered: !!point.metered && !!groundPoint.metered,
          confidence: point.confidence ?? groundPoint.confidence ?? null,
          precision_m: Math.max(point.precision_m ?? 0, groundPoint.precision_m ?? 0),
        });
      } finally {
        setLoading(false);
      }
    }
  }, [active, step, groundPoint, state.scene?.scene_id]);

  useImperativeHandle(ref, () => ({
    handleTerrainClick,
    clear: clearMeasurement,
  }));

  if (!active && !result) return null;

  return (
    <div
      aria-label="Height measurement results"
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
          <ArrowUpDown size={15} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11.5,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
          }}>
            HEIGHT MEASUREMENT
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
          Click ground surface point…
        </p>
      )}

      {step === 1 && !result && (
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          color: 'var(--dw-accent)',
          margin: 0,
        }}>
          Now click structure top…
        </p>
      )}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Ground Elevation
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {result.groundElevation.toFixed(1)} {unitLabel}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
              Top Elevation
            </span>
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 14, color: 'var(--dw-fg)' }}>
              {result.topElevation.toFixed(1)} {unitLabel}
            </span>
          </div>

          <div style={{ width: '100%', height: 1, background: 'var(--dw-rim)', margin: '2px 0' }} />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13.5, fontWeight: 600, color: 'var(--dw-fg)' }}>
              {result.metered ? 'Metered Height' : 'Estimated Height'}
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 16,
              fontWeight: 600,
              color: 'var(--dw-accent)',
            }}>
              {result.estimatedHeight.toFixed(1)} {unitLabel}
            </span>
          </div>

          {result.metered && result.confidence && (
            <div style={{
              borderTop: '1px solid var(--dw-rim)',
              paddingTop: 8,
              display: 'flex',
              flexDirection: 'column',
              gap: 3,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 12, color: 'var(--dw-fg-muted)' }}>
                  Height accuracy
                </span>
                <span style={{
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 12,
                  fontWeight: 600,
                  color: result.confidence.level === 'high' ? '#4ade80'
                    : result.confidence.level === 'medium' ? '#facc15' : 'var(--dw-fg-ghost)',
                }}>
                  {result.confidence.percent != null
                    ? `${result.confidence.percent}% confidence`
                    : `${result.confidence.level} confidence`}
                </span>
              </div>
              <div style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 11,
                lineHeight: 1.4,
                color: 'var(--dw-fg-ghost)',
              }}>
                {result.precision_m > 0 && `± ${result.precision_m.toFixed(4)} ${unitLabel} · `}
                {result.confidence.basis}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default HeightMeasurement;

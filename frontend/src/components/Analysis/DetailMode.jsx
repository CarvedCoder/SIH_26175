/**
 * DepthWizard — DetailMode UI (Phase 13, Tasks 13.1, 13.3, 13.4)
 *
 * Small-Structure / Detail Mode for high-resolution local tile depth refinement.
 * Addresses known limitation: loss of small structures during global depth estimation.
 *
 * Flow:
 *   Select Region (BBox) → Submit POST /scenes/{id}/refine → Polling Job → Local Update
 *
 * Spec §18, §65.
 * DESIGN.md: Operate mode, monospace data coordinates, thin borders, no card shadows.
 */
import { useState, useRef, useEffect } from 'react';
import { refineScene, getJobStatus } from '../../api/processing.js';
import {
  Sliders,
  Crop,
  CheckCircle2,
  AlertCircle,
  RotateCcw,
  Sparkles,
  Loader2,
  X,
} from 'lucide-react';

/**
 * @param {{
 *   sceneId?: string | null,
 *   selectedBbox?: { x_min: number, y_min: number, x_max: number, y_max: number } | null,
 *   onStartSelection?: () => void,
 *   onClearBbox?: () => void,
 *   isSelecting?: boolean,
 *   onRefineComplete?: (result: any) => void,
 *   compact?: boolean,
 * }} props
 */
export default function DetailMode({
  sceneId = null,
  selectedBbox = null,
  onStartSelection,
  onClearBbox,
  isSelecting = false,
  onRefineComplete,
  compact = false,
}) {
  const [resolution, setResolution] = useState('high'); // 'standard' | 'high'
  const [status, setStatus] = useState('idle'); // 'idle' | 'submitting' | 'processing' | 'completed' | 'error'
  const [errorMessage, setErrorMessage] = useState(null);
  const [activeJobId, setActiveJobId] = useState(null);
  const pollIntervalRef = useRef(null);

  // Clean up poll on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const handleStartRefine = async () => {
    if (!sceneId || !selectedBbox) return;

    setStatus('submitting');
    setErrorMessage(null);

    try {
      const res = await refineScene(sceneId, {
        bbox: selectedBbox,
        resolution,
      });

      const jobId = res?.job_id;
      if (!jobId) {
        // Immediate completion or mock fallback
        setStatus('completed');
        onRefineComplete?.(res);
        return;
      }

      setActiveJobId(jobId);
      setStatus('processing');

      // Poll job status
      pollIntervalRef.current = setInterval(async () => {
        try {
          const job = await getJobStatus(jobId);
          if (job.status === 'completed') {
            clearInterval(pollIntervalRef.current);
            setStatus('completed');
            onRefineComplete?.(job);
          } else if (job.status === 'failed') {
            clearInterval(pollIntervalRef.current);
            setStatus('error');
            setErrorMessage(job.message ?? 'Refinement processing failed.');
          }
        } catch (pollErr) {
          console.warn('[DetailMode] polling err', pollErr);
        }
      }, 2000);
    } catch (err) {
      setStatus('error');
      setErrorMessage(err.message ?? 'Failed to submit refinement job.');
    }
  };

  const handleReset = () => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    setStatus('idle');
    setErrorMessage(null);
    setActiveJobId(null);
    onClearBbox?.();
  };

  const hasBbox = !!selectedBbox;
  const isBusy = status === 'submitting' || status === 'processing';

  return (
    <div
      role="region"
      aria-label="Detail refinement mode"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: compact ? 10 : 14,
        width: '100%',
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingBottom: 6,
        borderBottom: '1px solid var(--dw-rim)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Sparkles size={13} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 10,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            fontWeight: 500,
          }}>
            Detail Mode (§18)
          </span>
        </div>

        {hasBbox && !isBusy && (
          <button
            onClick={handleReset}
            aria-label="Reset selection"
            title="Clear selection"
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--dw-fg-muted)',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
              outline: 'none',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '2px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            <X size={12} strokeWidth={1.5} />
          </button>
        )}
      </div>

      {/* Resolution Selector (Standard <-> High Resolution) */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 11, color: 'var(--dw-fg-muted)' }}>
            Target Detail Level
          </span>
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 10,
            color: resolution === 'high' ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
            textTransform: 'uppercase',
          }}>
            {resolution === 'high' ? 'High-Res (0.5× Sub-Tile)' : 'Standard'}
          </span>
        </div>

        {/* Segmented slider/toggle buttons */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          background: 'var(--dw-surface)',
          padding: 2,
          borderRadius: 'var(--dw-radius-sm)',
          border: '1px solid var(--dw-rim)',
          gap: 2,
        }}>
          <button
            onClick={() => setResolution('standard')}
            disabled={isBusy}
            aria-pressed={resolution === 'standard'}
            style={{
              height: 28,
              background: resolution === 'standard' ? 'var(--dw-panel)' : 'transparent',
              border: resolution === 'standard' ? '1px solid var(--dw-accent)' : '1px solid transparent',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              color: resolution === 'standard' ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
              cursor: isBusy ? 'not-allowed' : 'pointer',
              outline: 'none',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '1px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            Standard
          </button>

          <button
            onClick={() => setResolution('high')}
            disabled={isBusy}
            aria-pressed={resolution === 'high'}
            style={{
              height: 28,
              background: resolution === 'high' ? 'var(--dw-panel)' : 'transparent',
              border: resolution === 'high' ? '1px solid var(--dw-accent)' : '1px solid transparent',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              fontWeight: resolution === 'high' ? 500 : 400,
              color: resolution === 'high' ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
              cursor: isBusy ? 'not-allowed' : 'pointer',
              outline: 'none',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '1px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            High Resolution
          </button>
        </div>
      </div>

      {/* Selected Area Coordinates / Selection Prompt */}
      <div style={{
        padding: '8px 10px',
        background: 'var(--dw-surface)',
        border: '1px solid ' + (isSelecting ? 'var(--dw-accent)' : 'var(--dw-rim)'),
        borderRadius: 'var(--dw-radius-sm)',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 10, textTransform: 'uppercase', color: 'var(--dw-fg-ghost)' }}>
            Selected Region (BBox)
          </span>
          {hasBbox && (
            <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 9, color: 'var(--dw-confirm)' }}>
              READY
            </span>
          )}
        </div>

        {hasBbox ? (
          <div style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 11,
            color: 'var(--dw-fg)',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}>
            <div>X: [{selectedBbox.x_min.toFixed(3)}, {selectedBbox.x_max.toFixed(3)}]</div>
            <div>Y: [{selectedBbox.y_min.toFixed(3)}, {selectedBbox.y_max.toFixed(3)}]</div>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
            <span style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              color: isSelecting ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
            }}>
              {isSelecting ? 'Click and drag on terrain…' : 'No region selected'}
            </span>
            {onStartSelection && (
              <button
                onClick={onStartSelection}
                disabled={isBusy}
                aria-pressed={isSelecting}
                style={{
                  height: 24,
                  padding: '0 8px',
                  background: isSelecting ? 'var(--dw-accent)' : 'transparent',
                  border: '1px solid var(--dw-accent)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 11,
                  color: isSelecting ? '#fff' : 'var(--dw-accent)',
                  cursor: isBusy ? 'not-allowed' : 'pointer',
                  outline: 'none',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                }}
                onFocus={e => {
                  e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                  e.currentTarget.style.outlineOffset = '2px';
                }}
                onBlur={e => { e.currentTarget.style.outline = 'none'; }}
              >
                <Crop size={11} strokeWidth={1.5} />
                {isSelecting ? 'Selecting…' : 'Select Box'}
              </button>
            )}
          </div>
        )}
      </div>

      {/* Action Button: Refine Area */}
      <div>
        <button
          onClick={handleStartRefine}
          disabled={!hasBbox || isBusy}
          aria-label="Refine selected area"
          style={{
            width: '100%',
            height: 34,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
            background: hasBbox && !isBusy ? 'var(--dw-accent)' : 'var(--dw-surface)',
            border: hasBbox && !isBusy ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 12,
            fontWeight: 500,
            color: hasBbox && !isBusy ? '#fff' : 'var(--dw-fg-ghost)',
            cursor: hasBbox && !isBusy ? 'pointer' : 'not-allowed',
            outline: 'none',
            transition: 'background 120ms ease, border-color 120ms ease',
          }}
          onFocus={e => {
            if (hasBbox && !isBusy) {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '2px';
            }
          }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        >
          {isBusy ? (
            <>
              <Loader2 size={13} strokeWidth={1.5} className="animate-spin" />
              <span>Refining selected area…</span>
            </>
          ) : (
            <>
              <Sparkles size={13} strokeWidth={1.5} />
              <span>Refine Area</span>
            </>
          )}
        </button>
      </div>

      {/* Processing Status Banner (§18: "The UI should clearly indicate when local refinement is being processed.") */}
      {isBusy && (
        <div style={{
          padding: '8px 10px',
          background: 'var(--dw-surface)',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          borderRadius: 'var(--dw-radius-sm)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <div style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: 'var(--dw-live)',
            animation: 'pulse 1.2s infinite',
          }} />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            color: 'var(--dw-fg)',
          }}>
            Processing high-res local tile depth refinement…
          </span>
        </div>
      )}

      {/* Completed State */}
      {status === 'completed' && (
        <div style={{
          padding: '8px 10px',
          background: 'rgba(34, 197, 94, 0.08)',
          border: '1px solid rgba(34, 197, 94, 0.25)',
          borderRadius: 'var(--dw-radius-sm)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <CheckCircle2 size={14} strokeWidth={1.5} color="var(--dw-confirm)" aria-hidden="true" />
            <span style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              color: 'var(--dw-fg)',
            }}>
              Local DSM updated with refined detail.
            </span>
          </div>
          <button
            onClick={handleReset}
            style={{
              background: 'none',
              border: 'none',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 10,
              color: 'var(--dw-accent)',
              cursor: 'pointer',
              padding: '2px 4px',
              outline: 'none',
            }}
          >
            Refine another
          </button>
        </div>
      )}

      {/* Error State */}
      {status === 'error' && (
        <div style={{
          padding: '8px 10px',
          background: 'rgba(239, 68, 68, 0.08)',
          border: '1px solid rgba(239, 68, 68, 0.25)',
          borderRadius: 'var(--dw-radius-sm)',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}>
          <AlertCircle size={14} strokeWidth={1.5} color="var(--dw-fault)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            color: 'var(--dw-fault)',
          }}>
            {errorMessage ?? 'Refinement failed.'}
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * DepthWizard — ApiErrorAlert (Phase 18, Task 18.2)
 *
 * Renders structured diagnostic errors adhering to spec §31 Rule 6:
 *   - What happened
 *   - Why it happened
 *   - What can I do next?
 *
 * Uses formatApiError from api/errors.js.
 *
 * DESIGN.md: Operate mode. Instrument panel register. No card shadows. Monospace error code.
 */
import { AlertTriangle, RotateCcw, ArrowRight, Home } from 'lucide-react';
import { formatApiError } from '../../api/errors.js';
import { useApp } from '../../store/appStore.jsx';

/**
 * @param {{
 *   error: any,
 *   onRetry?: () => void,
 *   onFallbackRelative?: () => void,
 *   onReset?: () => void,
 * }} props
 */
export default function ApiErrorAlert({
  error,
  onRetry,
  onFallbackRelative,
  onReset,
}) {
  const { actions } = useApp();
  const diagnosis = formatApiError(error);

  const handleRetry = onRetry ?? (() => actions.retry?.());
  const handleFallback = onFallbackRelative ?? (() => actions.fallbackRelative?.());
  const handleReset = onReset ?? (() => actions.reset?.());

  return (
    <div
      role="alert"
      aria-live="polite"
      style={{
        width: '100%',
        maxWidth: 580,
        background: 'var(--dw-panel)',
        border: '1px solid rgba(239,68,68,0.3)',
        borderRadius: 'var(--dw-radius-md)',
        padding: '16px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        boxSizing: 'border-box',
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingBottom: 8,
        borderBottom: '1px solid var(--dw-rim)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <AlertTriangle size={15} strokeWidth={1.5} color="var(--dw-fault)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13,
            fontWeight: 600,
            color: 'var(--dw-fg)',
          }}>
            {diagnosis.title}
          </span>
        </div>

        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 10,
          color: 'var(--dw-fault)',
          padding: '1px 6px',
          background: 'rgba(239,68,68,0.08)',
          borderRadius: 3,
          border: '1px solid rgba(239,68,68,0.2)',
        }}>
          {diagnosis.code}
        </span>
      </div>

      {/* 3-Part Diagnostic Body (§31 Rule 6) */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 12,
        lineHeight: 1.5,
      }}>
        {/* What happened */}
        <div>
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 9,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            display: 'block',
            fontWeight: 500,
            marginBottom: 2,
          }}>
            What happened
          </span>
          <span style={{ color: 'var(--dw-fg)' }}>
            {diagnosis.what}
          </span>
        </div>

        {/* Why it happened */}
        <div>
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 9,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            display: 'block',
            fontWeight: 500,
            marginBottom: 2,
          }}>
            Why it happened
          </span>
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 11,
            color: 'var(--dw-fg-muted)',
            display: 'block',
            background: 'var(--dw-surface)',
            padding: '4px 8px',
            borderRadius: 'var(--dw-radius-sm)',
            border: '1px solid var(--dw-rim)',
          }}>
            {diagnosis.why}
          </span>
        </div>

        {/* What can I do next */}
        <div>
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 9,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            display: 'block',
            fontWeight: 500,
            marginBottom: 2,
          }}>
            What can I do next?
          </span>
          <span style={{ color: 'var(--dw-fg-muted)' }}>
            {diagnosis.whatNext}
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        paddingTop: 10,
        borderTop: '1px solid var(--dw-rim)',
        flexWrap: 'wrap',
      }}>
        {diagnosis.recoverable && (
          <button
            onClick={handleRetry}
            style={{
              height: 30,
              padding: '0 12px',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              background: 'var(--dw-accent)',
              border: 'none',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              fontWeight: 500,
              color: '#fff',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            <RotateCcw size={12} strokeWidth={1.5} />
            Retry
          </button>
        )}

        {diagnosis.canFallbackRelative && (
          <button
            onClick={handleFallback}
            style={{
              height: 30,
              padding: '0 12px',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              background: 'var(--dw-surface)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              color: 'var(--dw-fg)',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            <span>Continue with Relative DSM</span>
            <ArrowRight size={12} strokeWidth={1.5} />
          </button>
        )}

        <button
          onClick={handleReset}
          style={{
            height: 30,
            padding: '0 10px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            background: 'none',
            border: 'none',
            color: 'var(--dw-fg-ghost)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            cursor: 'pointer',
            marginLeft: 'auto',
            outline: 'none',
          }}
          onMouseEnter={e => { e.currentTarget.style.color = 'var(--dw-fg-muted)'; }}
          onMouseLeave={e => { e.currentTarget.style.color = 'var(--dw-fg-ghost)'; }}
        >
          <Home size={12} strokeWidth={1.5} />
          Return to Landing
        </button>
      </div>
    </div>
  );
}

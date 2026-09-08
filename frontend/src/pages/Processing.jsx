/**
 * DepthWizard — Processing page
 *
 * Shown while state.status === 'PROCESSING'.
 * Mounts useProcessing hook which drives all state transitions.
 * Spec §6. DESIGN.md: no generic spinner, no invented ETA.
 */
import { useCallback, useState } from 'react';
import Header from '../components/common/Header.jsx';
import PipelineProgress from '../components/Processing/PipelineProgress.jsx';
import ProcessingStatus from '../components/Processing/ProcessingStatus.jsx';
import { useApp } from '../store/appStore.jsx';
import { useProcessing } from '../hooks/useProcessing.js';

export default function Processing() {
  const { state } = useApp();
  const { cancel } = useProcessing(); // mounts polling
  const [confirming, setConfirming] = useState(false);

  const handleCancel = useCallback(async () => {
    if (!confirming) {
      setConfirming(true);
      return;
    }
    setConfirming(false);
    await cancel();
  }, [confirming, cancel]);

  const scene = state.scene;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '64px 24px',
        gap: 24,
      }}>

        {/* Scene identifier */}
        {scene && (
          <p style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 13.5,
            fontWeight: 500,
            color: 'var(--dw-fg)',
            letterSpacing: '0.04em',
            margin: 0,
          }}>
            {scene.filename}
          </p>
        )}

        <PipelineProgress />

        <ProcessingStatus />

        {/* Cancel */}
        {confirming ? (
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 14, color: 'var(--dw-fg)' }}>
              Cancel processing?
            </span>
            <button
              onClick={handleCancel}
              style={{
                background: 'none',
                border: '1px solid var(--dw-fault)',
                borderRadius: 'var(--dw-radius-sm)',
                padding: '6px 14px',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 13,
                fontWeight: 500,
                color: 'var(--dw-fault)',
                cursor: 'pointer',
                outline: 'none',
              }}
              onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-fault)'; e.currentTarget.style.outlineOffset = '2px'; }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              Confirm cancel
            </button>
            <button
              onClick={() => setConfirming(false)}
              style={{
                background: 'none',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                padding: '6px 14px',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 13,
                color: 'var(--dw-fg-muted)',
                cursor: 'pointer',
                outline: 'none',
              }}
              onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              Keep processing
            </button>
          </div>
        ) : (
          <button
            onClick={handleCancel}
            style={{
              background: 'none',
              border: 'none',
              padding: '6px 12px',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              color: 'var(--dw-fg-muted)',
              cursor: 'pointer',
              outline: 'none',
              letterSpacing: '0.01em',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = 'var(--dw-fault)'; }}
            onMouseLeave={e => { e.currentTarget.style.color = 'var(--dw-fg-muted)'; }}
            onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            Cancel processing
          </button>
        )}

        {/* Failure panel — shown if processing transitions to FAILED from another path */}
        {state.status === 'FAILED' && state.error && (
          <FailurePanel error={state.error} />
        )}
      </main>
    </div>
  );
}

function FailurePanel({ error }) {
  const { actions } = useApp();
  return (
    <div
      role="alert"
      style={{
        width: '100%',
        maxWidth: 520,
        padding: '20px 24px',
        background: 'rgba(239,68,68,0.06)',
        border: '1px solid rgba(239,68,68,0.25)',
        borderRadius: 'var(--dw-radius)',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
    >
      <p style={{ fontFamily: 'var(--dw-font-data)', fontSize: 12, letterSpacing: '0.06em', color: 'var(--dw-fault)', margin: 0, fontWeight: 600 }}>
        DSM GENERATION INTERRUPTED
      </p>
      <p style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 15, fontWeight: 500, color: 'var(--dw-fg)', margin: 0 }}>
        We could not complete this stage.
      </p>
      {error.message && (
        <p style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13.5, color: 'var(--dw-fg-muted)', margin: 0, lineHeight: 1.5 }}>
          {error.message}
        </p>
      )}
      <div style={{ display: 'flex', gap: 10, marginTop: 6 }}>
        <button
          onClick={() => actions.retry()}
          style={{
            background: 'none',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius-sm)',
            padding: '6px 16px',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13,
            fontWeight: 500,
            color: 'var(--dw-fg)',
            cursor: 'pointer',
            outline: 'none',
          }}
          onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        >
          Retry
        </button>
        {error.recoverable && (
          <button
            onClick={() => actions.fallbackRelative()}
            style={{
              background: 'none',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              padding: '6px 16px',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              color: 'var(--dw-fg-muted)',
              cursor: 'pointer',
              outline: 'none',
            }}
            onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            Continue with Relative DSM
          </button>
        )}
      </div>
    </div>
  );
}

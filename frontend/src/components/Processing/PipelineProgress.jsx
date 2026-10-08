/**
 * DepthWizard — Pipeline Progress
 *
 * Stage-by-stage checklist (✓ done, ● active, ○ pending) over a real
 * progress bar driven ONLY by backend-reported job.progress — every
 * percentage point is real completed work (tiled inference reports
 * done/total per forward; the service maps it onto 5%–80%).
 * Maps backend stage names → human labels (spec §49).
 * No fake ETAs. No generic spinner. No animated fake increments.
 * DESIGN.md: pipeline stages are a list, not a set of cards.
 */
import { useState } from 'react';
import { useApp } from '../../store/appStore.jsx';

/** Maps backend stage string → human-readable label */
const STAGE_LABELS = {
  queued:               'Waiting in queue',
  preprocessing:        'Preparing image & loading model',
  depth_inference:      'Estimating depth & height',
  // terraheight_s override — TerraHeight-S runs NO relative-depth stage:
  // RGB -> tiled AGL inference -> stitched height raster, directly.
  'terraheight_s:depth_inference': 'Loading TerraHeight-S · running tiled AGL inference',
  calibration:          'Recovering metric scale',
  dsm_generation:       'Generating DSM',
  terrain_generation:   'Building 3D terrain',
  validation:           'Validating result',
  disaster_assessment:  'Assessing disaster damage',
  disaster_complete:    'Disaster assessment complete',
  dsm_ready:            'Elevation products ready',
  analyzing:            'Running deferred analysis',
  finalizing:           'Finalizing',
  completed:            'Complete',
  failed:               'Failed',
  cancelled:            'Cancelled',
};

// Must mirror the stages the backend actually emits
// (backend/app/schemas/job.py JobStage, processing_service.py).
const STAGE_ORDER = [
  'queued',
  'preprocessing',
  'depth_inference',
  'disaster_assessment',
  'disaster_complete',
];

function stageIndex(stage) {
  return STAGE_ORDER.indexOf(stage);
}

function StageRow({ stage, currentStage, jobStatus, heightBackend, wasActive }) {
  const currentIdx = stageIndex(currentStage);
  const thisIdx    = stageIndex(stage);

  const isComplete = jobStatus === 'completed' || thisIdx < currentIdx;
  const isActive   = currentStage === stage && jobStatus !== 'completed'
    && jobStatus !== 'failed' && jobStatus !== 'cancelled';
  // A failure lands on the generic 'failed' stage — the row that was live
  // when it happened is the one that failed (tracked by the parent).
  const isFailed   = jobStatus === 'failed' && wasActive;

  let icon, iconColor;
  if (isFailed) {
    icon = '✕'; iconColor = 'var(--dw-fault)';
  } else if (isComplete) {
    icon = '✓'; iconColor = 'var(--dw-confirm)';
  } else if (isActive) {
    icon = '●'; iconColor = 'var(--dw-live)';
  } else {
    icon = '○'; iconColor = 'var(--dw-fg-ghost)';
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '7px 0',
        position: 'relative',
        zIndex: 1,
      }}
      aria-current={isActive ? 'step' : undefined}
    >
      <span
        aria-hidden="true"
        style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 14,
          color: iconColor,
          width: 18,
          flexShrink: 0,
          textAlign: 'center',
          background: 'var(--dw-surface)',
          transition: 'color 200ms ease',
          ...(isActive && {
            animation: 'dw-pulse 1.2s ease-in-out infinite',
          }),
        }}
      >
        {icon}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 14.5,
        color: isFailed ? 'var(--dw-fault)'
          : isActive ? 'var(--dw-fg)'
          : isComplete ? 'var(--dw-fg-muted)'
          : 'var(--dw-fg-ghost)',
        fontWeight: isActive || isFailed ? 600 : 400,
        transition: 'color 200ms ease',
      }}>
        {STAGE_LABELS[`${heightBackend}:${stage}`] ?? STAGE_LABELS[stage] ?? stage}
      </span>
    </div>
  );
}

export default function PipelineProgress() {
  const { state } = useApp();
  const job = state.job;

  const currentStage = job?.stage  ?? job?.status ?? 'queued';
  const jobStatus    = job?.status ?? 'queued';

  // Monotonic display value: a poll can deliver a record older than the
  // previous one (two in-flight fetches crossing) — the bar never moves
  // backwards. Backend progress is real work, so no synthetic increments.
  // "Adjust state during render" pattern (react.dev): derive from the
  // previous committed job record, conditionally.
  const [shown, setShown] = useState(0);
  const [lastActive, setLastActive] = useState('queued');
  const [prevJob, setPrevJob] = useState(null);
  if (job !== prevJob) {
    setPrevJob(job);
    const raw = jobStatus === 'completed' ? 100
      : typeof job?.progress === 'number' ? job.progress : 0;
    setShown(s => Math.max(s, Math.round(raw)));
    if (jobStatus === 'processing' && job?.stage) setLastActive(job.stage);
  }
  const pct = shown;

  const isFailed     = jobStatus === 'failed';
  const isCancelled  = jobStatus === 'cancelled';
  const isDone       = jobStatus === 'completed';
  const fillColor    = isFailed || isCancelled ? 'var(--dw-fault)' : isDone ? 'var(--dw-confirm)' : 'var(--dw-live)';
  const message      = job?.message ?? null;

  return (
    <section
      aria-label="Processing pipeline progress"
      style={{
        width: '100%',
        maxWidth: 520,
        background: 'var(--dw-surface)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius)',
        padding: '20px 24px',
        display: 'flex',
        flexDirection: 'column',
        gap: 0,
      }}
    >
      {/* Header row: section label + live percentage readout */}
      <div style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        margin: '0 0 10px 0',
      }}>
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-muted)',
          margin: 0,
        }}>
          {isFailed ? 'Pipeline failed' : isCancelled ? 'Cancelled' : isDone ? 'Complete' : 'Processing'}
        </p>
        <p
          aria-live="polite"
          style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 15,
            fontWeight: 500,
            fontVariantNumeric: 'tabular-nums',
            color: isFailed || isCancelled ? 'var(--dw-fault)' : 'var(--dw-fg)',
            margin: 0,
          }}
        >
          {pct}
          <span style={{ color: 'var(--dw-fg-ghost)', fontSize: 12 }}>%</span>
        </p>
      </div>

      {/* Progress bar — driven only by backend-reported job.progress */}
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        style={{
          height: 4,
          borderRadius: 2,
          background: 'var(--dw-rim)',
          overflow: 'hidden',
          marginBottom: 4,
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            borderRadius: 2,
            background: fillColor,
            transition: 'width 600ms ease, background 300ms ease',
          }}
        />
      </div>

      {/* Live backend message (e.g. "Tile 12 / 40") */}
      <p style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 12.5,
        color: 'var(--dw-fg-ghost)',
        margin: '0 0 14px 0',
        minHeight: 18,
        fontVariantNumeric: 'tabular-nums',
      }}>
        {message ?? (isDone ? 'All stages complete' : '\u00A0')}
      </p>

      {/* Stage list with a connector rail through the status icons */}
      <div style={{ position: 'relative' }}>
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            left: 8,
            top: 16,
            bottom: 16,
            width: 1,
            background: 'var(--dw-rim)',
          }}
        />
        {STAGE_ORDER.map(stage => (
          <StageRow
            key={stage}
            stage={stage}
            currentStage={currentStage}
            jobStatus={jobStatus}
            heightBackend={state.modelBackend}
            wasActive={lastActive === stage}
          />
        ))}
      </div>

      <style>{`
        @keyframes dw-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </section>
  );
}

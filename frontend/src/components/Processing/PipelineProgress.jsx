/**
 * DepthWizard — Pipeline Progress
 *
 * Stage-by-stage checklist backed by the job's REAL state: the backend
 * reports (stage, progress%) through cooperative checkpoints — per tile
 * during depth inference, around refinement, DSM writing and validation —
 * and the poll surfaces it here.
 *
 *   - Overall bar: live job.progress with a smooth width transition and a
 *     moving sheen while a stage is running.
 *   - Stage list: ✓ done, ● active (pulsing, with its own live percent),
 *     ○ pending.
 * No fake ETAs. No generic spinner.
 * DESIGN.md: pipeline stages are a list, not a set of cards.
 */
import { useApp } from '../../store/appStore.jsx';

/** Maps backend stage string → human-readable label */
const STAGE_LABELS = {
  queued:           'Waiting in queue',
  preprocessing:    'Preparing image',
  depth_inference:  'Estimating depth',
  refinement:       'Refining structures',
  dsm_generation:   'Generating DSM',
  validation:       'Validating result',
  completed:        'Complete',
  failed:           'Failed',
  cancelled:        'Cancelled',
};

const STAGE_ORDER = [
  'queued',
  'preprocessing',
  'depth_inference',
  'refinement',
  'dsm_generation',
  'validation',
];

function stageIndex(stage) {
  return STAGE_ORDER.indexOf(stage);
}

function StageRow({ stage, currentStage, jobStatus, jobProgress }) {
  const currentIdx = stageIndex(currentStage);
  const thisIdx    = stageIndex(stage);

  const isComplete = jobStatus === 'completed' || (thisIdx < currentIdx && thisIdx >= 0);
  const isActive   = currentStage === stage && jobStatus !== 'completed';
  const isFailed   = jobStatus === 'failed' && currentStage === stage;

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
        gap: 10,
        padding: '6px 0',
      }}
      aria-current={isActive ? 'step' : undefined}
    >
      <span
        aria-hidden="true"
        style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 14,
          color: iconColor,
          width: 16,
          flexShrink: 0,
          textAlign: 'center',
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
        color: isActive ? 'var(--dw-fg)' : isComplete ? 'var(--dw-fg-muted)' : 'var(--dw-fg-ghost)',
        fontWeight: isActive ? 600 : 400,
        transition: 'color 200ms ease',
        flex: 1,
      }}>
        {STAGE_LABELS[stage] ?? stage}
      </span>
      {isActive && (
        <span
          style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 12.5,
            color: 'var(--dw-accent)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {Math.round(jobProgress)}%
        </span>
      )}
    </div>
  );
}

export default function PipelineProgress() {
  const { state } = useApp();
  const job = state.job;

  const currentStage = job?.stage  ?? job?.status ?? 'queued';
  const jobStatus    = job?.status ?? 'queued';
  const rawProgress  = typeof job?.progress === 'number' ? job.progress : 0;
  const percent = jobStatus === 'completed' ? 100 : Math.min(99, Math.max(0, rawProgress));
  const running = jobStatus === 'processing' || jobStatus === 'queued';
  const activeStep = STAGE_ORDER.indexOf(currentStage) >= 0 ? STAGE_ORDER.indexOf(currentStage) + 1 : 1;

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
      {/* Section label + step counter */}
      <div style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        margin: '0 0 14px 0',
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
          PROCESSING
        </p>
        <p
          aria-live="polite"
          style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 12,
            color: 'var(--dw-fg-ghost)',
            margin: 0,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          STEP {activeStep} / {STAGE_ORDER.length}
        </p>
      </div>

      {/* Overall progress bar — live percent from the job record */}
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(percent)}
        aria-label="Overall pipeline progress"
        style={{
          height: 8,
          width: '100%',
          background: 'var(--dw-void)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 999,
          overflow: 'hidden',
          marginBottom: 6,
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${percent}%`,
            borderRadius: 999,
            background: 'linear-gradient(90deg, #2b9fe0, var(--dw-accent))',
            transition: 'width 700ms cubic-bezier(0.25, 1, 0.4, 1)',
            ...(running && {
              backgroundImage:
                'linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0) 40%, rgba(255,255,255,0.35) 50%, rgba(255,255,255,0) 60%, rgba(255,255,255,0) 100%), linear-gradient(90deg, #2b9fe0, var(--dw-accent))',
              backgroundSize: '48px 100%, 100% 100%',
              animation: 'dw-progress-sheen 1.1s linear infinite',
            }),
            ...(jobStatus === 'completed' && {
              background: 'var(--dw-confirm)',
            }),
            ...(jobStatus === 'failed' && {
              background: 'var(--dw-fault)',
            }),
          }}
        />
      </div>

      {/* Percent + current stage caption */}
      <div style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        margin: '0 0 14px 0',
      }}>
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13,
          color: 'var(--dw-fg-muted)',
          margin: 0,
        }}>
          {STAGE_LABELS[currentStage] ?? currentStage}
          {running && currentStage !== 'queued' ? ' — working…' : ''}
        </p>
        <p
          style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 18,
            fontWeight: 500,
            color: jobStatus === 'completed' ? 'var(--dw-confirm)' : 'var(--dw-fg)',
            margin: 0,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {Math.round(percent)}%
        </p>
      </div>

      {/* Stage list */}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {STAGE_ORDER.map(stage => (
          <StageRow
            key={stage}
            stage={stage}
            currentStage={currentStage}
            jobStatus={jobStatus}
            jobProgress={percent}
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

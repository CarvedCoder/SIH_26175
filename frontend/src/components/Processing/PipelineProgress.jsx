/**
 * DepthWizard — Pipeline Progress
 *
 * Stage-by-stage checklist: ✓ done, ● active, ○ pending.
 * STAGE_ORDER mirrors backend.app.schemas.job.JobStage exactly — the
 * backend emits every one of these stages with real progress, so the
 * list is a live mirror of what the worker is doing right now.
 * No fake ETAs. No generic spinner.
 * DESIGN.md: pipeline stages are a list, not a set of cards.
 */
import { useApp } from '../../store/appStore.jsx';

/** Maps backend stage string → human-readable label */
const STAGE_LABELS = {
  queued:          'Waiting in queue',
  validating:      'Validating input',
  preprocessing:   'Preparing image',
  depth_inference: 'Estimating depth',
  calibration:     'Refining & calibrating scale',
  dsm_generation:  'Generating DSM',
  validation:      'Validating result',
  finalizing:      'Publishing outputs',
  completed:       'Complete',
  failed:          'Failed',
  cancelled:       'Cancelled',
};

const STAGE_ORDER = [
  'queued',
  'validating',
  'preprocessing',
  'depth_inference',
  'calibration',
  'dsm_generation',
  'validation',
  'finalizing',
];

function stageIndex(stage) {
  return STAGE_ORDER.indexOf(stage);
}

function StageRow({ stage, currentStage, jobStatus, index }) {
  const currentIdx = stageIndex(currentStage);
  const thisIdx    = stageIndex(stage);

  const isComplete = jobStatus === 'completed' || thisIdx < currentIdx;
  const isActive   = currentStage === stage && jobStatus !== 'completed';
  const isFailed   = jobStatus === 'failed' && currentStage === stage;

  let icon, iconColor, iconClass;
  if (isFailed) {
    icon = '✕'; iconColor = 'var(--dw-fault)';
  } else if (isComplete) {
    icon = '✓'; iconColor = 'var(--dw-confirm)'; iconClass = 'dw-stage-check';
  } else if (isActive) {
    icon = '●'; iconColor = 'var(--dw-live)'; iconClass = 'dw-stage-dot';
  } else {
    icon = '○'; iconColor = 'var(--dw-fg-ghost)';
  }

  return (
    <div
      className="dw-stage-row"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '6px 0',
        animation: `dw-row-in 420ms ease backwards`,
        animationDelay: `${index * 60}ms`,
      }}
      aria-current={isActive ? 'step' : undefined}
    >
      <span
        aria-hidden="true"
        className={iconClass}
        style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 14,
          color: iconColor,
          width: 16,
          flexShrink: 0,
          textAlign: 'center',
          transition: 'color 200ms ease',
          ...(isActive && { animation: 'dw-pulse 1.2s ease-in-out infinite' }),
        }}
      >
        {icon}
      </span>
      <span
        className={isActive ? 'dw-stage-label-active' : undefined}
        style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 14.5,
          color: isActive ? 'var(--dw-fg)' : isComplete ? 'var(--dw-fg-muted)' : 'var(--dw-fg-ghost)',
          fontWeight: isActive ? 600 : 400,
          transition: 'color 200ms ease',
        }}
      >
        {STAGE_LABELS[stage] ?? stage}
      </span>
      {isActive && (
        <span
          aria-hidden="true"
          style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 11,
            letterSpacing: '0.08em',
            color: 'var(--dw-live)',
            marginLeft: 'auto',
            animation: 'dw-dots 1.4s steps(4, end) infinite',
          }}
        >
          WORKING
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
  const progress     = job?.progress;

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
        animation: 'dw-card-in 500ms ease backwards',
      }}
    >
      {/* Section label */}
      <p style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 12,
        fontWeight: 600,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-muted)',
        margin: '0 0 14px 0',
      }}>
        PROCESSING
      </p>

      {/* Stage list */}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {STAGE_ORDER.map((stage, i) => (
          <StageRow
            key={stage}
            stage={stage}
            index={i}
            currentStage={currentStage}
            jobStatus={jobStatus}
          />
        ))}
      </div>

      {/* Overall progress bar — mirrors the real job progress */}
      {progress != null && (
        <div
          role="progressbar"
          aria-valuenow={Math.round(progress)}
          aria-valuemin={0}
          aria-valuemax={100}
          style={{
            marginTop: 16,
            height: 4,
            borderRadius: 2,
            background: 'var(--dw-rim)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${Math.min(Math.max(progress, 0), 100)}%`,
              borderRadius: 2,
              background: 'linear-gradient(90deg, var(--dw-accent), var(--dw-live))',
              transition: 'width 700ms cubic-bezier(0.22, 1, 0.36, 1)',
              ...(jobStatus !== 'completed' && {
                backgroundImage:
                  'linear-gradient(90deg, var(--dw-accent), var(--dw-live), var(--dw-accent))',
                backgroundSize: '200% 100%',
                animation: 'dw-bar-shimmer 2.4s linear infinite',
              }),
            }}
          />
        </div>
      )}

      <style>{`
        @keyframes dw-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        @keyframes dw-row-in {
          from { opacity: 0; transform: translateX(-10px); }
          to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes dw-card-in {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes dw-check-in {
          from { transform: scale(0.3); opacity: 0; }
          to   { transform: scale(1); opacity: 1; }
        }
        @keyframes dw-dots {
          0%   { clip-path: inset(0 100% 0 0); }
          100% { clip-path: inset(0 0 0 0); }
        }
        @keyframes dw-bar-shimmer {
          from { background-position: 0% 0; }
          to   { background-position: 200% 0; }
        }
        .dw-stage-check {
          animation: dw-check-in 260ms cubic-bezier(0.34, 1.56, 0.64, 1) backwards;
        }
        .dw-stage-label-active {
          text-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
        }
      `}</style>
    </section>
  );
}

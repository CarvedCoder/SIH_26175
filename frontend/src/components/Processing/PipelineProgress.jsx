/**
 * DepthWizard — Pipeline Progress
 *
 * Stage-by-stage checklist: ✓ done, ● active, ○ pending.
 * Maps backend stage names → human labels (spec §49).
 * No fake ETAs. No generic spinner.
 * DESIGN.md: pipeline stages are a list, not a set of cards.
 */
import { useApp } from '../../store/appStore.jsx';

/** Maps backend stage string → human-readable label */
const STAGE_LABELS = {
  queued:               'Waiting in queue',
  preprocessing:        'Preparing image',
  depth_estimation:     'Estimating depth',
  geospatial_alignment: 'Aligning geospatial data',
  scale_calibration:    'Recovering metric scale',
  refinement:           'Refining structures',
  dsm_generation:       'Generating DSM',
  validation:           'Validating result',
  terrain_generation:   'Building 3D terrain',
  completed:            'Complete',
  failed:               'Failed',
  cancelled:            'Cancelled',
};

const STAGE_ORDER = [
  'queued',
  'preprocessing',
  'depth_estimation',
  'geospatial_alignment',
  'scale_calibration',
  'refinement',
  'dsm_generation',
  'validation',
  'terrain_generation',
];

function stageIndex(stage) {
  return STAGE_ORDER.indexOf(stage);
}

function StageRow({ stage, currentStage, jobStatus }) {
  const currentIdx = stageIndex(currentStage);
  const thisIdx    = stageIndex(stage);

  const isComplete = jobStatus === 'completed' || thisIdx < currentIdx;
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
      }}>
        {STAGE_LABELS[stage] ?? stage}
      </span>
    </div>
  );
}

export default function PipelineProgress() {
  const { state } = useApp();
  const job = state.job;

  const currentStage = job?.stage  ?? job?.status ?? 'queued';
  const jobStatus    = job?.status ?? 'queued';

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
        {STAGE_ORDER.map(stage => (
          <StageRow
            key={stage}
            stage={stage}
            currentStage={currentStage}
            jobStatus={jobStatus}
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

/**
 * DepthWizard — Home page
 *
 * Three states driven by app state machine:
 *  NO_SCENE   → hero + upload zone
 *  UPLOADING  → hero + upload zone (uploading state shown inside UploadZone)
 *  SCENE_READY → file info + start processing CTA
 *
 * Design: instrument-panel aesthetic per DESIGN.md.
 * No kicker/eyebrow labels (craft-floor ban).
 * No hero-metric template.
 * Signature element: pipeline flow (RGB→Depth→DSM→3D→Analysis).
 */
import { useCallback } from 'react';
import { ArrowRight, Play } from 'lucide-react';
import Header from '../components/common/Header.jsx';
import UploadZone from '../components/Upload/UploadZone.jsx';
import FileInfo from '../components/Upload/FileInfo.jsx';
import ProcessingPath from '../components/Upload/ProcessingPath.jsx';
import RecentProjects from '../components/common/RecentProjects.jsx';
import ApiErrorAlert from '../components/common/ApiErrorAlert.jsx';
import { useApp, AppState } from '../store/appStore.jsx';
import { startProcessing } from '../api/processing.js';

/* Pipeline diagram steps */
const PIPELINE = [
  { id: 'rgb',      label: 'RGB IMAGE',  sub: 'Source' },
  { id: 'depth',    label: 'DEPTH',      sub: 'Model output' },
  { id: 'dsm',      label: 'DSM',        sub: 'Elevation' },
  { id: 'terrain',  label: '3D TERRAIN', sub: 'Mesh' },
  { id: 'analysis', label: 'ANALYSIS',   sub: 'Measurement' },
];

function PipelineDiagram() {
  return (
    <div
      aria-label="Processing pipeline: RGB Image → Depth → DSM → 3D Terrain → Analysis"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 0,
        flexWrap: 'nowrap',
        overflowX: 'auto',
      }}
    >
      {PIPELINE.map((step, i) => (
        <span key={step.id} style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
          <span style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <span style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 12.5,
              fontWeight: 600,
              letterSpacing: '0.08em',
              color: 'var(--dw-fg)',
            }}>
              {step.label}
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11.5,
              color: 'var(--dw-fg-muted)',
              letterSpacing: '0.03em',
            }}>
              {step.sub}
            </span>
          </span>
          {i < PIPELINE.length - 1 && (
            <span style={{
              margin: '0 12px',
              color: 'var(--dw-rim)',
              fontSize: 16,
              fontFamily: 'var(--dw-font-data)',
              marginBottom: 10,
            }}>
              ↓
            </span>
          )}
        </span>
      ))}
    </div>
  );
}

function StartProcessingPanel({ sceneId }) {
  const { actions } = useApp();

  const handleStart = useCallback(async () => {
    try {
      const job = await startProcessing(sceneId);
      actions.startProcessing(job.job_id);
    } catch (err) {
      actions.processingFail({
        code:        err.code        ?? 'PROCESSING_ERROR',
        message:     err.message     ?? 'Could not start processing.',
        recoverable: err.recoverable ?? true,
      });
    }
  }, [sceneId, actions]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 16,
      width: '100%',
      maxWidth: 580,
    }}>
      <button
        onClick={handleStart}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '12px 32px',
          background: 'var(--dw-accent)',
          border: 'none',
          borderRadius: 'var(--dw-radius-sm)',
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 15,
          fontWeight: 600,
          color: '#fff',
          cursor: 'pointer',
          transition: 'opacity 150ms ease',
          outline: 'none',
          letterSpacing: '0.01em',
        }}
        onMouseEnter={e => { e.currentTarget.style.opacity = '0.88'; }}
        onMouseLeave={e => { e.currentTarget.style.opacity = '1'; }}
        onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
        onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        aria-label="Start depth and DSM processing"
      >
        <Play size={16} strokeWidth={1.5} />
        Start Processing
      </button>
      <p style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 13,
        color: 'var(--dw-fg-muted)',
        margin: 0,
        textAlign: 'center',
      }}>
        Processing time depends on image size and selected model.
      </p>
    </div>
  );
}

export default function Home() {
  const { state, actions } = useApp();

  const showFileInfo   = state.status === AppState.SCENE_READY;
  const showUploadZone = state.status === AppState.NO_SCENE || state.status === AppState.UPLOADING;
  const showStart      = showFileInfo;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '72px 24px 64px',
        gap: 0,
      }}>

        {/* Hero */}
        <section
          style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 24, marginBottom: 56, textAlign: 'center', maxWidth: 720 }}
          aria-label="DepthWizard hero"
        >
          <h1 style={{
            fontFamily: 'var(--dw-font-ui)',
            fontWeight: 600,
            fontSize: 'clamp(2.2rem, 5vw, 3.25rem)',
            letterSpacing: '-0.03em',
            color: 'var(--dw-fg)',
            lineHeight: 1.15,
            margin: 0,
          }}>
            From a single image<br />to an interactive 3D terrain.
          </h1>
          <p style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 17,
            color: 'var(--dw-fg-muted)',
            margin: 0,
            lineHeight: 1.65,
            maxWidth: 540,
          }}>
            Turn optical remote-sensing imagery into depth, elevation and an explorable 3D terrain you can navigate, measure and validate.
          </p>

          {/* Pipeline flow — horizontal on desktop */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 0,
            padding: '16px 24px',
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius)',
            marginTop: 8,
          }}>
            {PIPELINE.map((step, i) => (
              <span key={step.id} style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
                <span style={{
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 12.5,
                  fontWeight: 500,
                  letterSpacing: '0.06em',
                  color: 'var(--dw-fg)',
                }}>
                  {step.label}
                </span>
                {i < PIPELINE.length - 1 && (
                  <span style={{
                    margin: '0 14px',
                    color: 'var(--dw-rim)',
                    fontSize: 14,
                  }}>→</span>
                )}
              </span>
            ))}
          </div>

          {/* Quick Demo CTA (§1.2, §33) */}
          {showUploadZone && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 4 }}>
              <button
                onClick={() => {
                  const demoProject = state.recentScenes?.[0];
                  if (demoProject) {
                    actions.resumeSession(demoProject, AppState.RESULTS_READY);
                  }
                }}
                style={{
                  height: 38,
                  padding: '0 18px',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  background: 'none',
                  border: '1px solid var(--dw-accent)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 14,
                  fontWeight: 500,
                  color: 'var(--dw-accent)',
                  cursor: 'pointer',
                  outline: 'none',
                  transition: 'background 150ms ease',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--dw-surface)'}
                onMouseLeave={e => e.currentTarget.style.background = 'none'}
                aria-label="View pre-calibrated demo scene (Scene_042 · Absolute DSM)"
              >
                <Play size={14} strokeWidth={1.5} />
                View Demo (Scene_042 · Absolute DSM)
              </button>
            </div>
          )}
        </section>

        {/* Upload or file info */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 16,
          width: '100%',
        }}>
          {showUploadZone && <UploadZone />}
          {showFileInfo   && <FileInfo />}
          {showStart      && state.scene && <StartProcessingPanel sceneId={state.scene.scene_id} />}

          {/* Recent projects list (§28) */}
          {showUploadZone && (
            <div style={{
              width: '100%',
              maxWidth: 560,
              marginTop: 32,
            }}>
              <RecentProjects />
            </div>
          )}
        </div>

        {/* Error state (§31 Rule 6, §69) */}
        {state.status === AppState.FAILED && state.error && (
          <div style={{ marginTop: 32, width: '100%', maxWidth: 560 }}>
            <ApiErrorAlert error={state.error} />
          </div>
        )}
      </main>
    </div>
  );
}

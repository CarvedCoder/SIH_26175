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
              fontSize: 10,
              fontWeight: 500,
              letterSpacing: '0.08em',
              color: 'var(--dw-fg)',
            }}>
              {step.label}
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 10,
              color: 'var(--dw-fg-ghost)',
              letterSpacing: '0.03em',
            }}>
              {step.sub}
            </span>
          </span>
          {i < PIPELINE.length - 1 && (
            <span style={{
              margin: '0 10px',
              color: 'var(--dw-rim)',
              fontSize: 14,
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
      maxWidth: 560,
    }}>
      <button
        onClick={handleStart}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '10px 24px',
          background: 'var(--dw-accent)',
          border: 'none',
          borderRadius: 'var(--dw-radius-sm)',
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13,
          fontWeight: 500,
          color: '#fff',
          cursor: 'pointer',
          transition: 'opacity 150ms ease',
          outline: 'none',
          letterSpacing: '0em',
        }}
        onMouseEnter={e => { e.currentTarget.style.opacity = '0.88'; }}
        onMouseLeave={e => { e.currentTarget.style.opacity = '1'; }}
        onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
        onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        aria-label="Start depth and DSM processing"
      >
        <Play size={14} strokeWidth={1.5} />
        Start Processing
      </button>
      <p style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 11,
        color: 'var(--dw-fg-ghost)',
        margin: 0,
        textAlign: 'center',
      }}>
        Processing time depends on image size and selected model.
      </p>
    </div>
  );
}

export default function Home() {
  const { state } = useApp();

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
          style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20, marginBottom: 52, textAlign: 'center', maxWidth: 640 }}
          aria-label="DepthWizard hero"
        >
          <h1 style={{
            fontFamily: 'var(--dw-font-ui)',
            fontWeight: 600,
            fontSize: 'clamp(1.75rem, 4vw, 2.75rem)',
            letterSpacing: '-0.03em',
            color: 'var(--dw-fg)',
            lineHeight: 1.1,
            margin: 0,
          }}>
            From a single image<br />to an interactive 3D terrain.
          </h1>
          <p style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 15,
            color: 'var(--dw-fg-muted)',
            margin: 0,
            lineHeight: 1.6,
            maxWidth: 480,
          }}>
            Turn optical remote-sensing imagery into depth, elevation and an explorable 3D terrain you can navigate, measure and validate.
          </p>

          {/* Pipeline flow — horizontal on desktop */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 0,
            padding: '14px 20px',
            background: 'var(--dw-surface)',
            border: '1px solid var(--dw-rim)',
            borderRadius: 'var(--dw-radius)',
            marginTop: 8,
          }}>
            {PIPELINE.map((step, i) => (
              <span key={step.id} style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
                <span style={{
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 10.5,
                  letterSpacing: '0.06em',
                  color: 'var(--dw-fg-muted)',
                }}>
                  {step.label}
                </span>
                {i < PIPELINE.length - 1 && (
                  <span style={{
                    margin: '0 10px',
                    color: 'var(--dw-rim)',
                    fontSize: 12,
                  }}>→</span>
                )}
              </span>
            ))}
          </div>
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

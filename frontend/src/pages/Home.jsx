/**
 * DepthWizard — Home / Workspace dashboard
 *
 * Three states driven by app state machine:
 *  NO_SCENE   → upload zone + demo + recent projects
 *  UPLOADING  → upload zone (uploading state shown inside UploadZone)
 *  SCENE_READY → file info + start processing CTA
 *
 * Design: dashboard layout per DESIGN.md — compact operate-mode heading,
 * two-column task zone (primary action left, recent sessions right,
 * sticky on desktop), full-width pipeline overview below.
 * No kicker/eyebrow labels (craft-floor ban). No hero-metric template.
 * Signature element: pipeline flow (RGB→Depth→DSM→3D→Analysis).
 */
import { Fragment, useCallback } from 'react';
import { ArrowRight, Box, FileText, Image as ImageIcon, Layers, Mountain, Play, Ruler } from 'lucide-react';
import Header from '../components/common/Header.jsx';
import UploadZone from '../components/Upload/UploadZone.jsx';
import FileInfo from '../components/Upload/FileInfo.jsx';
import RecentProjects from '../components/common/RecentProjects.jsx';
import ApiErrorAlert from '../components/common/ApiErrorAlert.jsx';
import { useApp, AppState } from '../store/appStore.jsx';
import { useAuth } from '../store/authContext.jsx';
import { useViewport } from '../hooks/useMediaQuery.js';
import { startProcessing } from '../api/processing.js';

/* Pipeline overview steps (§1.2 pipeline flow) — shared with HelpPage */
export const STEPS = [
  { id: 'rgb',      label: 'RGB IMAGE',  sub: 'Optical source upload',      Icon: ImageIcon },
  { id: 'depth',    label: 'DEPTH',      sub: 'Monocular depth model',      Icon: Layers },
  { id: 'dsm',      label: 'DSM',        sub: 'Surface elevation model',    Icon: Mountain },
  { id: 'terrain',  label: '3D TERRAIN', sub: 'Navigable reconstruction',   Icon: Box },
  { id: 'analysis', label: 'ANALYSIS',   sub: 'Measure and validate',       Icon: Ruler },
];

function PipelineOverview() {
  const { isDesktop } = useViewport();

  return (
    <section
      aria-label="How DepthWizard works"
      style={{
        background: 'var(--dw-panel)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius)',
        padding: '18px 24px 20px',
        width: '100%',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-muted)',
        }}>
          How it works
        </span>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 11.5,
          letterSpacing: '0.04em',
          color: 'var(--dw-fg-ghost)',
        }}>
          ONE IMAGE → EXPLORE · MEASURE · EXPORT
        </span>
      </div>

      {/* Steps — horizontal on desktop, wrapped chips below 1024px */}
      <div style={{
        display: 'flex',
        alignItems: 'stretch',
        justifyContent: 'space-between',
        flexWrap: isDesktop ? 'nowrap' : 'wrap',
        gap: '14px 0',
        marginTop: 16,
      }}>
        {STEPS.map((step, i) => (
          <Fragment key={step.id}>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 6,
              flex: isDesktop ? '1 1 0' : '1 1 150px',
              minWidth: isDesktop ? 0 : 150,
              paddingRight: isDesktop ? 8 : 12,
            }}>
              <step.Icon size={16} strokeWidth={1.5} color="var(--dw-fg-muted)" aria-hidden="true" />
              <span style={{
                fontFamily: 'var(--dw-font-data)',
                fontSize: 12.5,
                fontWeight: 600,
                letterSpacing: '0.06em',
                color: 'var(--dw-fg)',
                whiteSpace: 'nowrap',
              }}>
                {step.label}
              </span>
              <span style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 12,
                lineHeight: 1.45,
                color: 'var(--dw-fg-muted)',
              }}>
                {step.sub}
              </span>
            </div>
            {i < STEPS.length - 1 && isDesktop && (
              <span
                aria-hidden="true"
                style={{
                  alignSelf: 'center',
                  flexShrink: 0,
                  marginTop: 26,
                  color: 'var(--dw-rim-strong)',
                }}
              >
                <ArrowRight size={14} strokeWidth={1.5} />
              </span>
            )}
          </Fragment>
        ))}
      </div>

      {/* Two processing paths — capability truth from D10 */}
      <div style={{
        borderTop: '1px solid var(--dw-rim)',
        marginTop: 18,
        paddingTop: 14,
        display: 'flex',
        flexDirection: isDesktop ? 'row' : 'column',
        alignItems: isDesktop ? 'center' : 'flex-start',
        flexWrap: 'wrap',
        gap: '8px 32px',
      }}>
        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          rowGap: 2,
          gap: 8,
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 12.5,
          color: 'var(--dw-fg-muted)',
        }}>
          <FileText size={13} strokeWidth={1.5} color="var(--dw-fg-ghost)" aria-hidden="true" />
          GeoTIFF with georeference
          <ArrowRight size={12} strokeWidth={1.5} color="var(--dw-fg-ghost)" aria-hidden="true" />
          <span style={{ color: 'var(--dw-fg)', fontFamily: 'var(--dw-font-data)', fontSize: 12 }}>
            Absolute DSM
          </span>
          — metric elevations
        </span>
        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          rowGap: 2,
          gap: 8,
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 12.5,
          color: 'var(--dw-fg-muted)',
        }}>
          <ImageIcon size={13} strokeWidth={1.5} color="var(--dw-fg-ghost)" aria-hidden="true" />
          PNG / JPG
          <ArrowRight size={12} strokeWidth={1.5} color="var(--dw-fg-ghost)" aria-hidden="true" />
          <span style={{ color: 'var(--dw-fg)', fontFamily: 'var(--dw-font-data)', fontSize: 12 }}>
            Relative DSM
          </span>
          — scene units
        </span>
      </div>
    </section>
  );
}

/* Height-model backend switch (RDAH integration): pretrained RDAH-Net by
 * default, legacy CalibrationNet one click away. The choice rides on the
 * process request and is kept for refine so a scene stays consistent. */
const BACKEND_OPTIONS = [
  {
    value: 'rdah',
    label: 'RDAH-Net',
    note: 'Pretrained height regression · unclamped nDSM (m)',
  },
  {
    value: 'calibration_net',
    label: 'CalibrationNet',
    note: 'Legacy per-tile affine calibration · clamped heights',
  },
];

function ModelBackendSelector() {
  const { state, actions } = useApp();
  const active = BACKEND_OPTIONS.find(o => o.value === state.modelBackend)
    ?? BACKEND_OPTIONS[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 11,
        fontWeight: 500,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-ghost)',
      }}>
        Height model
      </span>
      <div role="radiogroup" aria-label="Height model backend" style={{
        display: 'flex',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        overflow: 'hidden',
      }}>
        {BACKEND_OPTIONS.map((opt, i) => {
          const selected = opt.value === state.modelBackend;
          return (
            <button
              key={opt.value}
              role="radio"
              aria-checked={selected}
              onClick={() => actions.setModelBackend(opt.value)}
              style={{
                flex: 1,
                height: 36,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                background: selected ? 'var(--dw-accent)' : 'var(--dw-surface)',
                border: 'none',
                borderLeft: i > 0 ? '1px solid var(--dw-rim)' : 'none',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 13,
                fontWeight: selected ? 600 : 500,
                color: selected ? 'var(--dw-fg-invert)' : 'var(--dw-fg-muted)',
                cursor: 'pointer',
                outline: 'none',
                transition: 'background 120ms ease, color 120ms ease',
              }}
              onMouseEnter={e => {
                if (!selected) e.currentTarget.style.background = 'var(--dw-hover)';
              }}
              onMouseLeave={e => {
                if (!selected) e.currentTarget.style.background = 'var(--dw-surface)';
              }}
              onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '-2px'; }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 12,
        color: 'var(--dw-fg-muted)',
      }}>
        {active.note}
      </span>
    </div>
  );
}

function StartProcessingPanel({ sceneId }) {
  const { state, actions } = useApp();
  const architecture = state.modelBackend;

  const handleStart = useCallback(async () => {
    try {
      const job = await startProcessing(sceneId, { architecture });
      actions.startProcessing(job.job_id);
    } catch (err) {
      actions.processingFail({
        code:        err.code        ?? 'PROCESSING_ERROR',
        message:     err.message     ?? 'Could not start processing.',
        recoverable: err.recoverable ?? true,
      });
    }
  }, [sceneId, architecture, actions]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
      width: '100%',
    }}>
      <ModelBackendSelector />
      <button
        onClick={handleStart}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 10,
          height: 44,
          width: '100%',
          background: 'var(--dw-accent)',
          border: '1px solid var(--dw-accent)',
          borderRadius: 'var(--dw-radius-sm)',
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 15,
          fontWeight: 600,
          color: 'var(--dw-fg-invert)',
          cursor: 'pointer',
          transition: 'background 120ms ease, border-color 120ms ease',
          outline: 'none',
          letterSpacing: '0.01em',
        }}
        onMouseEnter={e => {
          e.currentTarget.style.background = '#ffffff';
          e.currentTarget.style.borderColor = '#ffffff';
        }}
        onMouseLeave={e => {
          e.currentTarget.style.background = 'var(--dw-accent)';
          e.currentTarget.style.borderColor = 'var(--dw-accent)';
        }}
        onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
        onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        aria-label="Start depth and DSM processing"
      >
        <Play size={16} strokeWidth={1.5} />
        Start Processing
        <ArrowRight size={16} strokeWidth={1.5} aria-hidden="true" />
      </button>
      <p style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 12.5,
        color: 'var(--dw-fg-muted)',
        margin: 0,
      }}>
        Processing time depends on image size and selected model. You will see each
        pipeline stage live while it runs.
      </p>
    </div>
  );
}

/* Quiet secondary CTA — resumes the seeded demo scene (jury path) */
function DemoSceneButton() {
  const { state, actions } = useApp();

  const demoProject = state.recentScenes?.[0];
  if (!demoProject) return null;

  return (
    <button
      onClick={() => actions.resumeSession(demoProject, AppState.RESULTS_READY)}
      style={{
        minHeight: 36,
        padding: '0 14px',
        alignSelf: 'flex-start',
        maxWidth: '100%',
        textAlign: 'left',
        display: 'inline-flex',
        alignItems: 'center',
        gap: 8,
        background: 'none',
        border: '1px solid var(--dw-rim-strong)',
        borderRadius: 'var(--dw-radius-sm)',
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 13.5,
        fontWeight: 500,
        color: 'var(--dw-fg-muted)',
        cursor: 'pointer',
        outline: 'none',
        transition: 'background 150ms ease, border-color 150ms ease, color 150ms ease',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = 'var(--dw-surface)';
        e.currentTarget.style.borderColor = 'var(--dw-fg-ghost)';
        e.currentTarget.style.color = 'var(--dw-fg)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = 'none';
        e.currentTarget.style.borderColor = 'var(--dw-rim-strong)';
        e.currentTarget.style.color = 'var(--dw-fg-muted)';
      }}
      onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
      onBlur={e => { e.currentTarget.style.outline = 'none'; }}
      aria-label="View pre-calibrated demo scene (Scene_042 · Absolute DSM)"
    >
      <Play size={13} strokeWidth={1.5} />
      Open demo scene (Scene_042 · Absolute DSM)
    </button>
  );
}

export default function Home() {
  const { state, actions } = useApp();
  const { setView } = useAuth();
  const { isDesktop } = useViewport();

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
        gap: 28,
        width: '100%',
        maxWidth: 1200,
        margin: '0 auto',
        padding: '36px 24px 64px',
      }}>

        {/* Page heading — compact, operate mode */}
        <section style={{ width: '100%' }} aria-label="Workspace overview">
          <h1 style={{
            fontFamily: 'var(--dw-font-ui)',
            fontWeight: 600,
            fontSize: 'clamp(1.45rem, 2.2vw, 1.75rem)',
            letterSpacing: '-0.02em',
            color: 'var(--dw-fg)',
            lineHeight: 1.2,
            margin: 0,
          }}>
            From a single image to an interactive 3D terrain.
          </h1>
          <p style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 14.5,
            color: 'var(--dw-fg-muted)',
            margin: '8px 0 0',
            lineHeight: 1.6,
            maxWidth: 640,
          }}>
            Upload optical imagery or resume a recent scene — depth, elevation and 3D
            reconstruction run automatically, then explore, measure and validate the result.
          </p>
        </section>

        {/* Dashboard grid — task zone left, session history right */}
        <section style={{
          display: 'grid',
          gridTemplateColumns: isDesktop ? 'minmax(0, 1fr) 380px' : 'minmax(0, 1fr)',
          gap: 32,
          alignItems: 'start',
          width: '100%',
        }} aria-label="Workspace dashboard">

          {/* Primary task column */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'stretch',
            gap: 16,
            minWidth: 0,
          }}>
            {showUploadZone && <UploadZone />}
            {showUploadZone && <DemoSceneButton />}
            {showFileInfo && <FileInfo />}
            {showStart && state.scene && <StartProcessingPanel sceneId={state.scene.scene_id} />}

            {/* Error state (§31 Rule 6, §69) */}
            {state.status === AppState.FAILED && state.error && (
              <ApiErrorAlert error={state.error} />
            )}
          </div>

          {/* Session history sidebar — sticky on desktop, capped to the
              3 most recent so a long history cannot stretch the layout;
              the full list lives on the Recent Projects page */}
          <aside
            aria-label="Recent projects sidebar"
            style={{
              minWidth: 0,
              ...(isDesktop && { position: 'sticky', top: 'calc(var(--dw-header-h) + 28px)' }),
            }}
          >
            <RecentProjects
              compact
              limit={3}
              onViewAll={() => setView('recent')}
            />
          </aside>
        </section>

        {/* Pipeline overview — full width band */}
        <PipelineOverview />
      </main>
    </div>
  );
}

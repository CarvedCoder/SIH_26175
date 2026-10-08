/**
 * DepthWizard — Result Dashboard (Phase 3)
 *
 * Shown when state.status === 'RESULTS_READY'.
 * Displays three result layers: RGB source, Depth map, DSM.
 * DSM is hidden when elevation_mode === 'relative' (§54, §D10).
 * "Enter 3D Terrain" CTA transitions to TERRAIN_LOADING (task 3.3).
 *
 * Data flow:
 *   - state.results already contains ResultsMeta from processingDone
 *   - Depth metadata fetched from GET /scenes/{id}/depth
 *   - DSM metadata fetched from GET /scenes/{id}/dsm
 *   - RGB preview URL constructed from scene data or results
 *
 * DESIGN.md: Operate mode. Instrument panel aesthetic.
 * No card shadows. No decorative statistics. Monospace for data values.
 * Spec §33 Scene 3, §53 (Depth), §54 (DSM).
 */
import { useEffect, useRef, useState } from 'react';
import { Mountain, Download, RefreshCw, Loader2, AlertTriangle } from 'lucide-react';
import Header from '../components/common/Header.jsx';
import LayerImageCard from '../components/common/LayerImageCard.jsx';
import MetricsPanel from '../components/Validation/MetricsPanel.jsx';
import ExportPanel from '../components/Export/ExportPanel.jsx';
import PartialResultBanner from '../components/common/PartialResultBanner.jsx';
import ApiErrorAlert from '../components/common/ApiErrorAlert.jsx';
import { useValidation } from '../hooks/useValidation.js';
import { useApp } from '../store/appStore.jsx';
import { getDepth, getDsm, getScene, getResults } from '../api/results.js';
import { getRouteHeatmap } from '../api/route.js';
import { startAnalysis, getJobStatus } from '../api/processing.js';
import { resolveAssetUrl } from '../api/client.js';

/**
 * @typedef {'idle'|'loading'|'done'|'error'} FetchState
 */

export default function ResultDashboard() {
  const { state, actions } = useApp();
  const scene  = state.scene;
  const results = state.results; // ResultsMeta from processingDone

  /** @type {[import('../types/api.js').DepthResult|null, Function]} */
  const [depthData, setDepthData]   = useState(null);
  /** @type {[import('../types/api.js').DsmResult|null, Function]} */
  const [dsmData, setDsmData]       = useState(null);
  /** Mini passability heat map (jury round 2): URL + blocked/caution stats */
  const [heatmap, setHeatmap]       = useState(null);
  const [fetchState, setFetchState] = useState(/** @type {FetchState} */ ('loading'));
  // Authoritative scene record from the backend — resumed sessions carry a
  // reconstructed scene object whose georef/CRS/dimensions may be guesses.
  const [sceneDetail, setSceneDetail] = useState(null);
  const [showExport, setShowExport] = useState(false);

  // Deferred-analysis state (lazy pipeline): the processing job completed
  // with elevation products only; validation / disaster / buildings-3D run
  // on demand when the user clicks "Run Full Analysis". Tracked locally —
  // the dashboard stays usable while the analysis job runs.
  const [analysisJob, setAnalysisJob] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [refreshTick, setRefreshTick] = useState(0);
  const [showLeavePopup, setShowLeavePopup] = useState(false);
  const analysisPollRef = useRef(null);

  // Capability flags from ResultsMeta. A resumed session may carry a
  // reconstructed results object, so prefer the backend's authoritative
  // processing_path once the scene detail has loaded.
  const elevationMode = results?.elevation_mode ?? 'relative';
  const isAbsolute    = sceneDetail
    ? sceneDetail.processing_path === 'absolute'
    : elevationMode === 'absolute';

  const { validation, reference, isLoading: valLoading } = useValidation(scene?.scene_id, isAbsolute);

  useEffect(() => {
    if (!scene?.scene_id) return;

    let cancelled = false;
    setFetchState('loading');

    async function fetchLayerMeta() {
      // Each layer degrades independently: a 404 on one endpoint (e.g. DSM
      // on a relative pipeline) must not blank the others — RELIEF, the
      // NO-GO metadata and the mini heat map stay visible.
      const settled = await Promise.allSettled([
        getScene(scene.scene_id),
        getDepth(scene.scene_id),
        isAbsolute ? getDsm(scene.scene_id) : Promise.resolve(null),
        getRouteHeatmap(scene.scene_id),
      ]);
      if (cancelled) return;
      const [fresh, depth, dsm, heat] = settled.map(
        (r) => (r.status === 'fulfilled' ? r.value : null),
      );
      setSceneDetail(fresh);
      if (depth === null && dsm === null && heat === null) {
        setFetchState('error');
        return;
      }
      setDepthData(depth);
      setDsmData(dsm);
      setHeatmap(heat);
      setFetchState('done');
    }

    fetchLayerMeta();
    return () => { cancelled = true; };
  }, [scene?.scene_id, isAbsolute, refreshTick]);

  /** Units label from D10: 'm' for absolute, 'scene units' for relative */
  const elevUnits = 'm'; // world scale is metres (1 m/pixel documented fallback)

  // -- Deferred analysis (lazy pipeline) ------------------------------------
  const analysisPending = results?.analysis_status === 'pending';
  const analysisRunning =
    !!analysisJob && !['completed', 'failed', 'cancelled'].includes(analysisJob.status);

  const runAnalysis = async () => {
    if (!scene?.scene_id || analysisRunning) return;
    setAnalysisError(null);
    try {
      const accepted = await startAnalysis(scene.scene_id);
      setAnalysisJob({
        job_id: accepted.job_id,
        status: accepted.status ?? 'queued',
        stage: accepted.stage ?? 'queued',
        progress: accepted.progress ?? 0,
        message: null,
      });
      setShowLeavePopup(true);
    } catch (err) {
      setAnalysisError(err?.message ?? 'Could not start the analysis job.');
    }
  };

  // Browser-level guard: while the analysis job runs, warn before the user
  // closes / reloads / navigates away — the live progress and the
  // auto-refresh of the results live on this page.
  useEffect(() => {
    if (!analysisRunning) return undefined;
    const warn = (e) => {
      e.preventDefault();
      e.returnValue = ''; // required for Chrome to show the dialog
      return '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [analysisRunning]);

  // Poll the analysis job; on completion refresh the authoritative results
  // (analysis_status flips to "complete", validation metrics appear) and
  // re-run the layer-metadata fetch via refreshTick.
  useEffect(() => {
    const jobId = analysisJob?.job_id;
    if (!jobId) return undefined;
    let stopped = false;
    const poll = async () => {
      try {
        const job = await getJobStatus(jobId);
        if (stopped) return;
        setAnalysisJob(job);
        if (job.status === 'completed') {
          clearInterval(analysisPollRef.current);
          try {
            const fresh = await getResults(scene.scene_id);
            actions.processingDone(fresh);
          } catch { /* results can be re-fetched later */ }
          setRefreshTick(t => t + 1);
        } else if (job.status === 'failed') {
          clearInterval(analysisPollRef.current);
          setAnalysisError(job.error?.message ?? job.message ?? 'Analysis failed.');
        }
      } catch { /* transient network error — the interval keeps polling */ }
    };
    poll();
    analysisPollRef.current = setInterval(poll, 2000);
    return () => {
      stopped = true;
      clearInterval(analysisPollRef.current);
    };
  }, [analysisJob?.job_id, scene?.scene_id, actions]);

  const ANALYSIS_STAGE_LABELS = {
    queued: 'Waiting in queue',
    analyzing: 'Running deferred analysis',
    validation: 'Validating against reference',
    disaster_assessment: 'Assessing disaster damage',
    disaster_complete: 'Disaster assessment complete',
    completed: 'Analysis complete',
  };

  /**
   * The backend exposes depth and DSM preview images via their `url` / `download_url`.
   * RGB comes from the scene source — `results.preview_url` when the session carries
   * it (fresh processing), else the canonical /results/preview endpoint for resumed
   * sessions whose reconstructed results object lacks the URL.
   */
  const rgbUrl  = resolveAssetUrl(
    results?.preview_url
      ?? (scene?.scene_id ? `/api/v1/scenes/${scene.scene_id}/results/preview` : null)
  );
  const depthUrl = resolveAssetUrl(depthData?.url ?? null);
  const dsmUrl   = resolveAssetUrl(dsmData?.download_url ?? null);

  const isLoading = fetchState === 'loading';

  // Prefer the backend's authoritative scene record; fall back to the
  // (possibly reconstructed) state.scene for offline-resumed sessions.
  const geo = sceneDetail?.georeference ?? null;
  const metaScene = sceneDetail
    ? {
        ...scene,
        filename: sceneDetail.filename ?? scene.filename,
        dimensions: sceneDetail.dimensions ?? scene.dimensions,
        format: sceneDetail.format ?? scene.format,
        georeferenced: geo ? geo.available === true : scene.georeferenced,
        crs: geo ? geo.crs : scene.crs,
      }
    : scene;

  // Processing provenance (Part O): the backend mirrors the inference
  // payload meta through GET /scenes/{id}/results — height model, output
  // type, DEM provenance. Absent on resumed/legacy sessions; every item
  // degrades independently and never shows an unearned "Absolute DSM".
  const provMeta = results?.metadata ?? null;
  const MODEL_LABELS = {
    terraheight_s: 'TerraHeight-S',
    rdah: 'RDAH',
    calibration_net: 'CalibrationNet',
  };
  const heightModelLabel =
    provMeta?.height_model_label
    ?? MODEL_LABELS[provMeta?.model_architecture]
    ?? null;
  const outputType = provMeta?.output_type ?? null;
  const pipelineLabel =
    outputType === 'absolute_dsm' ? 'Absolute DSM'
    : outputType === 'anchored_constant_dsm' ? 'Anchored DSM (constant datum)'
    : outputType === 'relative_height' ? 'Relative (AGL)'
    : (isAbsolute ? 'Absolute DSM' : 'Relative DSM');
  const demProv = provMeta?.provenance ?? null;
  const gsdM = Array.isArray(provMeta?.pixel_size_m)
    ? provMeta.pixel_size_m[0]
    : null;

  // Scene metadata line items
  const meta = metaScene ? [
    { label: 'SOURCE', value: metaScene.filename },
    { label: 'DIMENSIONS', value: `${metaScene.dimensions?.width ?? '—'} × ${metaScene.dimensions?.height ?? '—'} px` },
    { label: 'FORMAT', value: metaScene.format },
    { label: 'GEOREF', value: metaScene.georeferenced ? 'YES' : 'NO' },
    metaScene.crs ? { label: 'CRS', value: metaScene.crs } : null,
    { label: 'PIPELINE', value: pipelineLabel },
    heightModelLabel ? { label: 'HEIGHT MODEL', value: heightModelLabel } : null,
    demProv?.dem_source ? { label: 'DEM SOURCE', value: demProv.dem_source } : null,
    demProv?.dem_vertical_reference ? { label: 'DEM VERT. REF', value: demProv.dem_vertical_reference } : null,
    gsdM != null ? { label: 'GSD', value: `${gsdM.toFixed(2)} m/px` } : null,
    depthData?.statistics ? { label: 'RELIEF', value: `${depthData.statistics.relief?.toFixed(1)} ${elevUnits}` } : null,
    heatmap ? { label: 'NO-GO AREA', value: `${heatmap.blocked_pct}% (fire truck)` } : null,
  ].filter(Boolean) : [];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        padding: '40px 24px 64px',
        maxWidth: 1200,
        width: '100%',
        margin: '0 auto',
        gap: 32,
      }}>

        {/* Page header — compact identifier, no decoration */}
        <section aria-labelledby="dashboard-heading">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <h1
              id="dashboard-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 22,
                fontWeight: 600,
                color: 'var(--dw-fg)',
                margin: 0,
              }}
            >
              Processing Complete
            </h1>
            <p style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 15,
              color: 'var(--dw-fg-muted)',
              margin: 0,
            }}>
              Review outputs below, then enter the 3D terrain workspace.
            </p>
          </div>
        </section>

        {/* Scene metadata strip — thin horizontal list of labelled fields */}
        {meta.length > 0 && (
          <section aria-label="Scene metadata">
            <div style={{
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius)',
              padding: '14px 20px',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '16px 36px',
              alignItems: 'center',
            }}>
              {meta.map(({ label, value }) => (
                <MetaField key={label} label={label} value={value} />
              ))}
              {/* Mini heat map inline (jury round 2): traffic-light risk
                  thumbnail for rapid visual recognition of no-go regions */}
              {heatmap?.url && (
                <div
                  data-testid="route-mini-heatmap"
                  style={{ display: 'flex', alignItems: 'center', gap: 10, marginLeft: 'auto' }}
                >
                  <img
                    src={resolveAssetUrl(heatmap.url)}
                    alt="Vehicle passability heat map (green = passable, red = blocked)"
                    style={{
                      height: 44, width: 'auto', maxWidth: 110, objectFit: 'contain',
                      borderRadius: 4, border: '1px solid var(--dw-rim)', imageRendering: 'pixelated',
                    }}
                  />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2, fontSize: 11 }}>
                    <span style={{ color: '#e74c3c' }}>■ blocked {heatmap.blocked_pct}%</span>
                    <span style={{ color: '#f1c40f' }}>■ caution {heatmap.caution_pct}%</span>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Pipeline capability evaluation banner (§38) */}
        <section aria-label="Pipeline capability evaluation">
          <PartialResultBanner
            format={metaScene?.format ?? scene?.format ?? (isAbsolute ? 'GeoTIFF' : 'PNG')}
            georeferenced={metaScene?.georeferenced ?? isAbsolute}
            elevationMode={isAbsolute ? 'absolute' : 'relative'}
            hasReference={isAbsolute && results?.reference_source != null}
            compact={false}
          />
        </section>

        {/* Deferred-analysis panel (lazy pipeline): elevation products are
            ready; validation / disaster / buildings-3D run on click. */}
        {(analysisPending || analysisRunning || analysisError) && (
          <section aria-label="Deferred analysis">
            <div style={{
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius)',
              padding: '16px 20px',
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
              maxWidth: 760,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: 240 }}>
                  <p style={{
                    fontFamily: 'var(--dw-font-ui)',
                    fontSize: 14.5,
                    fontWeight: 600,
                    color: 'var(--dw-fg)',
                    margin: 0,
                  }}>
                    {analysisRunning
                      ? 'Full analysis running…'
                      : analysisError
                        ? 'Analysis failed'
                        : 'Elevation products ready'}
                  </p>
                  <p style={{
                    fontFamily: 'var(--dw-font-ui)',
                    fontSize: 13,
                    color: 'var(--dw-fg-muted)',
                    margin: 0,
                    lineHeight: 1.5,
                  }}>
                    {analysisRunning
                      ? (analysisJob.message ?? ANALYSIS_STAGE_LABELS[analysisJob.stage] ?? 'Working…')
                      : analysisError
                        ? analysisError
                        : 'Validation metrics, disaster assessment and 3D building reconstruction were deferred to keep this step fast. Run them when you need them.'}
                  </p>
                </div>
                {!analysisRunning && !analysisError && (
                  <button
                    onClick={runAnalysis}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 8,
                      height: 38,
                      padding: '0 18px',
                      background: 'var(--dw-panel)',
                      border: '1px solid var(--dw-accent)',
                      borderRadius: 'var(--dw-radius-sm)',
                      fontFamily: 'var(--dw-font-ui)',
                      fontSize: 14,
                      fontWeight: 500,
                      color: 'var(--dw-accent)',
                      cursor: 'pointer',
                      outline: 'none',
                      transition: 'background 120ms ease',
                    }}
                    onFocus={e => {
                      e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                      e.currentTarget.style.outlineOffset = '3px';
                    }}
                    onBlur={e => { e.currentTarget.style.outline = 'none'; }}
                  >
                    <RefreshCw size={16} strokeWidth={1.5} aria-hidden="true" />
                    Run Full Analysis
                  </button>
                )}
                {analysisRunning && (
                  <Loader2 size={18} strokeWidth={1.5} aria-hidden="true"
                    style={{ animation: 'dw-pulse 1.2s ease-in-out infinite', color: 'var(--dw-live)' }} />
                )}
              </div>
              {analysisRunning && typeof analysisJob.progress === 'number' && (
                <div
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(analysisJob.progress)}
                  style={{ height: 4, borderRadius: 2, background: 'var(--dw-rim)', overflow: 'hidden' }}
                >
                  <div style={{
                    height: '100%',
                    width: `${Math.round(analysisJob.progress)}%`,
                    borderRadius: 2,
                    background: 'var(--dw-live)',
                    transition: 'width 600ms ease',
                  }} />
                </div>
              )}
              {analysisRunning && (
                <style>{`@keyframes dw-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
              )}
            </div>
          </section>
        )}

        {/* Result cards — 3 columns (RGB / Depth / DSM) */}
        <section aria-labelledby="layers-heading">
          <p
            id="layers-heading"
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              fontWeight: 600,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-muted)',
              margin: '0 0 14px 0',
            }}
          >
            Output Layers
          </p>

          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 16,
            alignItems: 'flex-start',
          }}>

            {/* RGB Source */}
            <LayerImageCard
              label="RGB"
              sublabel="Source image"
              imageUrl={rgbUrl}
              disabled={false}
              isLoading={false}
            />

            {/* Depth Map */}
            <LayerImageCard
              label="DEPTH"
              sublabel="Monocular depth estimate"
              imageUrl={depthUrl}
              minValue={depthData?.min ?? null}
              maxValue={depthData?.max ?? null}
              units={elevUnits}
              isLoading={isLoading}
              disabled={!isLoading && !depthData}
              disabledReason={fetchState === 'error' ? 'Could not load depth metadata.' : 'Depth map not available.'}
            />

            {/* DSM — shown only for absolute pipeline */}
            {isAbsolute ? (
              <LayerImageCard
                label="DSM"
                sublabel={`Absolute DSM · ${elevUnits}`}
                imageUrl={dsmUrl}
                minValue={dsmData?.min_elevation ?? null}
                maxValue={dsmData?.max_elevation ?? null}
                units={elevUnits}
                isLoading={isLoading}
                disabled={!isLoading && !dsmData}
                disabledReason={fetchState === 'error' ? 'Could not load DSM metadata.' : 'DSM not available.'}
              />
            ) : (
              <LayerImageCard
                label="DSM"
                sublabel="Not available"
                imageUrl={null}
                disabled
                disabledReason="Absolute DSM requires georeferenced GeoTIFF input. This scene used the relative pipeline."
              />
            )}
          </div>
        </section>

        {/* Validation Accuracy Evaluation (§16, §63) */}
        {isAbsolute && (
          <section aria-labelledby="validation-accuracy-heading">
            <h2
              id="validation-accuracy-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 12,
                fontWeight: 600,
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                color: 'var(--dw-fg-muted)',
                margin: '0 0 12px 0',
              }}
            >
              ACCURACY EVALUATION
            </h2>
            <div
              key={`validation-${refreshTick}`}
              style={{
                background: 'var(--dw-panel)',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-md)',
                padding: '20px',
                maxWidth: 760,
              }}
            >
              <MetricsPanel
                available={validation?.available === true && !!validation?.metrics}
                metrics={validation?.metrics ?? null}
                units={validation?.units ?? 'm'}
                reference={reference?.source ?? 'SRTM'}
                sceneTypes={validation?.scene_types ?? null}
                isLoading={valLoading}
              />
            </div>
          </section>
        )}

        {/* Error state — API load failure (§31 Rule 6, §69) */}
        {fetchState === 'error' && (
          <ApiErrorAlert
            error={{
              code: 'SCENE_NOT_FOUND',
              message: 'Could not fetch result metadata from the server.',
              recoverable: true,
            }}
            onRetry={() => {
              setFetchState('loading');
              const id = scene?.scene_id;
              if (id) {
                Promise.all([getDepth(id), isAbsolute ? getDsm(id) : null])
                  .then(([d, m]) => {
                    setDepthData(d);
                    setDsmData(m);
                    setFetchState('done');
                  })
                  .catch(() => setFetchState('error'));
              }
            }}
          />
        )}

        {/* Enter 3D Terrain CTA — task 3.3 */}
        <section aria-label="Proceed to terrain workspace">
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            maxWidth: 480,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <button
                onClick={() => actions.startTerrainLoad()}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 10,
                  height: 44,
                  padding: '0 24px',
                  background: 'var(--dw-accent)',
                  border: '1px solid var(--dw-accent)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 15,
                  fontWeight: 600,
                  color: 'var(--dw-fg-invert)',
                  cursor: 'pointer',
                  letterSpacing: '0.01em',
                  outline: 'none',
                  transition: 'background 120ms ease, border-color 120ms ease',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = '#ffffff';
                  e.currentTarget.style.borderColor = '#ffffff';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'var(--dw-accent)';
                  e.currentTarget.style.borderColor = 'var(--dw-accent)';
                }}
                onFocus={e => {
                  e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                  e.currentTarget.style.outlineOffset = '3px';
                }}
                onBlur={e => { e.currentTarget.style.outline = 'none'; }}
              >
                <Mountain size={18} strokeWidth={1.5} aria-hidden="true" />
                Enter 3D Terrain
              </button>

              <button
                onClick={() => setShowExport(v => !v)}
                aria-expanded={showExport}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  height: 44,
                  padding: '0 20px',
                  background: showExport ? 'var(--dw-surface)' : 'var(--dw-panel)',
                  border: showExport ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 14.5,
                  fontWeight: 500,
                  color: showExport ? 'var(--dw-accent)' : 'var(--dw-fg)',
                  cursor: 'pointer',
                  letterSpacing: '0.01em',
                  outline: 'none',
                  transition: 'background 120ms ease, border-color 120ms ease, color 120ms ease',
                }}
                onFocus={e => {
                  e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                  e.currentTarget.style.outlineOffset = '3px';
                }}
                onBlur={e => { e.currentTarget.style.outline = 'none'; }}
              >
                <Download size={18} strokeWidth={1.5} aria-hidden="true" />
                Export Outputs
              </button>
            </div>

            <p style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13.5,
              color: 'var(--dw-fg-muted)',
              margin: 0,
              lineHeight: 1.5,
            }}>
              Explore the reconstructed terrain, measure elevations, and analyse structures.
            </p>

            {/* Collapsible Export Panel (§27, §66) */}
            {showExport && (
              <div style={{
                marginTop: 8,
                background: 'var(--dw-panel)',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-md)',
                padding: 18,
                maxWidth: 560,
              }}>
                <ExportPanel
                  sceneId={scene?.scene_id}
                  compact={false}
                  onClose={() => setShowExport(false)}
                />
              </div>
            )}
          </div>
        </section>

      </main>

      {/* "Please don't leave the page" popup — shown the moment the
          deferred-analysis job starts (dismissed with the button; the
          beforeunload guard keeps protecting until the job finishes). */}
      {showLeavePopup && analysisRunning && (
        <div
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="stay-popup-title"
          onClick={() => setShowLeavePopup(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.55)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: 24,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-accent)',
              borderRadius: 'var(--dw-radius-md)',
              padding: '24px 28px',
              maxWidth: 440,
              width: '100%',
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <AlertTriangle size={20} strokeWidth={1.5} aria-hidden="true"
                style={{ color: 'var(--dw-live)', flexShrink: 0 }} />
              <p
                id="stay-popup-title"
                style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 16,
                  fontWeight: 600,
                  color: 'var(--dw-fg)',
                  margin: 0,
                }}
              >
                Please don&rsquo;t leave the page
              </p>
            </div>
            <p style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 14,
              color: 'var(--dw-fg-muted)',
              margin: 0,
              lineHeight: 1.55,
            }}>
              Full analysis is running on the server. Stay on this page until
              it completes — your validation metrics, disaster assessment and
              3D building results will appear here automatically. Leaving or
              reloading now would interrupt the live progress tracking.
            </p>
            <button
              onClick={() => setShowLeavePopup(false)}
              autoFocus
              style={{
                alignSelf: 'flex-end',
                height: 36,
                padding: '0 18px',
                background: 'var(--dw-accent)',
                border: '1px solid var(--dw-accent)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 14,
                fontWeight: 600,
                color: 'var(--dw-fg-invert)',
                cursor: 'pointer',
                outline: 'none',
              }}
              onFocus={e => {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '3px';
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              Got it, I&rsquo;ll stay
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/** Compact labelled field for the metadata strip */
function MetaField({ label, value }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 11,
        letterSpacing: '0.07em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-muted)',
        fontWeight: 600,
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 14,
        fontWeight: 500,
        color: 'var(--dw-fg)',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        maxWidth: 260,
      }}>
        {value}
      </span>
    </div>
  );
}

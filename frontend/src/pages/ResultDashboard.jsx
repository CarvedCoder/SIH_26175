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
import { useEffect, useState } from 'react';
import { Mountain, Download } from 'lucide-react';
import Header from '../components/common/Header.jsx';
import LayerImageCard from '../components/common/LayerImageCard.jsx';
import MetricsPanel from '../components/Validation/MetricsPanel.jsx';
import ExportPanel from '../components/Export/ExportPanel.jsx';
import PartialResultBanner from '../components/common/PartialResultBanner.jsx';
import ApiErrorAlert from '../components/common/ApiErrorAlert.jsx';
import { useValidation } from '../hooks/useValidation.js';
import { useApp } from '../store/appStore.jsx';
import { getDepth, getDsm } from '../api/results.js';

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
  const [fetchState, setFetchState] = useState(/** @type {FetchState} */ ('loading'));
  const [showExport, setShowExport] = useState(false);

  // Capability flags from ResultsMeta
  const elevationMode = results?.elevation_mode ?? 'relative';
  const isAbsolute    = elevationMode === 'absolute';

  const { validation, reference, isLoading: valLoading } = useValidation(scene?.scene_id, isAbsolute);

  useEffect(() => {
    if (!scene?.scene_id) return;

    let cancelled = false;
    setFetchState('loading');

    async function fetchLayerMeta() {
      try {
        const promises = [getDepth(scene.scene_id)];
        if (isAbsolute) promises.push(getDsm(scene.scene_id));
        else promises.push(Promise.resolve(null));

        const [depth, dsm] = await Promise.all(promises);
        if (cancelled) return;
        setDepthData(depth);
        setDsmData(dsm);
        setFetchState('done');
      } catch {
        if (!cancelled) setFetchState('error');
      }
    }

    fetchLayerMeta();
    return () => { cancelled = true; };
  }, [scene?.scene_id, isAbsolute]);

  /** Units label from D10: 'm' for absolute, 'scene units' for relative */
  const elevUnits = isAbsolute ? 'm' : 'scene units';

  /**
   * The backend exposes depth and DSM preview images via their `url` / `download_url`.
   * RGB comes from the scene source — there is no dedicated RGB endpoint in this phase,
   * so we point to the scene's upload if the backend serves it, or show a placeholder.
   * When the terrain texture URL is available in results, prefer that.
   */
  const rgbUrl  = results?.preview_url ?? null;
  const depthUrl = depthData?.url ?? null;
  const dsmUrl   = dsmData?.download_url ?? null;

  const isLoading = fetchState === 'loading';

  // Scene metadata line items
  const meta = scene ? [
    { label: 'SOURCE', value: scene.filename },
    { label: 'DIMENSIONS', value: `${scene.width} × ${scene.height} px` },
    { label: 'FORMAT', value: scene.format },
    { label: 'GEOREF', value: scene.georeferenced ? 'YES' : 'NO' },
    scene.crs ? { label: 'CRS', value: scene.crs } : null,
    { label: 'PIPELINE', value: isAbsolute ? 'Absolute DSM' : 'Relative DSM' },
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
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <h1
              id="dashboard-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 15,
                fontWeight: 500,
                color: 'var(--dw-fg)',
                margin: 0,
              }}
            >
              Processing Complete
            </h1>
            <p style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
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
              padding: '12px 16px',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '12px 32px',
              alignItems: 'center',
            }}>
              {meta.map(({ label, value }) => (
                <MetaField key={label} label={label} value={value} />
              ))}
            </div>
          </section>
        )}

        {/* Pipeline capability evaluation banner (§38) */}
        <section aria-label="Pipeline capability evaluation">
          <PartialResultBanner
            format={scene?.format ?? (isAbsolute ? 'GeoTIFF' : 'PNG')}
            georeferenced={isAbsolute}
            elevationMode={elevationMode}
            hasReference={isAbsolute && results?.reference_source != null}
            compact={false}
          />
        </section>

        {/* Result cards — 3 columns (RGB / Depth / DSM) */}
        <section aria-labelledby="layers-heading">
          <p
            id="layers-heading"
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              fontWeight: 500,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-muted)',
              margin: '0 0 12px 0',
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
                fontSize: 10,
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                color: 'var(--dw-fg-ghost)',
                margin: '0 0 10px 0',
              }}
            >
              ACCURACY EVALUATION
            </h2>
            <div style={{
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-md)',
              padding: '16px',
              maxWidth: 720,
            }}>
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
            gap: 8,
            maxWidth: 400,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <button
                onClick={() => actions.startTerrainLoad()}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  height: 40,
                  padding: '0 20px',
                  background: 'var(--dw-accent)',
                  border: '1px solid var(--dw-accent)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 13,
                  fontWeight: 500,
                  color: '#fff',
                  cursor: 'pointer',
                  letterSpacing: '0.01em',
                  outline: 'none',
                  transition: 'background 120ms ease, border-color 120ms ease',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = '#2563eb';
                  e.currentTarget.style.borderColor = '#2563eb';
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
                <Mountain size={15} strokeWidth={1.5} aria-hidden="true" />
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
                  height: 40,
                  padding: '0 16px',
                  background: showExport ? 'var(--dw-surface)' : 'var(--dw-panel)',
                  border: showExport ? '1px solid var(--dw-accent)' : '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 13,
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
                <Download size={15} strokeWidth={1.5} aria-hidden="true" />
                Export Outputs
              </button>
            </div>

            <p style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11,
              color: 'var(--dw-fg-ghost)',
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
                padding: 16,
                maxWidth: 540,
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
    </div>
  );
}

/** Compact labelled field for the metadata strip */
function MetaField({ label, value }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 9,
        letterSpacing: '0.07em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-ghost)',
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 12,
        color: 'var(--dw-fg)',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        maxWidth: 240,
      }}>
        {value}
      </span>
    </div>
  );
}

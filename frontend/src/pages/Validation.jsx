/**
 * DepthWizard — Validation Page (Phase 12)
 *
 * Dedicated validation assessment and reference comparison dashboard.
 * Evaluates reconstructed DSM against aligned ground-truth elevation DEM (SRTM/ALOS).
 *
 * Spec §16, §17, §62, §63, §64.
 * DESIGN.md: Operate mode, scientific/geospatial mission-control visual register.
 */
import { useState } from 'react';
import { useApp } from '../store/appStore.jsx';
import Header from '../components/common/Header.jsx';
import ReferenceComparison from '../components/Validation/ReferenceComparison.jsx';
import { Mountain, ArrowLeft, ShieldCheck, FileSpreadsheet } from 'lucide-react';

export default function ValidationPage({ onBack }) {
  const { state, actions } = useApp();
  const scene = state.scene;
  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const [activeLayer, setActiveLayer] = useState('dsm');

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        padding: '32px 24px 64px',
        maxWidth: 1200,
        width: '100%',
        margin: '0 auto',
        gap: 28,
      }}>
        {/* Navigation & Header */}
        <section aria-labelledby="validation-heading">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            {onBack && (
              <button
                onClick={onBack}
                aria-label="Back to results"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  height: 28,
                  padding: '0 8px',
                  background: 'transparent',
                  border: '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  color: 'var(--dw-fg-muted)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 12,
                  cursor: 'pointer',
                  outline: 'none',
                }}
                onFocus={e => {
                  e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                  e.currentTarget.style.outlineOffset = '2px';
                }}
                onBlur={e => { e.currentTarget.style.outline = 'none'; }}
              >
                <ArrowLeft size={13} strokeWidth={1.5} />
                Results
              </button>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ShieldCheck size={16} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
              <span style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 10,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--dw-accent)',
                fontWeight: 600,
              }}>
                Accuracy Assessment
              </span>
            </div>
          </div>

          <h1
            id="validation-heading"
            style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 18,
              fontWeight: 600,
              color: 'var(--dw-fg)',
              margin: '0 0 6px 0',
              letterSpacing: '-0.01em',
            }}
          >
            Reference Elevation Comparison
          </h1>
          <p style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13,
            color: 'var(--dw-fg-muted)',
            margin: 0,
            maxWidth: 680,
            lineHeight: 1.45,
          }}>
            Cross-validation of the model-derived Digital Surface Model against aligned reference topography (SRTM 30m / ALOS). Evaluates spatial RMSE, mean bias, and relief correlation.
          </p>
        </section>

        {/* Comparison Suite */}
        <section aria-label="Reference comparison controls and metrics">
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(320px, 420px) 1fr',
            gap: 24,
            alignItems: 'start',
          }}>
            {/* Left Column: Reference Comparison & Accuracy Metrics */}
            <div style={{
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-md)',
              padding: '18px 16px',
            }}>
              <ReferenceComparison
                sceneId={scene?.scene_id}
                isGeoreferenced={isAbsolute}
                activeLayer={activeLayer}
                onSelectLayer={setActiveLayer}
              />
            </div>

            {/* Right Column: Visual Comparison Display & Actions */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 16,
            }}>
              {/* Layer Information Banner */}
              <div style={{
                background: 'var(--dw-panel)',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-md)',
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{
                    fontFamily: 'var(--dw-font-ui)',
                    fontSize: 11,
                    fontWeight: 500,
                    color: 'var(--dw-fg)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}>
                    Active View: {activeLayer === 'dsm' ? 'Estimated DSM' : activeLayer === 'reference_dem' ? 'Reference DEM (SRTM)' : 'Difference / Residual Map'}
                  </span>
                  <span style={{
                    fontFamily: 'var(--dw-font-data)',
                    fontSize: 11,
                    color: 'var(--dw-accent)',
                  }}>
                    {isAbsolute ? 'EPSG:4326 • Calibrated' : 'Relative Scale'}
                  </span>
                </div>

                <div style={{
                  height: 320,
                  background: 'var(--dw-surface)',
                  border: '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 8,
                    textAlign: 'center',
                    padding: 20,
                  }}>
                    <Mountain size={28} strokeWidth={1.5} color="var(--dw-accent)" />
                    <span style={{
                      fontFamily: 'var(--dw-font-ui)',
                      fontSize: 13,
                      fontWeight: 500,
                      color: 'var(--dw-fg)',
                    }}>
                      Interactive 3D Comparison Available
                    </span>
                    <span style={{
                      fontFamily: 'var(--dw-font-ui)',
                      fontSize: 11,
                      color: 'var(--dw-fg-muted)',
                      maxWidth: 380,
                    }}>
                      Switch layers directly in the 3D Terrain Workspace to inspect differences with vert-exaggeration and contour overlays.
                    </span>
                    <button
                      onClick={() => actions.startTerrainLoad()}
                      style={{
                        marginTop: 6,
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                        height: 32,
                        padding: '0 14px',
                        background: 'var(--dw-accent)',
                        border: '1px solid var(--dw-accent)',
                        borderRadius: 'var(--dw-radius-sm)',
                        color: '#fff',
                        fontFamily: 'var(--dw-font-ui)',
                        fontSize: 12,
                        cursor: 'pointer',
                        outline: 'none',
                      }}
                      onFocus={e => {
                        e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                        e.currentTarget.style.outlineOffset = '2px';
                      }}
                      onBlur={e => { e.currentTarget.style.outline = 'none'; }}
                    >
                      <Mountain size={13} strokeWidth={1.5} />
                      Enter 3D Terrain
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

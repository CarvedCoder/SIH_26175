/**
 * DepthWizard — AnalysisPanel (Phase 10 & Phase 12)
 *
 * Collapsible side analysis panel (§25, §16, §17).
 * Contains context-sensitive information:
 *
 * 1. Default context:
 *    - Tab "Overview":
 *      SCENE (Image, Resolution, Mode, Reference, Georeferenced)
 *      MODEL (Depth Model, Status, Calibration, Segments)
 *    - Tab "Validation" (§16, §17, §63, §64):
 *      ReferenceComparison suite:
 *        - ComparisonView (Estimated DSM / Reference DEM / Difference Map)
 *        - MetricsPanel (RMSE, MAE, Correlation)
 *        - Ground truth reference information
 *
 * 2. When user selects a point (context: location):
 *    SELECTED LOCATION:
 *      Elevation: 142.63 m
 *      Slope: 12.4°
 *      Position: X / Z
 *
 * 3. When user inspects a structure (context: structure):
 *    SELECTED STRUCTURE:
 *      Structure ID: STR-042
 *      Height: 18.4 m
 *      Ground: 126.2 m
 *      Top: 144.6 m
 *
 * DESIGN.md:
 *   - 280px width, right-side dock
 *   - Collapsible with 200ms ease-out GPU transform
 *   - Data values in monospace data face, labels in UI face
 *   - Uppercase compact group headers
 *   - Thin 1px --dw-rim dividers
 *   - No card shadows
 *
 * Spec §16, §17, §25, §63, §64, §D05, §D10.
 */
import { useState, useEffect } from 'react';
import { useApp } from '../../store/appStore.jsx';
import ReferenceComparison from '../Validation/ReferenceComparison.jsx';
import DetailMode from '../Analysis/DetailMode.jsx';
import ScenarioSwitcher from '../Analysis/ScenarioSwitcher.jsx';
import DisasterAssessmentPanel from '../Analysis/DisasterAssessmentPanel.jsx';
import {
  ChevronRight,
  Info,
  MapPin,
  Building2,
  X,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';

/**
 * @param {{
 *   open?: boolean,
 *   onToggle?: () => void,
 *   selectedLocation?: { x: number, z: number, elevation: number, slope?: number } | null,
 *   selectedStructure?: { id: string, ground: number, top: number, height: number } | null,
 *   onClearSelection?: () => void,
 *   activeLayer?: string,
 *   onSelectLayer?: (layerId: string) => void,
 *   panelTab?: 'overview' | 'validation',
 *   onSelectTab?: (tab: 'overview' | 'validation') => void,
 *   activeTool?: string,
 *   refineBbox?: { x_min: number, y_min: number, x_max: number, y_max: number } | null,
 *   onStartRegionSelect?: () => void,
 *   onClearRefineBbox?: () => void,
 *   isSelectingRegion?: boolean,
 *   onRefineComplete?: (res: any) => void,
 *   scenario?: 'exploration' | 'disaster',
 *   onSelectScenario?: (scenario: 'exploration' | 'disaster') => void,
 * }} props
 */
export default function AnalysisPanel({
  open = true,
  onToggle,
  selectedLocation = null,
  selectedStructure = null,
  onClearSelection,
  activeLayer = 'rgb',
  onSelectLayer,
  panelTab,
  onSelectTab,
  activeTool = 'none',
  refineBbox = null,
  onStartRegionSelect,
  onClearRefineBbox,
  isSelectingRegion = false,
  onRefineComplete,
  scenario = 'exploration',
  onSelectScenario,
}) {
  const { state } = useApp();

  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const unitLabel  = isAbsolute ? 'm' : 'scene units';

  // Internal scenario state if not controlled externally
  const [internalScenario, setInternalScenario] = useState('exploration');
  const currentScenario = onSelectScenario ? scenario : internalScenario;
  const setScenario = onSelectScenario ?? setInternalScenario;

  // Internal tab state if not controlled externally
  const [internalTab, setInternalTab] = useState('overview');
  const currentTab = panelTab ?? internalTab;

  const setTab = (tab) => {
    if (onSelectTab) onSelectTab(tab);
    else setInternalTab(tab);
  };

  // Automatically switch to validation tab when user chooses a comparison layer
  useEffect(() => {
    if (activeLayer === 'error' || activeLayer === 'reference_dem') {
      setTab('validation');
    }
  }, [activeLayer]);

  // Determine current context view
  const isStructure = !!selectedStructure;
  const isLocation  = !isStructure && !!selectedLocation;
  const isRefine    = !isStructure && !isLocation && (activeTool === 'refine' || !!refineBbox);

  return (
    <aside
      aria-label="Side analysis panel"
      aria-expanded={open}
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        bottom: 0,
        width: 'min(var(--dw-panel-w, 320px), 100vw)',
        maxWidth: '100vw',
        background: 'var(--dw-panel)',
        borderLeft: '1px solid var(--dw-rim)',
        zIndex: 15,
        display: 'flex',
        flexDirection: 'column',
        transform: open ? 'translateX(0)' : 'translateX(100%)',
        transition: 'transform 200ms ease-out',
        pointerEvents: open ? 'auto' : 'none',
      }}
    >
      {/* Panel Header */}
      <div style={{
        height: 48,
        padding: '0 14px',
        borderBottom: '1px solid var(--dw-rim)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {isStructure ? (
            <Building2 size={16} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          ) : isLocation ? (
            <MapPin size={16} strokeWidth={1.5} color="var(--dw-probe)" aria-hidden="true" />
          ) : isRefine ? (
            <Sparkles size={16} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          ) : currentTab === 'validation' ? (
            <ShieldCheck size={16} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          ) : (
            <Info size={16} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          )}
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13.5,
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg)',
            fontWeight: 600,
          }}>
            {isStructure
              ? 'Selected Structure'
              : isLocation
              ? 'Selected Location'
              : isRefine
              ? 'Detail Refinement'
              : currentTab === 'validation'
              ? 'Validation & Accuracy'
              : 'Scene Analysis'}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {(isStructure || isLocation || (isRefine && refineBbox)) && (
            <button
              onClick={() => {
                if (isStructure || isLocation) onClearSelection?.();
                if (isRefine) onClearRefineBbox?.();
              }}
              aria-label="Back to overview"
              title="Reset"
              style={{
                height: 28,
                padding: '0 8px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                background: 'none',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 12,
                color: 'var(--dw-fg-muted)',
                cursor: 'pointer',
                outline: 'none',
              }}
              onFocus={e => {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '1px';
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              <X size={12} strokeWidth={1.5} aria-hidden="true" />
              Reset
            </button>
          )}

          {onToggle && (
            <button
              onClick={onToggle}
              aria-label={open ? 'Collapse analysis panel' : 'Expand analysis panel'}
              title={open ? 'Collapse' : 'Expand'}
              style={{
                width: 28,
                height: 28,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'none',
                border: 'none',
                borderRadius: 'var(--dw-radius-sm)',
                color: 'var(--dw-fg-muted)',
                cursor: 'pointer',
                outline: 'none',
              }}
              onFocus={e => {
                e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                e.currentTarget.style.outlineOffset = '1px';
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
            >
              <ChevronRight size={16} strokeWidth={1.5} aria-hidden="true" />
            </button>
          )}
        </div>
      </div>

      {/* Overview / Validation Tab Switcher (when not inspecting specific point/structure/refine) */}
      {!isStructure && !isLocation && !isRefine && (
        <div
          role="tablist"
          aria-label="Analysis sections"
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            borderBottom: '1px solid var(--dw-rim)',
            background: 'var(--dw-surface)',
          }}
        >
          <button
            role="tab"
            aria-selected={currentTab === 'overview'}
            onClick={() => setTab('overview')}
            style={{
              height: 38,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: currentTab === 'overview' ? 'var(--dw-panel)' : 'transparent',
              border: 'none',
              borderBottom: currentTab === 'overview' ? '2px solid var(--dw-accent)' : '2px solid transparent',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12.5,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              fontWeight: currentTab === 'overview' ? 600 : 400,
              color: currentTab === 'overview' ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
              cursor: 'pointer',
              outline: 'none',
              transition: 'color 120ms ease, background 120ms ease',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '-2px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            Overview
          </button>

          <button
            role="tab"
            aria-selected={currentTab === 'validation'}
            onClick={() => setTab('validation')}
            style={{
              height: 38,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: currentTab === 'validation' ? 'var(--dw-panel)' : 'transparent',
              border: 'none',
              borderBottom: currentTab === 'validation' ? '2px solid var(--dw-accent)' : '2px solid transparent',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12.5,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              fontWeight: currentTab === 'validation' ? 600 : 400,
              color: currentTab === 'validation' ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
              cursor: 'pointer',
              outline: 'none',
              transition: 'color 120ms ease, background 120ms ease',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '-2px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            Validation
          </button>
        </div>
      )}

      {/* Scrollable Content Body */}
      <div style={{
        flex: 1,
        padding: '16px 14px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 18,
      }}>
        {/* Context 1: Structure Selected */}
        {isStructure && (
          <section aria-labelledby="selected-structure-heading">
            <h2
              id="selected-structure-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 11.5,
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--dw-fg-ghost)',
                margin: '0 0 10px 0',
              }}
            >
              SELECTED STRUCTURE
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
              <DataRow label="Structure ID" value={selectedStructure.id ?? 'STR-042'} />
              <DataRow
                label="Estimated Height"
                value={`${selectedStructure.height?.toFixed(1) ?? '—'} ${unitLabel}`}
                highlight
              />
              <DataRow
                label="Ground Elevation"
                value={`${selectedStructure.ground?.toFixed(1) ?? '—'} ${unitLabel}`}
              />
              <DataRow
                label="Top Elevation"
                value={`${selectedStructure.top?.toFixed(1) ?? '—'} ${unitLabel}`}
              />
            </div>
          </section>
        )}

        {/* Context 2: Location Point Selected */}
        {isLocation && (
          <section aria-labelledby="selected-location-heading">
            <h2
              id="selected-location-heading"
              style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 11.5,
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--dw-fg-ghost)',
                margin: '0 0 10px 0',
              }}
            >
              SELECTED LOCATION
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
              <DataRow
                label="Elevation"
                value={`${selectedLocation.elevation?.toFixed(2) ?? '—'} ${unitLabel}`}
                highlight
              />
              <DataRow
                label="Slope"
                value={`${selectedLocation.slope != null ? selectedLocation.slope.toFixed(1) : '—'}°`}
              />
              <DataRow
                label="Terrain Coordinates"
                value={`${selectedLocation.x?.toFixed(3) ?? '0.000'}, ${selectedLocation.z?.toFixed(3) ?? '0.000'}`}
              />
            </div>
          </section>
        )}

        {/* Context 3: Detail Refinement Mode (§18, §65) */}
        {isRefine && (
          <DetailMode
            sceneId={state.scene?.scene_id}
            selectedBbox={refineBbox}
            onStartSelection={onStartRegionSelect}
            onClearBbox={onClearRefineBbox}
            isSelecting={isSelectingRegion}
            onRefineComplete={onRefineComplete}
            compact={true}
          />
        )}

        {/* Context 4: Default — Validation Tab */}
        {!isStructure && !isLocation && !isRefine && currentTab === 'validation' && (
          <ReferenceComparison
            sceneId={state.scene?.scene_id}
            isGeoreferenced={isAbsolute}
            activeLayer={activeLayer}
            onSelectLayer={onSelectLayer ?? (() => {})}
            compact={true}
          />
        )}

        {/* Context 5: Default — Overview Tab (Scene & Model Info OR Disaster Assessment) */}
        {!isStructure && !isLocation && !isRefine && currentTab === 'overview' && (
          <>
            {/* Operational Scenario Switcher (§23) */}
            <ScenarioSwitcher
              scenario={currentScenario}
              onSelectScenario={setScenario}
            />

            <div style={{ height: 1, background: 'var(--dw-rim)', margin: '4px 0' }} />

            {/* Disaster Assessment Preset View (§23) */}
            {currentScenario === 'disaster' ? (
              <DisasterAssessmentPanel
                terrainMeta={state.terrain}
                selectedLocation={selectedLocation}
                selectedStructure={selectedStructure}
                isAbsolute={isAbsolute}
                unitLabel={unitLabel}
                onSelectLayer={onSelectLayer}
                onOpenValidation={() => setTab('validation')}
              />
            ) : (
              <>
                {/* SCENE Section */}
                <section aria-labelledby="scene-info-heading">
                  <h2
                    id="scene-info-heading"
                    style={{
                      fontFamily: 'var(--dw-font-ui)',
                      fontSize: 11.5,
                      fontWeight: 600,
                      letterSpacing: '0.08em',
                      textTransform: 'uppercase',
                      color: 'var(--dw-fg-ghost)',
                      margin: '0 0 10px 0',
                    }}
                  >
                    SCENE
                  </h2>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
                    <DataRow
                      label="Image"
                      value={state.scene?.filename ?? state.scene?.image_name ?? 'scene_042.tif'}
                    />
                    <DataRow
                      label="Resolution"
                      value={
                        state.scene?.width && state.scene?.height
                          ? `${state.scene.width} × ${state.scene.height}`
                          : '4096 × 4096'
                      }
                    />
                    <DataRow
                      label="Mode"
                      value={isAbsolute ? 'Absolute DSM' : 'Relative DSM'}
                      highlight={isAbsolute}
                    />
                    <DataRow
                      label="Reference"
                      value={state.results?.reference_source ?? (isAbsolute ? 'SRTM' : 'None')}
                    />
                    <DataRow
                      label="Georeferenced"
                      value={state.scene?.is_georeferenced ? 'Yes (EPSG:4326)' : 'No (Relative)'}
                    />
                  </div>
                </section>

                <div style={{ height: 1, background: 'var(--dw-rim)', margin: '4px 0' }} />

                {/* MODEL Section */}
                <section aria-labelledby="model-info-heading">
                  <h2
                    id="model-info-heading"
                    style={{
                      fontFamily: 'var(--dw-font-ui)',
                      fontSize: 11.5,
                      fontWeight: 600,
                      letterSpacing: '0.08em',
                      textTransform: 'uppercase',
                      color: 'var(--dw-fg-ghost)',
                      margin: '0 0 10px 0',
                    }}
                  >
                    MODEL
                  </h2>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
                    <DataRow
                      label="Depth Model"
                      value={state.scene?.model ?? 'Depth Anything V2'}
                    />
                    <DataRow
                      label="Status"
                      value={
                        state.status === 'TERRAIN_READY' || state.status === 'ANALYSIS'
                          ? 'Complete'
                          : state.status
                      }
                      accent
                    />
                    <DataRow
                      label="Elevation Calibration"
                      value={isAbsolute ? 'Metric (Ground GCPs)' : 'Relative (Estimated)'}
                    />
                    <DataRow
                      label="Terrain Segments"
                      value="256 × 256 (Hi-Res)"
                    />
                  </div>
                </section>
              </>
            )}
          </>
        )}
      </div>
    </aside>
  );
}

/** Two-column key-value row formatted per DESIGN.md */
function DataRow({ label, value, highlight = false, accent = false }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      gap: 8,
    }}>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 13,
        color: 'var(--dw-fg-muted)',
        flexShrink: 0,
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 14,
        fontWeight: highlight ? 600 : 400,
        color: accent
          ? 'var(--dw-confirm)'
          : highlight
          ? 'var(--dw-accent)'
          : 'var(--dw-fg)',
        textAlign: 'right',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}>
        {value}
      </span>
    </div>
  );
}

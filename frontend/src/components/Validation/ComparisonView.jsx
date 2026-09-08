/**
 * DepthWizard — ComparisonView (Phase 12, Task 12.4)
 *
 * Provides a 3-way toggle or slider between:
 * - Estimated DSM
 * - Reference DEM (Ground truth e.g. SRTM)
 * - Difference Map (Elevation Error)
 *
 * Includes:
 * - Diverging error colormap scale legend (Blue = negative delta, Red = positive delta)
 * - Min/Max error bounds from backend /validation/error-map (§64)
 * - Blend opacity slider for overlay comparisons
 *
 * Spec §16, §17, §64.
 */
import { useState } from 'react';
import { Layers, ArrowLeftRight, HelpCircle } from 'lucide-react';

const COMPARISON_LAYERS = [
  {
    id: 'dsm',
    label: 'Estimated DSM',
    desc: 'AI reconstructed surface',
  },
  {
    id: 'reference_dem',
    label: 'Reference DEM',
    desc: 'Ground-truth elevation (SRTM)',
    requiresReference: true,
  },
  {
    id: 'error',
    label: 'Difference Map',
    desc: 'Elevation residual delta',
    requiresReference: true,
  },
];

/**
 * @param {{
 *   activeLayer: string,
 *   onSelectLayer: (layerId: string) => void,
 *   isAbsolute?: boolean,
 *   disabled?: boolean,
 *   errorMapMeta?: { url?: string, units?: string, min_error?: number, max_error?: number } | null,
 *   referenceMeta?: { available?: boolean, source?: string, resolution?: string, crs?: string } | null,
 *   onOpacityChange?: (opacity: number) => void,
 * }} props
 */
export default function ComparisonView({
  activeLayer = 'dsm',
  onSelectLayer,
  isAbsolute = true,
  disabled = false,
  errorMapMeta = null,
  referenceMeta = null,
  onOpacityChange,
}) {
  const [blendOpacity, setBlendOpacity] = useState(1.0);
  const [showHelp, setShowHelp] = useState(false);

  const isRefAvailable = isAbsolute && referenceMeta?.available !== false;

  const minErr = errorMapMeta?.min_error != null ? errorMapMeta.min_error : -12.0;
  const maxErr = errorMapMeta?.max_error != null ? errorMapMeta.max_error : 18.0;
  const errUnits = errorMapMeta?.units ?? (isAbsolute ? 'm' : 'scene units');

  const handleOpacitySlider = (val) => {
    setBlendOpacity(val);
    onOpacityChange?.(val);
  };

  return (
    <div
      role="region"
      aria-label="Reference Comparison View"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        width: '100%',
      }}
    >
      {/* Group Title */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <ArrowLeftRight size={15} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11.5,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            fontWeight: 600,
          }}>
            Comparison Mode (§16, §17)
          </span>
        </div>

        <button
          onClick={() => setShowHelp(h => !h)}
          aria-label="Toggle comparison help explanation"
          title="How difference mapping works"
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: 3,
            display: 'flex',
            alignItems: 'center',
            color: showHelp ? 'var(--dw-accent)' : 'var(--dw-fg-ghost)',
            outline: 'none',
          }}
          onFocus={e => {
            e.currentTarget.style.outline = '2px solid var(--dw-accent)';
            e.currentTarget.style.outlineOffset = '2px';
          }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        >
          <HelpCircle size={15} strokeWidth={1.5} />
        </button>
      </div>

      {/* Helpful explanation when toggled */}
      {showHelp && (
        <div style={{
          padding: '10px 12px',
          background: 'var(--dw-surface)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
          fontSize: 13,
          fontFamily: 'var(--dw-font-ui)',
          color: 'var(--dw-fg-muted)',
          lineHeight: 1.5,
        }}>
          Compare reconstructed surface against known ground-truth. Difference Map computes{' '}
          <code style={{ fontFamily: 'var(--dw-font-data)', color: 'var(--dw-fg)' }}>Δz = z_model - z_ref</code>.{' '}
          Cool blue marks under-estimation (e.g. terrain valleys, occlusions); warm red indicates over-estimation (e.g. buildings, tree canopies).
        </div>
      )}

      {/* 3-Way Mode Switcher */}
      <div
        role="group"
        aria-label="Comparison layer options"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 4,
          background: 'var(--dw-surface)',
          padding: 3,
          borderRadius: 'var(--dw-radius-sm)',
          border: '1px solid var(--dw-rim)',
        }}
      >
        {COMPARISON_LAYERS.map(layer => {
          const isSelected = activeLayer === layer.id;
          const isDisabled = disabled || (layer.requiresReference && !isRefAvailable);

          return (
            <button
              key={layer.id}
              onClick={() => !isDisabled && onSelectLayer(layer.id)}
              disabled={isDisabled}
              aria-pressed={isSelected}
              title={
                isDisabled
                  ? 'Reference data unavailable for this scene'
                  : layer.desc
              }
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 3,
                padding: '8px 6px',
                height: 48,
                background: isSelected ? 'var(--dw-panel)' : 'transparent',
                border: isSelected ? '1px solid var(--dw-accent)' : '1px solid transparent',
                borderRadius: 'var(--dw-radius-sm)',
                cursor: isDisabled ? 'not-allowed' : 'pointer',
                opacity: isDisabled ? 0.35 : 1,
                outline: 'none',
                transition: 'border-color 120ms ease, background 120ms ease',
              }}
              onFocus={e => {
                if (!isDisabled) {
                  e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                  e.currentTarget.style.outlineOffset = '1px';
                }
              }}
              onBlur={e => { e.currentTarget.style.outline = 'none'; }}
              onMouseEnter={e => {
                if (!isDisabled && !isSelected) {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
                }
              }}
              onMouseLeave={e => {
                if (!isDisabled && !isSelected) {
                  e.currentTarget.style.background = 'transparent';
                }
              }}
            >
              <span style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 13,
                fontWeight: isSelected ? 600 : 400,
                color: isSelected ? 'var(--dw-accent)' : 'var(--dw-fg)',
                whiteSpace: 'nowrap',
              }}>
                {layer.label}
              </span>
              <span style={{
                fontFamily: 'var(--dw-font-data)',
                fontSize: 11,
                color: 'var(--dw-fg-muted)',
              }}>
                {layer.id === 'dsm' ? 'Est.' : layer.id === 'reference_dem' ? 'Ref.' : 'Δz'}
              </span>
            </button>
          );
        })}
      </div>

      {/* Difference Map Legend & Bounds (§17, §64) */}
      {activeLayer === 'error' && isRefAvailable && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          padding: '12px',
          background: 'var(--dw-surface)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11.5,
              fontWeight: 600,
              letterSpacing: '0.05em',
              textTransform: 'uppercase',
              color: 'var(--dw-fg-ghost)',
            }}>
              Residual Elevation Scale
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 11,
              color: 'var(--dw-fg-muted)',
            }}>
              Diverging
            </span>
          </div>

          {/* Diverging Colormap Bar */}
          <div
            role="img"
            aria-label={`Error scale from ${minErr} to ${maxErr} ${errUnits}`}
            style={{
              height: 12,
              width: '100%',
              borderRadius: 2,
              border: '1px solid var(--dw-rim)',
              background: 'linear-gradient(to right, #2563eb, #3b82f6, #64748b, #ef4444, #dc2626)',
            }}
          />

          {/* Numeric Range Labels */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontFamily: 'var(--dw-font-data)',
            fontSize: 12,
            color: 'var(--dw-fg-muted)',
          }}>
            <span style={{ color: '#60a5fa' }}>{minErr > 0 ? `+${minErr}` : minErr} {errUnits}</span>
            <span style={{ color: 'var(--dw-fg-ghost)' }}>0.0 {errUnits}</span>
            <span style={{ color: '#f87171' }}>{maxErr > 0 ? `+${maxErr}` : maxErr} {errUnits}</span>
          </div>

          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 11,
            fontFamily: 'var(--dw-font-ui)',
            color: 'var(--dw-fg-ghost)',
          }}>
            <span>Under-estimated</span>
            <span>Accurate</span>
            <span>Over-estimated</span>
          </div>
        </div>
      )}

      {/* Reference DEM Information Banner */}
      {activeLayer === 'reference_dem' && isRefAvailable && (
        <div style={{
          padding: '10px 12px',
          background: 'var(--dw-surface)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 13, color: 'var(--dw-fg-muted)' }}>
            Reference Source
          </span>
          <span style={{ fontFamily: 'var(--dw-font-data)', fontSize: 13, color: 'var(--dw-accent)' }}>
            {referenceMeta?.source ?? 'SRTM'} ({referenceMeta?.resolution ?? '30m'})
          </span>
        </div>
      )}

      {/* Opacity / Blend Slider (Optional enhancement for overlay comparison) */}
      {onOpacityChange && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
          paddingTop: 4,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 11.5,
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--dw-fg-ghost)',
            }}>
              Overlay Blend
            </span>
            <span style={{
              fontFamily: 'var(--dw-font-data)',
              fontSize: 12.5,
              color: 'var(--dw-fg)',
            }}>
              {Math.round(blendOpacity * 100)}%
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={blendOpacity}
            onChange={e => handleOpacitySlider(parseFloat(e.target.value))}
            aria-label="Comparison overlay blend opacity"
            style={{
              width: '100%',
              cursor: 'pointer',
              accentColor: 'var(--dw-accent)',
            }}
          />
        </div>
      )}
    </div>
  );
}

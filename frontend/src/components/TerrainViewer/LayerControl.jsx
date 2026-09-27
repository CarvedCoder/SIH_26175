/**
 * DepthWizard — LayerControl (Phase 8, task 8.1 + 8.2)
 *
 * Radio list for switching terrain texture layers.
 * Pulls available_layers from ResultsMeta in app state.
 * Disables unavailable layers with tooltip explaining why.
 *
 * Available layers from backend: ['rgb', 'depth', 'dsm', 'reference_dem', 'error', 'slope']
 *
 * On selection:
 *   - Calls onLayerChange(layerId) which the parent (TerrainWorkspace) uses to swap
 *     the terrain texture/colormap (task 8.2)
 *   - Camera and minimap state are NOT reset (§31 Rule 5)
 *
 * DESIGN.md:
 *   - Radio group, no cards — labelled list with left indicator dot
 *   - --dw-accent dot when active, --dw-fg-ghost when inactive
 *   - No card shadows, no border-left accents
 *   - Monospace for layer IDs in sublabels
 *
 * Spec §10, §52.
 */
import { useApp } from '../../store/appStore.jsx';

/** Human-readable names + colourmap description for each layer ID */
const LAYER_META = {
  buildings:     { label: 'Buildings',      sub: 'Solid-colour block model',     colormap: 'blocks' },
  semantics:     { label: 'Semantics',      sub: 'Learned semantic classification', colormap: 'categorical' },
  solid:         { label: 'Solid',          sub: 'Shaded surface + mesh',        colormap: 'solid' },
  rgb:           { label: 'RGB',            sub: 'Source photograph',           colormap: 'rgb' },
  depth:         { label: 'Depth',          sub: 'Monocular depth estimate',    colormap: 'greyscale' },
  dsm:           { label: 'DSM',            sub: 'Digital surface model',       colormap: 'viridis' },
  reference_dem: { label: 'Reference DEM',  sub: 'Ground-truth elevation',      colormap: 'viridis' },
  error:         { label: 'Error Map',       sub: 'Estimated − Reference',       colormap: 'diverging' },
  slope:         { label: 'Slope',          sub: 'Terrain gradient (°)',         colormap: 'viridis' },
  route_risk:    { label: 'Route Risk',     sub: 'Combined geometry & semantics risk', colormap: 'categorical' },
};

/**
 * @param {{
 *   activeLayer: string,
 *   onLayerChange: (layerId: string) => void,
 * }} props
 */
export default function LayerControl({ activeLayer, onLayerChange }) {
  const { state } = useApp();
  const isAbsolute = state.results?.elevation_mode === 'absolute' || state.scene?.is_georeferenced;
  const defaultAvailable = isAbsolute
    ? ['solid', 'rgb', 'depth', 'dsm', 'reference_dem', 'error', 'slope', 'semantics', 'route_risk']
    : ['solid', 'rgb', 'depth', 'dsm', 'slope', 'semantics', 'route_risk'];
  const availableLayers = state.results?.available_layers ?? defaultAvailable;

  return (
    <div
      aria-label="Layer selection"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
        width: '100%',
      }}
    >
      <p style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 11,
        letterSpacing: '0.07em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-ghost)',
        fontWeight: 600,
        margin: '0 0 10px 0',
      }}>
        Layers
      </p>

      {Object.entries(LAYER_META).map(([id, meta]) => {
        const isAvailable = availableLayers.includes(id);
        const isActive    = activeLayer === id;

        return (
          <LayerItem
            key={id}
            id={id}
            label={meta.label}
            sub={meta.sub}
            isActive={isActive}
            isAvailable={isAvailable}
            colormap={meta.colormap}
            onSelect={() => isAvailable && onLayerChange(id)}
          />
        );
      })}
    </div>
  );
}

/** Single layer list item */
function LayerItem({ id, label, sub, isActive, isAvailable, colormap, onSelect }) {
  const disabledReason = isAvailable ? null :
    id === 'dsm'           ? 'Requires georeferenced GeoTIFF input' :
    id === 'reference_dem' ? 'No reference DEM provided' :
    id === 'error'         ? 'Reference DEM required for error map' :
    id === 'semantics'     ? 'Semantic segmentation is unavailable for this scene' :
    id === 'route_risk'    ? 'Route-risk layer unavailable for this scene' :
    'Not available for this input';

  return (
    <button
      role="radio"
      aria-checked={isActive}
      aria-disabled={!isAvailable}
      title={disabledReason ?? undefined}
      onClick={onSelect}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        width: '100%',
        padding: '8px 10px',
        background: isActive ? 'var(--dw-surface)' : 'transparent',
        border: '1px solid ' + (isActive ? 'var(--dw-accent)' : 'transparent'),
        borderRadius: 'var(--dw-radius-sm)',
        cursor: isAvailable ? 'pointer' : 'not-allowed',
        opacity: isAvailable ? 1 : 0.38,
        outline: 'none',
        textAlign: 'left',
      }}
      onFocus={e => {
        if (isAvailable) {
          e.currentTarget.style.outline = '2px solid var(--dw-accent)';
          e.currentTarget.style.outlineOffset = '2px';
        }
      }}
      onBlur={e => { e.currentTarget.style.outline = 'none'; }}
    >
      {/* Left indicator dot */}
      <div style={{
        width: 7,
        height: 7,
        borderRadius: '50%',
        flexShrink: 0,
        background: isActive ? 'var(--dw-accent)' : isAvailable ? 'var(--dw-fg-ghost)' : 'var(--dw-rim)',
        transition: 'background 120ms ease',
      }} />

      {/* Label + sublabel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          fontWeight: isActive ? 600 : 500,
          color: isActive ? 'var(--dw-fg)' : isAvailable ? 'var(--dw-fg-muted)' : 'var(--dw-fg-ghost)',
          transition: 'color 120ms ease',
        }}>
          {label}
        </span>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 11.5,
          color: 'var(--dw-fg-ghost)',
        }}>
          {disabledReason ?? sub}
        </span>
      </div>

      {/* Colormap badge */}
      <ColormapBadge type={colormap} visible={isAvailable} />
    </button>
  );
}

/** Tiny colormap preview strip */
function ColormapBadge({ type, visible }) {
  if (!visible) return null;

  const gradients = {
    rgb:         'linear-gradient(to right, #e53e3e, #38a169, #3b82f6)',
    greyscale:   'linear-gradient(to right, #09090b, #f4f4f5)',
    viridis:     'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)',
    diverging:   'linear-gradient(to right, #2563eb, #dde4ef, #ef4444)',
    categorical: 'linear-gradient(to right, #e74c3c, #2ecc71, #9b9b9b, #3498db, #d2b48c)',
  };

  return (
    <div
      aria-hidden="true"
      style={{
        width: 28,
        height: 8,
        borderRadius: 2,
        background: gradients[type] ?? gradients.greyscale,
        flexShrink: 0,
        opacity: 0.7,
      }}
    />
  );
}

/** Export the LAYER_META for use in parent components (texture swap, colormap) */
export { LAYER_META };

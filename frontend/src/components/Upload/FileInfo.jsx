/**
 * DepthWizard — File Info Panel
 *
 * Shown after a successful upload. Displays image metadata and processing path.
 * Values come directly from the scene upload response — never invented.
 *
 * Spec §5 Upload Intelligence. DESIGN.md: two-column data-face table, no card chrome.
 */
import { useApp } from '../../store/appStore.jsx';
import PartialResultBanner from '../common/PartialResultBanner.jsx';

const LABELS = {
  absolute_dsm: 'Absolute DSM',
  relative_dsm: 'Relative DSM',
};

function Row({ label, value, valueStyle }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, minHeight: 24 }}>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 13,
        color: 'var(--dw-fg-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        minWidth: 130,
        flexShrink: 0,
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 14,
        color: 'var(--dw-fg)',
        ...valueStyle,
      }}>
        {value}
      </span>
    </div>
  );
}

function Tag({ label, positive }) {
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 8px',
      borderRadius: 'var(--dw-radius-sm)',
      fontFamily: 'var(--dw-font-data)',
      fontSize: 12,
      letterSpacing: '0.04em',
      background: positive ? 'rgba(74,222,128,0.1)' : 'rgba(107,125,150,0.1)',
      color: positive ? 'var(--dw-confirm)' : 'var(--dw-fg-muted)',
      border: `1px solid ${positive ? 'rgba(74,222,128,0.25)' : 'var(--dw-rim)'}`,
    }}>
      {label}
    </span>
  );
}

export default function FileInfo() {
  const { state } = useApp();
  const scene = state.scene;
  if (!scene) return null;

  const pathLabel = LABELS[scene.processing_path] ?? scene.processing_path;
  const isGeo = scene.georeferenced;

  return (
    <section
      aria-label="Image information"
      style={{
        width: '100%',
        background: 'var(--dw-surface)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius)',
        padding: '20px 24px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
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
        margin: 0,
      }}>
        IMAGE INFORMATION
      </p>

      {/* Divider */}
      <div style={{ height: 1, background: 'var(--dw-rim)', margin: '0 -24px' }} />

      {/* Data rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <Row label="File" value={scene.filename} />
        <Row label="Dimensions" value={`${scene.dimensions?.width ?? "—"} × ${scene.dimensions?.height ?? "—"}`} />
        <Row label="Format" value={scene.format} />
        {scene.crs && <Row label="CRS" value={scene.crs} />}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13,
            color: 'var(--dw-fg-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            minWidth: 130,
            flexShrink: 0,
          }}>Georeferenced</span>
          <Tag label={isGeo ? 'YES' : 'NO'} positive={isGeo} />
        </div>
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: 'var(--dw-rim)', margin: '0 -24px' }} />

      {/* Processing path */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-muted)',
          margin: 0,
        }}>
          Processing Path
        </p>
        <p style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 15,
          fontWeight: 600,
          color: 'var(--dw-accent)',
          margin: 0,
        }}>
          {pathLabel}
        </p>
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13.5,
          color: 'var(--dw-fg-muted)',
          margin: 0,
          lineHeight: 1.55,
        }}>
          {scene.processing_path === 'absolute_dsm'
            ? 'Georeferencing detected. Metric scale recovery enabled. Producing absolute elevation values in metres.'
            : 'No georeferencing. Producing relative elevation on the 1 m/pixel fallback scale. Absolute elevation requires a reference DEM.'}
        </p>
      </div>

      {/* Pipeline partial result capabilities (§38) */}
      <PartialResultBanner
        format={scene.format}
        georeferenced={scene.georeferenced}
        elevationMode={scene.processing_path === 'absolute_dsm' ? 'absolute' : 'relative'}
        hasReference={scene.reference_available}
        compact={true}
      />
    </section>
  );
}

/**
 * DepthWizard — File Info Panel
 *
 * Shown after a successful upload. Displays image metadata and processing path.
 * Values come directly from the scene upload response — never invented.
 *
 * Spec §5 Upload Intelligence. DESIGN.md: two-column data-face table, no card chrome.
 */
import { useApp } from '../../store/appStore.jsx';

const LABELS = {
  absolute_dsm: 'Absolute DSM',
  relative_dsm: 'Relative DSM',
};

function Row({ label, value, valueStyle }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, minHeight: 20 }}>
      <span style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 11,
        color: 'var(--dw-fg-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        minWidth: 110,
        flexShrink: 0,
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--dw-font-data)',
        fontSize: 12,
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
      padding: '1px 7px',
      borderRadius: 'var(--dw-radius-sm)',
      fontFamily: 'var(--dw-font-data)',
      fontSize: 11,
      letterSpacing: '0.04em',
      background: positive ? 'rgba(34,197,94,0.1)' : 'rgba(107,125,150,0.1)',
      color: positive ? 'var(--dw-confirm)' : 'var(--dw-fg-muted)',
      border: `1px solid ${positive ? 'rgba(34,197,94,0.25)' : 'var(--dw-rim)'}`,
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
        maxWidth: 560,
        background: 'var(--dw-surface)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius)',
        padding: '16px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
    >
      {/* Section label */}
      <p style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 10,
        fontWeight: 500,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-muted)',
        margin: 0,
      }}>
        IMAGE INFORMATION
      </p>

      {/* Divider */}
      <div style={{ height: 1, background: 'var(--dw-rim)', margin: '0 -20px' }} />

      {/* Data rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <Row label="File" value={scene.filename} />
        <Row label="Dimensions" value={`${scene.width} × ${scene.height}`} />
        <Row label="Format" value={scene.format} />
        {scene.crs && <Row label="CRS" value={scene.crs} />}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            color: 'var(--dw-fg-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            minWidth: 110,
            flexShrink: 0,
          }}>Georeferenced</span>
          <Tag label={isGeo ? 'YES' : 'NO'} positive={isGeo} />
        </div>
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: 'var(--dw-rim)', margin: '0 -20px' }} />

      {/* Processing path */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 10,
          fontWeight: 500,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-muted)',
          margin: 0,
        }}>
          Processing Path
        </p>
        <p style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 13,
          fontWeight: 500,
          color: 'var(--dw-accent)',
          margin: 0,
        }}>
          {pathLabel}
        </p>
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 11,
          color: 'var(--dw-fg-muted)',
          margin: 0,
          lineHeight: 1.5,
        }}>
          {scene.processing_path === 'absolute_dsm'
            ? 'Georeferencing detected. Metric scale recovery enabled. Producing absolute elevation values in metres.'
            : 'No georeferencing. Producing relative elevation in scene units. Absolute elevation is unavailable for this input.'}
        </p>
      </div>

      {!scene.reference_available && scene.processing_path === 'absolute_dsm' && (
        <p style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 11,
          color: 'var(--dw-live)',
          margin: 0,
        }}>
          Reference elevation unavailable — validation will be skipped.
        </p>
      )}
    </section>
  );
}

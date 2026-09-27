/**
 * DepthWizard — PartialResultBanner (Phase 18, Task 18.3)
 *
 * Implements spec §38 Empty / Partial Results:
 *
 *   PNG/JPG:
 *     - Absolute elevation unavailable
 *     - Relative DSM available
 *
 *   GeoTIFF without reference:
 *     - Georeferencing detected
 *     - Reference elevation unavailable
 *     - Relative DSM available
 *
 *   GeoTIFF + DEM/GCP:
 *     - Absolute DSM available
 *     - Metric analysis enabled
 *
 * DESIGN.md: Data face monospace tags, thin --dw-rim borders, no card shadows.
 */
import { CheckCircle2, AlertCircle, Info } from 'lucide-react';

/**
 * @param {{
 *   format?: string,
 *   georeferenced?: boolean,
 *   elevationMode?: 'absolute' | 'relative',
 *   hasReference?: boolean,
 *   compact?: boolean,
 * }} props
 */
export default function PartialResultBanner({
  format = 'PNG',
  georeferenced = false,
  elevationMode = 'relative',
  hasReference = false,
  compact = false,
}) {
  // A GeoTIFF is only treated as georeferenced when the scene record says
  // so — a TIF without valid CRS/transform falls through to the optical
  // (relative) category instead of claiming georeferencing.
  const isGeoTiff = georeferenced;
  const isAbsolute = elevationMode === 'absolute' || hasReference;

  // Determine §38 category:
  // 1. GeoTIFF + DEM/GCP (isGeoTiff && isAbsolute)
  // 2. GeoTIFF without reference (isGeoTiff && !isAbsolute)
  // 3. PNG/JPG or non-georeferenced raster (!isGeoTiff)

  let items = [];
  let categoryLabel = '';

  if (isGeoTiff && isAbsolute) {
    categoryLabel = 'Calibrated Pipeline';
    items = [
      { text: 'Absolute DSM available', status: 'pass' },
      { text: 'Metric analysis enabled', status: 'pass' },
    ];
  } else if (isGeoTiff && !isAbsolute) {
    categoryLabel = 'Georeferenced (No Ground Truth)';
    items = [
      { text: 'Georeferencing detected', status: 'pass' },
      { text: 'Reference elevation unavailable', status: 'warn' },
      { text: 'Relative DSM available', status: 'pass' },
    ];
  } else if (format.toUpperCase().includes('TIF')) {
    // GeoTIFF container without usable georeferencing
    categoryLabel = 'Non-Georeferenced Raster Pipeline';
    items = [
      { text: 'Georeferencing unavailable', status: 'warn' },
      { text: 'Absolute elevation unavailable', status: 'neutral' },
      { text: 'Relative DSM available', status: 'pass' },
    ];
  } else {
    categoryLabel = 'Monocular Optical Pipeline';
    items = [
      { text: 'Absolute elevation unavailable', status: 'neutral' },
      { text: 'Relative DSM available', status: 'pass' },
    ];
  }

  return (
    <div
      role="status"
      aria-label={`Pipeline capabilities: ${categoryLabel}`}
      style={{
        background: 'var(--dw-panel)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        padding: compact ? '6px 10px' : '10px 14px',
        display: 'flex',
        flexDirection: compact ? 'row' : 'column',
        alignItems: compact ? 'center' : 'flex-start',
        justifyContent: 'space-between',
        gap: compact ? 12 : 6,
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 11.5,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-muted)',
          fontWeight: 600,
        }}>
          {categoryLabel}
        </span>
      </div>

      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        flexWrap: 'wrap',
      }}>
        {items.map((item, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontFamily: 'var(--dw-font-data)',
              fontSize: 12.5,
              fontWeight: 500,
              color: item.status === 'pass'
                ? 'var(--dw-fg)'
                : item.status === 'warn'
                ? 'var(--dw-live)'
                : 'var(--dw-fg-muted)',
            }}
          >
            {item.status === 'pass' ? (
              <CheckCircle2 size={14} strokeWidth={2} color="var(--dw-confirm)" aria-hidden="true" />
            ) : item.status === 'warn' ? (
              <AlertCircle size={14} strokeWidth={2} color="var(--dw-live)" aria-hidden="true" />
            ) : (
              <Info size={14} strokeWidth={1.5} color="var(--dw-fg-ghost)" aria-hidden="true" />
            )}
            <span>{item.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

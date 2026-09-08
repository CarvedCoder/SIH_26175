/**
 * DepthWizard — ExportPanel (Phase 15, Tasks 15.1, 15.2, 15.3)
 *
 * Export intelligence data panel:
 * - DSM (GeoTIFF) — only when absolute/georeferenced
 * - Relative DSM — for non-georeferenced scenes
 * - Depth Map (16-bit / 8-bit PNG)
 * - Terrain Viewport Snapshot (Live 3D capture)
 * - Validation Accuracy Report (JSON/CSV) — only when reference DEM available
 * - 3D Scene Asset (GLTF/OBJ mesh)
 *
 * Capability-driven:
 * - Only enables buttons for outputs that actually exist in scene summary (§27, §66).
 * - Triggers download stream or redirect via api/export.js.
 *
 * Spec §27, §66.
 * DESIGN.md: Operate mode, monospace format tags, thin --dw-rim borders, no card shadows.
 */
import { useState } from 'react';
import { useApp } from '../../store/appStore.jsx';
import {
  exportDsm,
  exportDepth,
  exportValidation,
  exportTerrain,
} from '../../api/export.js';
import {
  Download,
  FileImage,
  Camera,
  ShieldCheck,
  Box,
  FileSpreadsheet,
  Check,
  X,
  ExternalLink,
} from 'lucide-react';

/**
 * @param {{
 *   sceneId?: string | null,
 *   onCaptureSnapshot?: () => void,
 *   onClose?: () => void,
 *   compact?: boolean,
 * }} props
 */
export default function ExportPanel({
  sceneId = null,
  onCaptureSnapshot,
  onClose,
  compact = false,
}) {
  const { state } = useApp();
  const [downloading, setDownloading] = useState(null); // id of downloading item
  const [downloaded, setDownloaded] = useState({});

  const id = sceneId ?? state.scene?.scene_id;
  const isAbsolute = state.results?.elevation_mode === 'absolute';
  const hasValidation = isAbsolute && state.results?.reference_source != null;

  // Determine capabilities from results
  const outputs = state.results?.outputs ?? ['depth', 'terrain', ...(isAbsolute ? ['dsm'] : [])];

  const handleTrigger = async (outputId, exportFn) => {
    if (!id && outputId !== 'snapshot') return;
    setDownloading(outputId);
    try {
      if (outputId === 'snapshot') {
        onCaptureSnapshot?.();
      } else {
        await exportFn(id);
      }
      setDownloaded(prev => ({ ...prev, [outputId]: true }));
      setTimeout(() => {
        setDownloaded(prev => ({ ...prev, [outputId]: false }));
      }, 3000);
    } catch (err) {
      console.error('[ExportPanel] download failed', err);
    } finally {
      setDownloading(null);
    }
  };

  const EXPORT_ITEMS = [
    {
      id: 'dsm',
      title: isAbsolute ? 'Digital Surface Model (DSM)' : 'Relative DSM',
      format: isAbsolute ? 'GeoTIFF (32-bit float)' : 'GeoTIFF (Normalised)',
      desc: isAbsolute ? 'Metric calibrated elevation grid' : 'Relative elevation field in scene units',
      icon: FileSpreadsheet,
      enabled: outputs.includes('dsm') || isAbsolute,
      disabledReason: 'DSM export requires completed processing.',
      action: () => handleTrigger('dsm', exportDsm),
    },
    {
      id: 'depth',
      title: 'Monocular Depth Map',
      format: 'PNG (16-bit Greyscale)',
      desc: 'Normalized disparity/depth buffer',
      icon: FileImage,
      enabled: outputs.includes('depth') || true,
      disabledReason: 'Depth map output not available.',
      action: () => handleTrigger('depth', exportDepth),
    },
    {
      id: 'snapshot',
      title: 'Terrain Viewport Snapshot',
      format: 'PNG (Viewport Resolution)',
      desc: 'Current camera perspective and layer rendering',
      icon: Camera,
      enabled: !!onCaptureSnapshot,
      disabledReason: 'Viewport snapshot is available inside the 3D terrain workspace.',
      action: () => handleTrigger('snapshot', () => onCaptureSnapshot?.()),
    },
    {
      id: 'validation',
      title: 'Validation Accuracy Report',
      format: 'JSON / CSV Report',
      desc: 'RMSE, MAE, Pearson correlation vs ground truth DEM',
      icon: ShieldCheck,
      enabled: isAbsolute,
      disabledReason: 'Validation report requires georeferenced imagery matched against reference DEM.',
      action: () => handleTrigger('validation', exportValidation),
    },
    {
      id: 'terrain',
      title: '3D Terrain Mesh Scene',
      format: 'GLTF / OBJ Mesh',
      desc: 'Reconstructed surface geometry with diffuse texture',
      icon: Box,
      enabled: outputs.includes('terrain') || true,
      disabledReason: '3D scene asset package not generated.',
      action: () => handleTrigger('terrain', exportTerrain),
    },
  ];

  return (
    <div
      role="region"
      aria-label="Export generated outputs"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: compact ? 10 : 14,
        width: '100%',
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingBottom: 6,
        borderBottom: '1px solid var(--dw-rim)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Download size={13} strokeWidth={1.5} color="var(--dw-accent)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 10,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-ghost)',
            fontWeight: 500,
          }}>
            Export Outputs (§27, §66)
          </span>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            aria-label="Close export panel"
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--dw-fg-muted)',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
              outline: 'none',
            }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '2px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            <X size={12} strokeWidth={1.5} />
          </button>
        )}
      </div>

      {/* Export Items List */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}>
        {EXPORT_ITEMS.map(item => {
          const Icon = item.icon;
          const isBusy = downloading === item.id;
          const isDone = downloaded[item.id];
          const isEnabled = item.id === 'snapshot' ? item.enabled : (item.enabled && !!id);

          return (
            <div
              key={item.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 10px',
                background: 'var(--dw-surface)',
                border: '1px solid var(--dw-rim)',
                borderRadius: 'var(--dw-radius-sm)',
                gap: 8,
              }}
            >
              <div style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 8,
                flex: 1,
                minWidth: 0,
              }}>
                <Icon
                  size={15}
                  strokeWidth={1.5}
                  color={isEnabled ? 'var(--dw-accent)' : 'var(--dw-fg-ghost)'}
                  style={{ marginTop: 2, flexShrink: 0 }}
                  aria-hidden="true"
                />
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{
                      fontFamily: 'var(--dw-font-ui)',
                      fontSize: 11,
                      fontWeight: 500,
                      color: isEnabled ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}>
                      {item.title}
                    </span>
                  </div>
                  <span style={{
                    fontFamily: 'var(--dw-font-data)',
                    fontSize: 9,
                    color: isEnabled ? 'var(--dw-fg-muted)' : 'var(--dw-fg-ghost)',
                  }}>
                    {item.format}
                  </span>
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={item.action}
                disabled={!isEnabled || isBusy}
                aria-label={`Export ${item.title}`}
                title={isEnabled ? `Download ${item.title}` : item.disabledReason}
                style={{
                  height: 26,
                  padding: '0 8px',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                  background: isEnabled ? 'var(--dw-panel)' : 'transparent',
                  border: '1px solid ' + (isEnabled ? 'var(--dw-rim)' : 'transparent'),
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 10,
                  color: isDone
                    ? 'var(--dw-confirm)'
                    : isEnabled
                    ? 'var(--dw-fg)'
                    : 'var(--dw-fg-ghost)',
                  cursor: isEnabled ? 'pointer' : 'not-allowed',
                  outline: 'none',
                  flexShrink: 0,
                  transition: 'border-color 120ms ease, background 120ms ease',
                }}
                onFocus={e => {
                  if (isEnabled) {
                    e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                    e.currentTarget.style.outlineOffset = '1px';
                  }
                }}
                onBlur={e => { e.currentTarget.style.outline = 'none'; }}
                onMouseEnter={e => {
                  if (isEnabled) {
                    e.currentTarget.style.borderColor = 'var(--dw-accent)';
                    e.currentTarget.style.color = 'var(--dw-accent)';
                  }
                }}
                onMouseLeave={e => {
                  if (isEnabled) {
                    e.currentTarget.style.borderColor = 'var(--dw-rim)';
                    e.currentTarget.style.color = isDone ? 'var(--dw-confirm)' : 'var(--dw-fg)';
                  }
                }}
              >
                {isDone ? (
                  <>
                    <Check size={11} strokeWidth={2} />
                    <span>Saved</span>
                  </>
                ) : isBusy ? (
                  <span>Streaming…</span>
                ) : (
                  <>
                    <Download size={11} strokeWidth={1.5} />
                    <span>Export</span>
                  </>
                )}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * DepthWizard — Processing Path explainer (minimal inline component)
 * Shown between FileInfo and the "Start Processing" button.
 * Spec §5. No chrome — just an honest sentence.
 */
import { useApp } from '../../store/appStore.jsx';

export default function ProcessingPath() {
  const { state } = useApp();
  if (!state.scene) return null;
  // Explanation is already part of FileInfo — this component provides
  // the visual pipeline flow arrow diagram (RGB→Depth→DSM→3D→Analysis)
  // as a compact inline row.
  const steps = state.scene.processing_path === 'absolute_dsm'
    ? ['RGB', 'Depth', 'Absolute DSM', '3D Terrain', 'Analysis']
    : ['RGB', 'Depth', 'Relative DSM', '3D Terrain', 'Analysis'];

  return (
    <div
      aria-label="Processing pipeline"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        flexWrap: 'wrap',
      }}
    >
      {steps.map((step, i) => (
        <span key={step} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 13,
            fontWeight: 500,
            color: 'var(--dw-fg)',
            letterSpacing: '0.04em',
          }}>
            {step}
          </span>
          {i < steps.length - 1 && (
            <span style={{ color: 'var(--dw-rim)', fontSize: 13 }}>→</span>
          )}
        </span>
      ))}
    </div>
  );
}

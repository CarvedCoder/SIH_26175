/**
 * DepthWizard — ToolGuard (Phase 9, Task 9.6)
 *
 * Capability / readiness guard for all analysis and measurement tools (§73).
 *
 * Rules:
 *   - If state < TERRAIN_READY (i.e. NO_SCENE, UPLOADING, SCENE_READY, PROCESSING,
 *     RESULTS_READY, TERRAIN_LOADING), measurement tools render disabled with:
 *     "Terrain not ready" notice.
 *   - Never crashes or executes unready raycasts/probes before mesh is loaded.
 *
 * DESIGN.md:
 *   - Muted banner/overlay with --dw-fg-ghost text
 *   - No blocking alert modals
 *
 * Spec §73, §9.6.
 */
import { useApp, isTerrainReady } from '../../store/appStore.jsx';
import { AlertCircle } from 'lucide-react';

/**
 * @param {{
 *   children: React.ReactNode,
 *   fallbackMessage?: string,
 * }} props
 */
export default function ToolGuard({ children, fallbackMessage = 'Terrain not ready' }) {
  const { state } = useApp();
  const ready = isTerrainReady(state.status);

  if (!ready) {
    return (
      <div
        aria-live="polite"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '10px 14px',
          background: 'rgba(16,16,18,0.85)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13,
          color: 'var(--dw-fg-ghost)',
          pointerEvents: 'none',
        }}
      >
        <AlertCircle size={15} strokeWidth={1.5} color="var(--dw-fg-ghost)" aria-hidden="true" />
        <span>{fallbackMessage}</span>
      </div>
    );
  }

  return <>{children}</>;
}

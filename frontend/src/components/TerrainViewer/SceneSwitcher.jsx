/**
 * DepthWizard — SceneSwitcher (multi-image batch navigation)
 *
 * Top-centre pill shown when the user uploaded several images in one
 * batch (§27-adjacent tiles flow). Adjacent GeoTIFF tiles merged into a
 * single mosaic scene never reach here — only genuinely separate scenes:
 *
 *   [◀]  1  2  3  [▶]
 *
 *   - Arrows cycle the batch in upload order (wrap around).
 *   - Numbered chips jump directly to a scene, like tabs.
 *   - A hollow chip marks a scene that has NOT been processed yet —
 *     switching to it returns to the workspace home to run processing.
 *   - Completed scenes reload straight into the 3D terrain.
 *
 * DESIGN.md: dark instrument-panel pill, one accent colour for the active
 * tab, no decorative chrome.
 */
import { useCallback } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useApp } from '../../store/appStore.jsx';

const btnBase = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  padding: 0,
  outline: 'none',
  color: 'var(--dw-fg-muted)',
};

export default function SceneSwitcher({ disabled = false }) {
  const { state, actions } = useApp();
  const queue = state.sceneQueue;
  const currentId = state.scene?.scene_id;

  const completed = useCallback(
    (id) => {
      if (id === state.scene?.scene_id && state.results) return true;
      return !!state.recentScenes.find(s => s.scene_id === id)?.results;
    },
    [state.scene?.scene_id, state.results, state.recentScenes],
  );

  const activeIndex = queue.findIndex(s => s.scene_id === currentId);
  if (queue.length < 2 || activeIndex < 0) return null;

  const switchTo = (index) => {
    if (disabled) return;
    const target = queue[(index + queue.length) % queue.length];
    if (!target || target.scene_id === currentId) return;
    actions.switchScene(target.scene_id);
    if (completed(target.scene_id)) actions.startTerrainLoad();
  };

  return (
    <div
      role="tablist"
      aria-label="Switch between uploaded scenes"
      style={{
        position: 'absolute',
        top: 12,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 12,
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        background: 'rgba(13,17,23,0.92)',
        border: '1px solid var(--dw-rim)',
        borderRadius: 'var(--dw-radius-sm)',
        padding: '3px 6px',
      }}
    >
      <button
        type="button"
        role="tab"
        aria-label="Previous scene"
        title="Previous scene"
        disabled={disabled}
        onClick={() => switchTo(activeIndex - 1)}
        style={{ ...btnBase, width: 26, height: 26, opacity: disabled ? 0.4 : 1 }}
        onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; }}
        onBlur={e => { e.currentTarget.style.outline = 'none'; }}
      >
        <ChevronLeft size={16} strokeWidth={1.8} />
      </button>

      {queue.map((s, i) => {
        const isActive = i === activeIndex;
        const done = completed(s.scene_id);
        return (
          <button
            key={s.scene_id}
            type="button"
            role="tab"
            aria-selected={isActive}
            aria-label={`Scene ${i + 1}: ${s.filename}${done ? '' : ' (not processed)'}`}
            title={`${s.filename}${done ? '' : ' — not processed yet'}`}
            disabled={disabled}
            onClick={() => switchTo(i)}
            style={{
              minWidth: 26,
              height: 26,
              borderRadius: 4,
              fontFamily: 'var(--dw-font-data)',
              fontSize: 12.5,
              fontWeight: isActive ? 700 : 500,
              color: isActive ? '#0b0f16' : done ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
              background: isActive ? 'var(--dw-accent)' : 'transparent',
              border: done || isActive ? 'none' : '1px dashed var(--dw-rim)',
              cursor: disabled ? 'wait' : 'pointer',
              padding: '0 4px',
              outline: 'none',
            }}
            onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            {i + 1}
          </button>
        );
      })}

      <button
        type="button"
        role="tab"
        aria-label="Next scene"
        title="Next scene"
        disabled={disabled}
        onClick={() => switchTo(activeIndex + 1)}
        style={{ ...btnBase, width: 26, height: 26, opacity: disabled ? 0.4 : 1 }}
        onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; }}
        onBlur={e => { e.currentTarget.style.outline = 'none'; }}
      >
        <ChevronRight size={16} strokeWidth={1.8} />
      </button>
    </div>
  );
}

/**
 * DepthWizard — RecentProjects (Phase 16, Tasks 16.1, 16.2)
 *
 * Displays saved / recent sessions from localStorage.
 * Spec §28:
 *   RECENT PROJECTS
 *   Scene_042 · Absolute DSM · Processed 2 min ago
 *   Hill_Area · Relative DSM · Processed yesterday
 *   Urban_Block · Absolute DSM · Processed yesterday
 *
 * Capabilities:
 *   - Restores session state (scene, results) and navigates to Results or Terrain
 *   - Shows filename, pipeline mode (Absolute DSM vs Relative DSM), relative timestamp
 *   - Quick "Open Terrain" jump button
 *   - Remove single entry or clear all
 *
 * DESIGN.md: Operate mode. Monospace data timestamps, thin --dw-rim borders, no shadows.
 */
import { useState } from 'react';
import { useApp, AppState } from '../../store/appStore.jsx';
import {
  Clock,
  FolderOpen,
  Mountain,
  Trash2,
  ExternalLink,
  Layers,
  ChevronRight,
  CheckCircle2,
} from 'lucide-react';

/**
 * Format a numeric timestamp into a human-readable relative string per §28.
 * @param {number} timestamp - ms since epoch
 * @returns {string} e.g. "Just now", "2 min ago", "Yesterday"
 */
function formatRelativeTime(timestamp) {
  if (!timestamp) return 'Recently';
  const diffMs = Date.now() - timestamp;
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return 'Just now';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin} min ago`;
  const diffHour = Math.floor(diffMin / 60);
  if (diffHour < 24) return `${diffHour} hr${diffHour > 1 ? 's' : ''} ago`;
  const diffDays = Math.floor(diffHour / 24);
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return new Date(timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/**
 * @param {{
 *   onSelectProject?: (project: any) => void,
 *   compact?: boolean,
 *   onClose?: () => void,
 * }} props
 */
export default function RecentProjects({
  onSelectProject,
  compact = false,
  onClose,
}) {
  const { state, actions } = useApp();
  const [hoveredId, setHoveredId] = useState(null);

  const projects = state.recentScenes ?? [];

  const handleResume = (project, targetStatus = null) => {
    actions.resumeSession(project, targetStatus);
    onSelectProject?.(project);
    onClose?.();
  };

  const handleRemove = (e, sceneId) => {
    e.stopPropagation();
    actions.removeRecentProject(sceneId);
  };

  const handleClearAll = () => {
    if (window.confirm('Clear all recent projects from this browser session?')) {
      actions.clearRecentProjects();
    }
  };

  return (
    <section
      aria-label="Recent projects"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: compact ? 8 : 12,
        width: '100%',
      }}
    >
      {/* Header strip */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingBottom: 4,
        borderBottom: '1px solid var(--dw-rim)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Clock size={15} strokeWidth={1.5} color="var(--dw-fg-muted)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 12,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: 'var(--dw-fg-muted)',
            fontWeight: 600,
          }}>
            RECENT PROJECTS (§28)
          </span>
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 11.5,
            color: 'var(--dw-fg)',
            padding: '2px 6px',
            background: 'var(--dw-surface)',
            borderRadius: 3,
            border: '1px solid var(--dw-rim)',
          }}>
            {projects.length}
          </span>
        </div>

        {projects.length > 0 && (
          <button
            onClick={handleClearAll}
            aria-label="Clear all recent projects"
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 12,
              color: 'var(--dw-fg-muted)',
              padding: '4px 8px',
              borderRadius: 'var(--dw-radius-sm)',
              outline: 'none',
              transition: 'color 120ms ease',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = 'var(--dw-fault)'; }}
            onMouseLeave={e => { e.currentTarget.style.color = 'var(--dw-fg-muted)'; }}
            onFocus={e => {
              e.currentTarget.style.outline = '2px solid var(--dw-accent)';
              e.currentTarget.style.outlineOffset = '1px';
            }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            Clear all
          </button>
        )}
      </div>

      {/* Projects list */}
      {projects.length === 0 ? (
        <div style={{
          padding: '28px 20px',
          background: 'var(--dw-surface)',
          border: '1px solid var(--dw-rim)',
          borderRadius: 'var(--dw-radius-sm)',
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 8,
        }}>
          <FolderOpen size={24} strokeWidth={1.5} color="var(--dw-fg-ghost)" aria-hidden="true" />
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 14,
            color: 'var(--dw-fg)',
            fontWeight: 500,
          }}>
            No recent projects found
          </span>
          <span style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 12.5,
            color: 'var(--dw-fg-muted)',
          }}>
            Uploaded and processed scenes are persisted here automatically.
          </span>
        </div>
      ) : (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}>
          {projects.map((proj) => {
            const isAbsolute = proj.elevation_mode === 'absolute' || proj.processing_path === 'absolute_dsm';
            const modeLabel = isAbsolute ? 'Absolute DSM' : 'Relative DSM';
            const isHovered = hoveredId === proj.scene_id;
            const isCurrent = state.scene?.scene_id === proj.scene_id;

            return (
              <div
                key={proj.scene_id}
                onClick={() => handleResume(proj)}
                onMouseEnter={() => setHoveredId(proj.scene_id)}
                onMouseLeave={() => setHoveredId(null)}
                role="button"
                tabIndex={0}
                aria-label={`Resume session for ${proj.filename || proj.scene_id}, ${modeLabel}`}
                onKeyDown={e => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleResume(proj);
                  }
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: compact ? '10px 12px' : '12px 16px',
                  background: isCurrent
                    ? 'rgba(59,130,246,0.06)'
                    : isHovered
                    ? 'var(--dw-surface)'
                    : 'var(--dw-panel)',
                  border: isCurrent
                    ? '1px solid var(--dw-accent)'
                    : isHovered
                    ? '1px solid rgba(59,130,246,0.5)'
                    : '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  cursor: 'pointer',
                  outline: 'none',
                  transition: 'background 120ms ease, border-color 120ms ease',
                  gap: 12,
                }}
                onFocus={e => {
                  e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                  e.currentTarget.style.outlineOffset = '2px';
                }}
                onBlur={e => { e.currentTarget.style.outline = 'none'; }}
              >
                {/* Left metadata */}
                <div style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 10,
                  minWidth: 0,
                  flex: 1,
                }}>
                  {/* Status / Mode icon */}
                  <div style={{
                    marginTop: 4,
                    flexShrink: 0,
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: isAbsolute ? 'var(--dw-accent)' : 'var(--dw-fg-muted)',
                    boxShadow: isAbsolute ? '0 0 0 2px rgba(59,130,246,0.2)' : 'none',
                  }} aria-hidden="true" />

                  <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 4,
                    minWidth: 0,
                  }}>
                    {/* Scene name / title */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{
                        fontFamily: 'var(--dw-font-ui)',
                        fontSize: 14,
                        fontWeight: 550,
                        color: 'var(--dw-fg)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}>
                        {proj.filename || proj.scene_id}
                      </span>
                      {isCurrent && (
                        <span style={{
                          fontFamily: 'var(--dw-font-data)',
                          fontSize: 10.5,
                          color: 'var(--dw-accent)',
                          padding: '1px 5px',
                          border: '1px solid rgba(59,130,246,0.4)',
                          borderRadius: 2,
                        }}>
                          ACTIVE
                        </span>
                      )}
                    </div>

                    {/* Metadata line: Mode + Timestamp */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      fontFamily: 'var(--dw-font-data)',
                      fontSize: 12,
                      color: 'var(--dw-fg-muted)',
                      flexWrap: 'wrap',
                    }}>
                      <span style={{
                        color: isAbsolute ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
                        fontWeight: 500,
                      }}>
                        {modeLabel}
                      </span>
                      <span style={{ color: 'var(--dw-rim)' }}>•</span>
                      <span>Processed {formatRelativeTime(proj.ts)}</span>
                    </div>
                  </div>
                </div>

                {/* Right actions */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  flexShrink: 0,
                }}>
                  {/* Enter Terrain shortcut button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleResume(proj, AppState.TERRAIN_LOADING);
                    }}
                    title="Jump directly to 3D Terrain"
                    aria-label={`Enter 3D terrain for ${proj.filename || proj.scene_id}`}
                    style={{
                      height: 30,
                      padding: '0 10px',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 6,
                      background: 'var(--dw-surface)',
                      border: '1px solid var(--dw-rim)',
                      borderRadius: 'var(--dw-radius-sm)',
                      fontFamily: 'var(--dw-font-ui)',
                      fontSize: 12.5,
                      color: 'var(--dw-fg)',
                      cursor: 'pointer',
                      outline: 'none',
                      transition: 'border-color 120ms ease, color 120ms ease, background 120ms ease',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.borderColor = 'var(--dw-accent)';
                      e.currentTarget.style.color = 'var(--dw-accent)';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.borderColor = 'var(--dw-rim)';
                      e.currentTarget.style.color = 'var(--dw-fg)';
                    }}
                    onFocus={e => {
                      e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                      e.currentTarget.style.outlineOffset = '1px';
                    }}
                    onBlur={e => { e.currentTarget.style.outline = 'none'; }}
                  >
                    <Mountain size={13} strokeWidth={1.5} />
                    <span>3D Terrain</span>
                  </button>

                  {/* Delete / remove from list */}
                  <button
                    onClick={(e) => handleRemove(e, proj.scene_id)}
                    title="Remove from recent list"
                    aria-label={`Remove ${proj.filename || proj.scene_id} from recent projects`}
                    style={{
                      width: 30,
                      height: 30,
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: 'none',
                      border: 'none',
                      borderRadius: 'var(--dw-radius-sm)',
                      color: 'var(--dw-fg-muted)',
                      cursor: 'pointer',
                      outline: 'none',
                      transition: 'color 120ms ease',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.color = 'var(--dw-fault)'; }}
                    onMouseLeave={e => { e.currentTarget.style.color = 'var(--dw-fg-muted)'; }}
                    onFocus={e => {
                      e.currentTarget.style.outline = '2px solid var(--dw-accent)';
                      e.currentTarget.style.outlineOffset = '1px';
                    }}
                    onBlur={e => { e.currentTarget.style.outline = 'none'; }}
                  >
                    <Trash2 size={14} strokeWidth={1.5} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

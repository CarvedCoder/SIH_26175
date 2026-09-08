/**
 * DepthWizard — ScenarioSwitcher (Phase 14, Tasks 14.1, 14.2)
 *
 * Switches between operational scenarios:
 * - [ Terrain Exploration ] (Default)
 * - [ Disaster Assessment ] (Rapid terrain intelligence for relief operations)
 *
 * Spec §23:
 * - Emphasizes elevation, slope, low/high terrain, accessibility, structure height.
 * - Explicitly labeled as "Terrain intelligence / preliminary terrain assessment support".
 * - Never claims automated flood/landslide probability predictions.
 */
import { Compass, AlertTriangle, ShieldCheck } from 'lucide-react';

/**
 * @param {{
 *   scenario: 'exploration' | 'disaster',
 *   onSelectScenario: (scenario: 'exploration' | 'disaster') => void,
 *   compact?: boolean,
 * }} props
 */
export default function ScenarioSwitcher({
  scenario = 'exploration',
  onSelectScenario,
  compact = false,
}) {
  return (
    <div
      role="group"
      aria-label="Scenario preset mode"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        width: '100%',
      }}
    >
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <span style={{
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 9,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          color: 'var(--dw-fg-ghost)',
          fontWeight: 500,
        }}>
          Operational Scenario (§23)
        </span>
        {scenario === 'disaster' && (
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 9,
            color: 'var(--dw-live)',
            background: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.25)',
            padding: '1px 5px',
            borderRadius: 'var(--dw-radius-sm)',
          }}>
            ACTIVE PRESET
          </span>
        )}
      </div>

      {/* Segmented Switcher */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        background: 'var(--dw-surface)',
        padding: 2,
        borderRadius: 'var(--dw-radius-sm)',
        border: '1px solid var(--dw-rim)',
        gap: 2,
      }}>
        {/* Option 1: Terrain Exploration */}
        <button
          onClick={() => onSelectScenario('exploration')}
          aria-pressed={scenario === 'exploration'}
          style={{
            height: 30,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
            background: scenario === 'exploration' ? 'var(--dw-panel)' : 'transparent',
            border: scenario === 'exploration' ? '1px solid var(--dw-accent)' : '1px solid transparent',
            borderRadius: 'var(--dw-radius-sm)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            color: scenario === 'exploration' ? 'var(--dw-fg)' : 'var(--dw-fg-muted)',
            cursor: 'pointer',
            outline: 'none',
            transition: 'background 120ms ease, border-color 120ms ease',
          }}
          onFocus={e => {
            e.currentTarget.style.outline = '2px solid var(--dw-accent)';
            e.currentTarget.style.outlineOffset = '1px';
          }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        >
          <Compass size={12} strokeWidth={1.5} color={scenario === 'exploration' ? 'var(--dw-accent)' : 'var(--dw-fg-muted)'} />
          Exploration
        </button>

        {/* Option 2: Disaster Assessment */}
        <button
          onClick={() => onSelectScenario('disaster')}
          aria-pressed={scenario === 'disaster'}
          style={{
            height: 30,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
            background: scenario === 'disaster' ? 'var(--dw-panel)' : 'transparent',
            border: scenario === 'disaster' ? '1px solid var(--dw-live)' : '1px solid transparent',
            borderRadius: 'var(--dw-radius-sm)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            fontWeight: scenario === 'disaster' ? 500 : 400,
            color: scenario === 'disaster' ? 'var(--dw-live)' : 'var(--dw-fg-muted)',
            cursor: 'pointer',
            outline: 'none',
            transition: 'background 120ms ease, border-color 120ms ease',
          }}
          onFocus={e => {
            e.currentTarget.style.outline = '2px solid var(--dw-live)';
            e.currentTarget.style.outlineOffset = '1px';
          }}
          onBlur={e => { e.currentTarget.style.outline = 'none'; }}
        >
          <AlertTriangle size={12} strokeWidth={1.5} color={scenario === 'disaster' ? 'var(--dw-live)' : 'var(--dw-fg-muted)'} />
          Disaster Mode
        </button>
      </div>

      {/* Mandatory Disclaimer Label (§23) */}
      {scenario === 'disaster' && (
        <div style={{
          padding: '6px 8px',
          background: 'rgba(245, 158, 11, 0.05)',
          border: '1px solid rgba(245, 158, 11, 0.18)',
          borderRadius: 'var(--dw-radius-sm)',
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 10,
          color: 'var(--dw-live)',
          lineHeight: 1.4,
        }}>
          Terrain intelligence / preliminary terrain assessment support.
        </div>
      )}
    </div>
  );
}

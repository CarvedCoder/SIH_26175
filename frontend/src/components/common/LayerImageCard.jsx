/**
 * DepthWizard — LayerImageCard
 *
 * Displays a single result layer image (RGB / Depth / DSM) with:
 * - Full-bleed image preview
 * - Label strip (top-left)
 * - Metadata strip (bottom): min/max values with correct units
 * - Disabled state when data is unavailable for this input type
 *
 * DESIGN.md: no rounded-pill, no glassmorphism, no big-number template.
 * Spec §53 Depth Result, §54 DSM Result, §34 Screenshot A.
 *
 * @param {{
 *   label: string,
 *   sublabel?: string,
 *   imageUrl: string|null,
 *   minValue?: number|null,
 *   maxValue?: number|null,
 *   units?: string,
 *   disabled?: boolean,
 *   disabledReason?: string,
 *   isLoading?: boolean,
 *   onClick?: () => void,
 *   selected?: boolean,
 * }} props
 */
export default function LayerImageCard({
  label,
  sublabel,
  imageUrl,
  minValue,
  maxValue,
  units = '',
  disabled = false,
  disabledReason,
  isLoading = false,
  onClick,
  selected = false,
}) {
  const hasRange = minValue != null && maxValue != null;

  function fmt(v) {
    if (v == null) return '—';
    // round to 2 decimal places, drop trailing zeros
    return Number(v.toFixed(2)).toString();
  }

  const borderColor = selected
    ? 'var(--dw-accent)'
    : disabled
    ? 'var(--dw-rim)'
    : 'var(--dw-rim)';

  return (
    <article
      aria-label={`${label} preview${disabled ? ' — unavailable' : ''}`}
      role={onClick && !disabled ? 'button' : undefined}
      tabIndex={onClick && !disabled ? 0 : undefined}
      onClick={() => !disabled && onClick?.()}
      onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !disabled) onClick?.(); }}
      style={{
        display: 'flex',
        flexDirection: 'column',
        flex: '1 1 260px',
        minWidth: 220,
        maxWidth: 420,
        background: 'var(--dw-surface)',
        border: `1px solid ${borderColor}`,
        borderRadius: 'var(--dw-radius)',
        overflow: 'hidden',
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'not-allowed' : onClick ? 'pointer' : 'default',
        outline: 'none',
        transition: 'border-color 150ms ease, opacity 150ms ease',
        boxShadow: selected ? `0 0 0 1px var(--dw-accent)` : 'none',
      }}
      onMouseEnter={e => {
        if (!disabled && onClick) e.currentTarget.style.borderColor = 'var(--dw-accent)';
      }}
      onMouseLeave={e => {
        if (!selected) e.currentTarget.style.borderColor = 'var(--dw-rim)';
      }}
      onFocus={e => {
        if (!disabled) e.currentTarget.style.outline = '2px solid var(--dw-accent)';
        e.currentTarget.style.outlineOffset = '2px';
      }}
      onBlur={e => { e.currentTarget.style.outline = 'none'; }}
    >
      {/* Image area */}
      <div style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '4/3',
        background: 'var(--dw-panel)',
        overflow: 'hidden',
        flexShrink: 0,
      }}>
        {isLoading && (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <LoadingShimmer />
          </div>
        )}

        {!isLoading && imageUrl && !disabled && (
          <img
            src={imageUrl}
            alt={label}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            loading="lazy"
          />
        )}

        {!isLoading && (!imageUrl || disabled) && (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: 6,
            padding: 16,
          }}>
            {/* Terrain placeholder glyph */}
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden="true">
              <polyline
                points="4,28 10,18 16,22 24,10 32,14"
                stroke="var(--dw-rim)"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <line x1="4" y1="28" x2="32" y2="28" stroke="var(--dw-rim)" strokeWidth="1" strokeLinecap="round" />
            </svg>
            {disabledReason && (
              <p style={{
                fontFamily: 'var(--dw-font-ui)',
                fontSize: 11,
                color: 'var(--dw-fg-ghost)',
                textAlign: 'center',
                margin: 0,
                lineHeight: 1.4,
              }}>
                {disabledReason}
              </p>
            )}
          </div>
        )}

        {/* Label chip — top left */}
        <div style={{
          position: 'absolute',
          top: 8,
          left: 8,
          padding: '2px 7px',
          background: 'rgba(7,9,14,0.75)',
          borderRadius: 'var(--dw-radius-sm)',
          backdropFilter: 'blur(4px)',
          WebkitBackdropFilter: 'blur(4px)',
        }}>
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 10,
            letterSpacing: '0.07em',
            color: 'var(--dw-fg)',
            fontWeight: 500,
          }}>
            {label}
          </span>
        </div>
      </div>

      {/* Metadata strip */}
      <div style={{
        padding: '10px 12px',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        borderTop: '1px solid var(--dw-rim)',
      }}>
        {sublabel && (
          <p style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            color: 'var(--dw-fg-muted)',
            margin: 0,
          }}>
            {sublabel}
          </p>
        )}

        {hasRange && !disabled && (
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <Stat label="MIN" value={fmt(minValue)} units={units} />
            {/* Mini gradient bar */}
            <div style={{
              flex: 1,
              height: 3,
              borderRadius: 2,
              background: 'linear-gradient(to right, var(--dw-fg-ghost), var(--dw-accent))',
              opacity: 0.6,
            }} />
            <Stat label="MAX" value={fmt(maxValue)} units={units} align="right" />
          </div>
        )}

        {disabled && disabledReason && (
          <p style={{
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 11,
            color: 'var(--dw-fg-ghost)',
            margin: 0,
          }}>
            {disabledReason}
          </p>
        )}
      </div>
    </article>
  );
}

function Stat({ label, value, units, align = 'left' }) {
  return (
    <div style={{ textAlign: align, flexShrink: 0 }}>
      <div style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 9,
        letterSpacing: '0.07em',
        textTransform: 'uppercase',
        color: 'var(--dw-fg-ghost)',
      }}>
        {label}
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 2, justifyContent: align === 'right' ? 'flex-end' : 'flex-start' }}>
        <span style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 12,
          color: 'var(--dw-fg)',
        }}>
          {value}
        </span>
        {units && (
          <span style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 10,
            color: 'var(--dw-fg-ghost)',
          }}>
            {units}
          </span>
        )}
      </div>
    </div>
  );
}

function LoadingShimmer() {
  return (
    <div style={{
      width: '100%',
      height: '100%',
      position: 'absolute',
      inset: 0,
      background: 'linear-gradient(90deg, var(--dw-surface) 25%, var(--dw-rim) 50%, var(--dw-surface) 75%)',
      backgroundSize: '200% 100%',
      animation: 'dw-shimmer 1.4s ease-in-out infinite',
    }}>
      <style>{`
        @keyframes dw-shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
        @media (prefers-reduced-motion: reduce) {
          .dw-shimmer { animation: none !important; }
        }
      `}</style>
    </div>
  );
}

/**
 * DepthWizard — Processing Status footer
 *
 * Shows "Estimating depth — Tile 3 / 12 (42.5%)" from live job data.
 * The percentage is ANIMATED: it glides toward the latest reported
 * progress (ease-out approach, never a hard jump), and while the worker
 * is busy between polls it creeps slowly — capped a few points past the
 * last real value so the number stays honest and never stalls.
 * No invented ETAs (spec §6).
 */
import { useEffect, useRef, useState } from 'react';
import { useApp } from '../../store/appStore.jsx';

/**
 * Ease the displayed percent toward `target`. Two regimes:
 *   behind the target  -> proportional glide (fast for big jumps),
 *   between polls      -> slow ~1.2%/s creep, capped at target + 4 so the
 *                         number keeps moving without outrunning the truth.
 */
function useSmoothProgress(target, running) {
  const [display, setDisplay] = useState(target ?? 0);
  const valRef  = useRef(target ?? 0);
  const rafRef  = useRef(null);
  const lastRef = useRef({ target, ts: performance.now() });

  useEffect(() => {
    if (target == null) return undefined;
    lastRef.current = { target, ts: performance.now() };

    const step = () => {
      const now = performance.now();
      const dt  = Math.min((now - lastRef.current.ts) / 1000, 0.1);
      const t   = lastRef.current.target;
      const cur = valRef.current;

      const ceiling = t + (running ? 4 : 0);
      const diff    = ceiling - cur;

      if (Math.abs(diff) >= 0.05) {
        const behindTarget = cur < t;
        const speed = behindTarget
          ? Math.max(Math.abs(t - cur) * 2.2, 8)   // percent per second
          : 1.2;                                    // idle creep
        const next = cur + Math.sign(diff) * Math.min(speed * dt, Math.abs(diff));
        valRef.current = next;
        setDisplay(next);
      }

      lastRef.current.ts = now;
      rafRef.current = requestAnimationFrame(step);
    };

    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, running]);

  return display;
}

export default function ProcessingStatus() {
  const { state } = useApp();
  const job = state.job;

  const running = job ? job.status !== 'completed' : false;
  const smooth  = useSmoothProgress(job?.progress, running);
  const shown   = job?.progress != null ? Math.min(Math.round(smooth * 10) / 10, 100) : null;

  if (!job) return null;

  const hasTiles = job.current_tile != null && job.total_tiles != null && job.total_tiles > 0;
  const stageName = job.message ?? (job.stage ? job.stage.replace(/_/g, ' ') : 'Processing');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <p
        aria-live="polite"
        aria-atomic="true"
        style={{
          fontFamily: 'var(--dw-font-data)',
          fontSize: 14,
          fontWeight: 500,
          color: 'var(--dw-fg)',
          margin: 0,
          letterSpacing: '0.03em',
        }}
      >
        {stageName}
        {hasTiles && (
          <span style={{ color: 'var(--dw-fg-muted)' }}>
            {' '}— Tile {job.current_tile} / {job.total_tiles}
          </span>
        )}
        {shown != null && (
          <span style={{ color: 'var(--dw-accent)', fontVariantNumeric: 'tabular-nums' }}>
            {' '}({shown % 1 === 0 ? shown : shown.toFixed(1)}%)
          </span>
        )}
      </p>

      {shown != null && (
        <div
          aria-hidden="true"
          style={{
            width: 260,
            height: 3,
            borderRadius: 1.5,
            background: 'var(--dw-rim)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${shown}%`,
              borderRadius: 1.5,
              background: 'var(--dw-accent)',
              transition: 'width 120ms linear',
            }}
          />
        </div>
      )}
    </div>
  );
}

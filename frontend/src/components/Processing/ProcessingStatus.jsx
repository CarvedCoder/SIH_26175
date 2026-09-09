/**
 * DepthWizard — Processing Status footer
 *
 * Shows "Processing scene… Tile N / M" from live job data.
 * No invented ETAs (spec §6).
 */
import { useApp } from '../../store/appStore.jsx';

export default function ProcessingStatus() {
  const { state } = useApp();
  const job = state.job;
  if (!job) return null;

  const hasTiles = job.current_tile != null && job.total_tiles != null && job.total_tiles > 0;
  const stageName = job.message ?? (job.stage ? job.stage.replace(/_/g, ' ') : 'Processing');

  return (
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
      {job.progress != null && (
        <span style={{ color: 'var(--dw-accent)' }}>
          {' '}({job.progress}%)
        </span>
      )}
    </p>
  );
}

/**
 * DepthWizard — useProcessing hook
 *
 * Polls GET /jobs/{jobId} every 2.5s while job is active.
 * Stops immediately on completed / failed / cancelled.
 * Calls the correct state-machine actions on each update.
 *
 * Spec §49, §70. DECISIONS.md §D08.
 */
import { useEffect, useRef, useCallback } from 'react';
import { useApp } from '../store/appStore.jsx';
import { getJobStatus, cancelJob } from '../api/processing.js';
import { getResults } from '../api/results.js';

const TERMINAL = new Set(['completed', 'failed', 'cancelled']);
const POLL_MS = 2500;

export function useProcessing() {
  const { state, actions } = useApp();
  const intervalRef = useRef(null);
  const activeRef   = useRef(false);

  const stop = useCallback(() => {
    clearInterval(intervalRef.current);
    activeRef.current = false;
  }, []);

  const poll = useCallback(async (jobId) => {
    try {
      const job = await getJobStatus(jobId);
      actions.processingUpdate(job);

      if (job.status === 'completed') {
        stop();
        // Fetch result metadata so the dashboard knows which layers are available
        try {
          const results = await getResults(state.scene?.scene_id ?? job.scene_id);
          actions.processingDone(results);
        } catch (e) {
          actions.processingDone(null); // still move forward; results can be fetched later
        }
      } else if (job.status === 'failed') {
        stop();
        actions.processingFail({
          code:        'PROCESSING_FAILED',
          message:     job.message ?? 'Processing failed.',
          recoverable: true,
        });
      } else if (job.status === 'cancelled') {
        stop();
        actions.retry(); // return to SCENE_READY so user can try again
      }
    } catch (err) {
      // Network error during poll — don't stop; try again next interval
      console.warn('[useProcessing] poll error', err);
    }
  }, [actions, state.scene, stop]);

  useEffect(() => {
    const jobId = state.jobId;
    if (!jobId || state.status !== 'PROCESSING') return;

    activeRef.current = true;
    poll(jobId); // immediate first check
    intervalRef.current = setInterval(() => poll(jobId), POLL_MS);

    return stop;
  }, [state.jobId, state.status, poll, stop]);

  const cancel = useCallback(async () => {
    if (!state.jobId) return;
    try {
      await cancelJob(state.jobId);
      // poll will pick up the 'cancelled' status on the next cycle
    } catch (err) {
      console.warn('[useProcessing] cancel error', err);
    }
  }, [state.jobId]);

  return { cancel };
}

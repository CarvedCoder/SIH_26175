/**
 * DepthWizard — useFpsMeter
 *
 * Measures render-loop FPS from requestAnimationFrame deltas. Independent
 * of the OGL render loop itself, so it works in every camera mode without
 * touching TerrainCanvas internals.
 */
import { useState, useEffect, useRef } from 'react';

const SAMPLE_WINDOW = 30; // frames averaged per reading

export function useFpsMeter(enabled = true) {
  const [fps, setFps] = useState(0);
  const frames = useRef([]);
  const rafId = useRef(0);
  const last = useRef(0);

  useEffect(() => {
    if (!enabled) return undefined;
    last.current = performance.now();
    const tick = (t) => {
      const dt = t - last.current;
      last.current = t;
      if (dt > 0 && dt < 500) {
        frames.current.push(1000 / dt);
        if (frames.current.length > SAMPLE_WINDOW) frames.current.shift();
        if (frames.current.length === SAMPLE_WINDOW) {
          const avg = frames.current.reduce((a, b) => a + b, 0) / frames.current.length;
          setFps(Math.round(avg));
        }
      }
      rafId.current = requestAnimationFrame(tick);
    };
    rafId.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId.current);
  }, [enabled]);

  return fps;
}

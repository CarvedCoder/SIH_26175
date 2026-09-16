/**
 * DepthWizard — useMediaQuery
 *
 * Reactive CSS media-query hook powering the responsive layout rules of
 * UX Spec §29:
 *
 *   Desktop  (>=1024px): 3D terrain + minimap + analysis panel
 *   Tablet   (768–1023px): 3D terrain + collapsible panels
 *   Mobile   (<768px):  3D terrain + bottom-sheet controls
 *
 * SSR-safe and listener-cleaned; `prefers-reduced-motion` also routes
 * through here so components share one media-state source.
 */
import { useState, useEffect } from 'react';

/**
 * @param {string} query - CSS media query, e.g. '(max-width: 767px)'
 * @returns {boolean} whether the query currently matches
 */
export function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined;
    const mql = window.matchMedia(query);
    const onChange = (e) => setMatches(e.matches);
    setMatches(mql.matches);
    mql.addEventListener?.('change', onChange);
    return () => mql.removeEventListener?.('change', onChange);
  }, [query]);

  return matches;
}

/** Viewport breakpoint helper (UX §29). */
export function useViewport() {
  const isMobile = useMediaQuery('(max-width: 767px)');
  const isTablet = useMediaQuery('(min-width: 768px) and (max-width: 1023px)');
  const isTouch = useMediaQuery('(hover: none) and (pointer: coarse)');
  return {
    isMobile,       // bottom-sheet controls, hide secondary overlays
    isTablet,       // collapsible panels only
    isDesktop: !isMobile && !isTablet,
    isTouch,        // joystick / touch affordances
  };
}

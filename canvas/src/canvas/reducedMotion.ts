/**
 * prefers-reduced-motion for JS-driven tldraw camera moves.
 * CSS transitions are handled by the @media block in ui2.tokens.css; camera
 * animations run through the tldraw editor API and need this runtime check.
 */
export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Camera animation duration (ms) honoring reduced-motion (0 = instant). */
export function cameraDuration(ms = 200): number {
  return prefersReducedMotion() ? 0 : ms;
}

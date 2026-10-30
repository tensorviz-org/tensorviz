import React from 'react';

/** Labels while the header fits on one row; icons once labels would wrap the controls
 *  or squeeze the model name. Measured from the rendered header rather than a fixed
 *  breakpoint, because what the header holds differs between hosts and models. */
export function useCompactHeader(header: HTMLElement | null): boolean {
  const [compact, setCompact] = React.useState(false);
  const compactRef = React.useRef(compact);
  compactRef.current = compact;
  // The header width at which labels last did not fit; a wider header tries them again.
  const failedAt = React.useRef(Infinity);
  const fits = React.useCallback(() => {
    const name = header?.querySelector('.local-browser-model-name strong');
    const actions = header?.querySelector('.local-browser-header-actions');
    return !(name && name.scrollWidth > name.clientWidth + 1)
      && !(actions && actions.getBoundingClientRect().height > 46);
  }, [header]);
  // Labels were just rendered (first paint, or a retry after widening): keep them only if they fit.
  React.useLayoutEffect(() => {
    if (header && !compact && !fits()) { failedAt.current = header.clientWidth; setCompact(true); }
  }, [header, compact, fits]);
  React.useEffect(() => {
    if (!header || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => {
      if (!compactRef.current) {
        if (!fits()) { failedAt.current = header.clientWidth; setCompact(true); }
      } else if (header.clientWidth > failedAt.current) setCompact(false);
    });
    // Controls wrapping, or the model name shrinking, show up as their own resize.
    for (const element of [header, header.querySelector('.local-browser-header-actions'), header.querySelector('.local-browser-model-name strong')]) {
      if (element) observer.observe(element);
    }
    return () => observer.disconnect();
  }, [header, fits]);
  return compact;
}

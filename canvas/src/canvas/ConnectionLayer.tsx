/**
 * ConnectionLayer — the drag-to-connect ghost, mounted through tldraw's
 * OnTheCanvas slot so it lives inside .tl-html-layer and inherits the camera
 * transform. Everything below is therefore in PAGE coordinates: no manual
 * projection, and the ghost stays correct at every zoom.
 *
 * The curve comes from the same edgeControlPoints/edgeCurveBounds the renderer
 * uses, so the preview traces exactly where the committed model-edge will land.
 */
import React from 'react';
import { useValue } from 'tldraw';
import { edgeControlPoints, edgeCurveBounds } from './edgeCurve';
import { connectionDrag } from './connectionDrag';

// Matches EDGE_PADDING in useTldrawBridge — the stroke, its glow and the ring
// all need room outside the curve's own extrema.
const PAD = 28;

export const ConnectionLayer: React.FC = () => {
  const drag = useValue(connectionDrag);
  if (!drag) return null;

  // Snap the loose end to the candidate port so the ghost lands where the real
  // edge will, not under the cursor.
  const loose = drag.candidate ? drag.candidate.point : drag.to;
  const start = drag.anchorSide === 'source' ? drag.from : loose;
  const end = drag.anchorSide === 'source' ? loose : drag.from;

  const { minX, minY, maxX, maxY } = edgeCurveBounds(start.x, start.y, end.x, end.y, PAD);
  const w = Math.max(1, maxX - minX);
  const h = Math.max(1, maxY - minY);
  const { c1x, c1y, c2x, c2y } = edgeControlPoints(start.x, start.y, end.x, end.y);
  const path =
    `M ${start.x - minX} ${start.y - minY} ` +
    `C ${c1x - minX} ${c1y - minY}, ${c2x - minX} ${c2y - minY}, ${end.x - minX} ${end.y - minY}`;

  const invalid = Boolean(drag.candidate && !drag.candidate.result.ok);
  const stateClass = invalid ? ' invalid' : drag.candidate ? ' valid' : '';

  return (
    <>
      <svg
        className="model-connect-svg"
        style={{ position: 'absolute', left: minX, top: minY, overflow: 'visible' }}
        width={w}
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        aria-hidden="true"
      >
        <path className={`model-connect-path${stateClass}`} d={path} />
      </svg>
      {drag.candidate && (
        <div
          className={`model-connect-ring${stateClass}`}
          style={{ position: 'absolute', left: drag.candidate.point.x, top: drag.candidate.point.y }}
        />
      )}
    </>
  );
};

import React from 'react';
import type { ModelNodeData } from '../types';
import { moduleBoundaryLayout } from '../services/modulePresentation';

export function ModuleBoundaryRows({ data, width = 250 }: { data: ModelNodeData; width?: number }) {
  const rows = moduleBoundaryLayout(data, width);
  if (!rows.length) return null;
  return <div className="module-boundary-shapes" role="group" aria-label="Input and output shapes">
    {rows.map(row => {
      // portType separates the tensor dimensions and dtype with this delimiter.
      // Keep the complete value in the title and text, only changing its emphasis.
      return <div className="module-boundary-row" data-direction={row.label.startsWith('Out') ? 'out' : 'in'}
        data-compact={row.compact || undefined} key={row.id} style={{ height: row.height, minHeight: row.height }}>
        <div className="module-boundary-label card-content-lines" title={row.path ?? row.label}>{row.labelLines.length === 1 ? row.labelLines[0] : row.labelLines.map((line, index) => <span key={index}>{line}</span>)}</div>
        <div className="module-boundary-value card-content-lines" title={row.value}>
          {row.valueLines.map((line, index) => <span key={index}>{line}</span>)}
        </div>
      </div>;
    })}
  </div>;
}

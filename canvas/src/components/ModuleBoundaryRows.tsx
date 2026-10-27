import React from 'react';
import type { ModelNodeData } from '../types';
import { compactBoundaryLabel, moduleBoundaryRows } from '../services/modulePresentation';

export function ModuleBoundaryRows({ data }: { data: ModelNodeData }) {
  const rows = moduleBoundaryRows(data);
  if (!rows.length) return null;
  return <div className="module-boundary-shapes" role="group" aria-label="Input and output shapes">
    {rows.map(row => {
      // portType separates the tensor dimensions and dtype with this delimiter.
      // Keep the complete value in the title and text, only changing its emphasis.
      const [value, dtype] = row.value.split(' · ');
      return <div className="module-boundary-row" data-direction={row.label.startsWith('Out') ? 'out' : 'in'}
        data-compact={compactBoundaryLabel(row.label) || undefined} key={row.id}>
        <div className="module-boundary-label" title={row.path ?? row.label}>{row.label}</div>
        <div className="module-boundary-value" title={row.value}>
          <span className="module-boundary-dimensions">{value}</span>
          {dtype && <span className="module-boundary-dtype">{' · '}{dtype}</span>}
        </div>
      </div>;
    })}
  </div>;
}

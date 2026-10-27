import React from 'react';
import { Maximize2, Minimize2 } from 'lucide-react';
import CanvasSegmentedControl from './CanvasSegmentedControl';
import type { GraphDetailLevel } from '../services/graphDetailLevels';
import { dispatchBuilderCommand } from './ui2/builderCommand';
import './GraphDetailToggle.css';

export default function GraphDetailToggle({ level, compact = false }: { level?: GraphDetailLevel; compact?: boolean }) {
  if (!level) return null;
  return <CanvasSegmentedControl label="Graph detail" value={level} compact={compact}
    className={`graph-detail-toggle${compact ? ' graph-detail-toggle-compact' : ''}`}
    onChange={value => dispatchBuilderCommand('set-graph-detail', { level: value })}
    options={[
      { value: 'layers', label: 'Layers', icon: <Minimize2 size={14} aria-hidden="true" />,
        help: 'Show layers with their tensor shapes. Internal operations stay folded; this does not run the model.' },
      { value: 'operations', label: 'All operations', icon: <Maximize2 size={14} aria-hidden="true" />,
        help: 'Show the individual PyTorch operations recorded during the captured call. This does not run the model.' },
    ]} />;
}

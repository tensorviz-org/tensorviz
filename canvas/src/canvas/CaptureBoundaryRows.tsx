import React from 'react';
import type { CapturePort } from '../types';
import { portLabel, portType } from './capturePorts';
import { boundaryTextLayout } from './cardContentLayout';

/** Typed root interfaces, including scalar/literal and empty-container outputs. */
export function CaptureBoundaryRows({ ports, width = 200 }: { ports: CapturePort[]; width?: number }) {
  return <div className="ui2-node-list" data-testid="capture-boundary-rows">
    {ports.length === 0 && <span className="ui2-node-shape-row">No boundary value recorded</span>}
    {ports.map((port, index) => {
      const value = portType(port);
      const layout = boundaryTextLayout(portLabel(port, index), value, width, false, true);
      return <div key={port.id} className="capture-boundary-row" data-direction={port.direction} style={{ height: layout.height }}>
        <div className="ui2-node-subtitle card-content-lines" title={portLabel(port, index)}>{layout.labelLines.map((line, i) => <span key={i}>{line}</span>)}</div>
        <div className="capture-boundary-value ui2-node-shape card-content-lines" title={value}>
          {layout.valueLines.map((line, i) => <span key={i}>{line}</span>)}
        </div>
      </div>;
    })}
  </div>;
}

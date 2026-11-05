import React from 'react';
import type { CapturePort } from '../types';
import { portLabel, portType } from './capturePorts';

/** Typed root interfaces, including scalar/literal and empty-container outputs. */
export function CaptureBoundaryRows({ ports }: { ports: CapturePort[] }) {
  return <div className="ui2-node-list" data-testid="capture-boundary-rows">
    {ports.length === 0 && <span className="ui2-node-shape-row">No boundary value recorded</span>}
    {ports.map((port, index) => {
      const value = portType(port);
      const [dimensions, dtype] = value.split(' · ');
      return <div key={port.id} className="capture-boundary-row" data-direction={port.direction}>
        <div className="ui2-node-subtitle" title={portLabel(port, index)}>{portLabel(port, index)}</div>
        <div className="capture-boundary-value ui2-node-shape" title={value}>
          <span className="capture-boundary-dimensions">{dimensions}</span>
          {dtype && <span className="capture-boundary-dtype">{' · '}{dtype}</span>}
        </div>
      </div>;
    })}
  </div>;
}

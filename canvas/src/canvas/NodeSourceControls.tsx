import React from 'react';
import { FileCode2 } from 'lucide-react';
import type { ModelNodeData } from '../types';
import { currentProjection, localBrowserMode, reportNodeClick } from '../ide/projection';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { sourceLinkLayout } from './cardContentLayout';

export function NodeSourceLink({ nodeId: shownId, data, width }: { nodeId: string; data: ModelNodeData; width?: number }) {
  const copy = React.useContext(LocalBrowserContext)?.sourceAction === 'copy';
  const location = data.sourceLocation;
  // A replaced captured call opens the assignment that replaced it, not the old class.
  const nodeId = data.sourceReplacement?.key ?? shownId;
  if (!location || !localBrowserMode()) return null;
  const layout = width ? sourceLinkLayout(location, width) : undefined;
  const label = layout ? <span className="card-content-lines">{layout.lines.map((line, index) => <span key={index}>{line}</span>)}</span>
    : <><span>{location.file.split(/[\\/]/).pop()}</span><span className="node-source-line">:{location.line}</span></>;
  // The host reveals only spans it projected. A display-only assignment without
  // one still names its line, as a plain reference rather than a dead link.
  if (!currentProjection()?.spanmap?.[nodeId]) return data.sourceAssignment ? <span className="node-source-location" data-static=""
    title={`Set at ${location.file}:${location.line}`} style={layout ? { height: layout.height, minHeight: layout.height } : undefined}><FileCode2 size={11} />{label}</span> : null;
  return <button type="button" className="node-source-location" aria-label={`${location.file}:${location.line}`} title={copy ? `Copy ${location.file}:${location.line}` : `Open ${location.file}:${location.line} in VS Code`}
    style={layout ? { height: layout.height, minHeight: layout.height } : undefined}
    onPointerDown={event => event.stopPropagation()} onClick={event => { event.stopPropagation(); reportNodeClick(nodeId); }}>
    <FileCode2 size={11} />{label}
  </button>;
}

import React from 'react';
import { FileCode2 } from 'lucide-react';
import type { ModelNodeData } from '../types';
import { currentProjection, localBrowserMode, reportNodeClick } from '../ide/projection';
import { LocalBrowserContext } from '../ide/localBrowserContext';

export function NodeSourceLink({ nodeId: shownId, data }: { nodeId: string; data: ModelNodeData }) {
  const copy = React.useContext(LocalBrowserContext)?.sourceAction === 'copy';
  const location = data.sourceLocation;
  // A replaced captured call opens the assignment that replaced it, not the old class.
  const nodeId = data.sourceReplacement?.key ?? shownId;
  if (!location || !localBrowserMode()) return null;
  // The host reveals only spans it projected. A display-only assignment without
  // one still names its line, as a plain reference rather than a dead link.
  if (!currentProjection()?.spanmap?.[nodeId]) return data.sourceAssignment ? <span className="node-source-location" data-static=""
    title={`Set at ${location.file}:${location.line}`}><FileCode2 size={11} /><span>{location.file.split(/[\\/]/).pop()}</span><span className="node-source-line">:{location.line}</span></span> : null;
  return <button type="button" className="node-source-location" aria-label={`${location.file}:${location.line}`} title={copy ? `Copy ${location.file}:${location.line}` : `Open ${location.file}:${location.line} in VS Code`}
    onPointerDown={event => event.stopPropagation()} onClick={event => { event.stopPropagation(); reportNodeClick(nodeId); }}>
    <FileCode2 size={11} /><span>{location.file.split(/[\\/]/).pop()}</span><span className="node-source-line">:{location.line}</span>
  </button>;
}

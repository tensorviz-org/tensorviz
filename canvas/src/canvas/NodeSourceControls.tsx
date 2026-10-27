import React from 'react';
import CanvasTooltip from '../components/CanvasTooltip';
import { FileCode2, Pencil } from 'lucide-react';
import type { ModelNodeData } from '../types';
import { currentProjection, localBrowserMode, reportNodeClick } from '../ide/projection';
import { LocalBrowserContext } from '../ide/localBrowserContext';

export const NodeEditContext = React.createContext<((nodeId: string) => void) | null>(null);

export function NodeEditButton({ nodeId, data }: { nodeId: string; data: ModelNodeData }) {
  const edit = React.useContext(NodeEditContext);
  if (!edit) return null;
  const supported = !data.captureRef || !!data.sourceEditCapabilities?.length;
  return <CanvasTooltip text={supported ? 'Edit this node, then review the proposed source diff in VS Code before applying it.' : 'No supported source edit for this call. Use its source link to edit in VS Code.'}><button type="button" className="node-inline-edit-button" aria-label={`Edit ${data.captureLabel || nodeId}`}
    disabled={!supported} title={supported ? 'Edit this node and review the source diff' : 'No supported source edit for this call. Use its source link to edit in VS Code.'}
    onPointerDown={event => event.stopPropagation()} onClick={event => { event.stopPropagation(); edit(nodeId); }}>
    <Pencil size={13} />
  </button></CanvasTooltip>;
}

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

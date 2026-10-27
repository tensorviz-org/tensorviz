import React from 'react';
import type { ModelNode } from '../types';
import { SourceEditForm } from './SourceEditForm';
import { NodeSourceLink } from '../canvas/NodeSourceControls';
import './SourceEditForm.css';

/** Keep the review's selection alive when Apply replaces the captured graph. */
export function SourceEditSession({ nodes, onClose }: { nodes: ModelNode[]; onClose(): void }) {
  const [selection] = React.useState(nodes);
  const panel = React.useRef<HTMLElement>(null);
  React.useEffect(() => {
    const opener = document.activeElement;
    panel.current?.querySelector<HTMLElement>('.source-edit-form input, .source-edit-form select')?.focus({ preventScroll: true });
    return () => { if (opener instanceof HTMLElement && opener.isConnected) opener.focus({ preventScroll: true }); };
  }, []);
  const node = selection[0];
  return <section ref={panel} className="node-source-editor" aria-label="Edit node source"
    onPointerDown={event => event.stopPropagation()} onWheel={event => event.stopPropagation()}
    onKeyDown={event => { event.stopPropagation(); if (event.key === 'Escape') onClose(); }}>
    <header><div><strong>{selection.length > 1 ? `${selection.length} selected layers` : node?.data.captureLabel || node?.id}</strong>
      {node && <NodeSourceLink nodeId={node.id} data={node.data} />}</div>
      <button type="button" aria-label="Close source editor" onClick={onClose}>×</button></header>
    <SourceEditForm nodes={selection} compact />
  </section>;
}

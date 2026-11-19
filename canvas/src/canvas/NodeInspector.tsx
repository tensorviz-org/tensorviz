import React from 'react';
import { ArrowLeft, Expand, ListTree, MessageCircle, Pencil } from 'lucide-react';
import type { ModelEdge, ModelNode } from '../types';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { BlockSummary } from './BlockSummary';
import { CaptureDetails } from './CaptureDetails';
import { NodeUnderstanding } from './NodeUnderstanding';
import { NodeExplanationContext } from './NodeExplanationContext';
import { LocalSourceEditor } from './LocalSourceEditor';
import { hasDeclarationEdit } from '../ide/directSourceEdits';
import { NodeActionButton } from './NodeActionToolbar';
import { ExplainNodeButton } from './ExplainNodeButton';
import { useNodeExplanationAction, type NodeExplanationAction } from './useNodeExplanationAction';
import './nodeInspector.css';

export type NodeInspectionAction = { nodeId: string; section: 'details' | 'capture' | 'edit' };

/** A deliberate inspection surface; merely selecting a node never mounts it. */
export function NodeInspector({ node, title, nodes, edges, onNavigate, onBack, onOpenInside, action, explanationAction }: {
  node: ModelNode;
  title?: string;
  nodes: readonly ModelNode[];
  edges: readonly ModelEdge[];
  onNavigate(id: string): void;
  onBack(): void;
  onOpenInside?(): void;
  action?: NodeInspectionAction;
  explanationAction?: NodeExplanationAction;
}) {
  const local = React.useContext(LocalBrowserContext);
  const explanations = React.useContext(NodeExplanationContext);
  const [section, setSection] = React.useState(action?.section ?? 'details');
  const editing = section === 'edit';
  const ownExplanationAction = useNodeExplanationAction(node.id);
  const explainAction = explanationAction ?? ownExplanationAction;
  React.useEffect(() => {
    if (!action) return;
    setSection(action.section);
  }, [action]);
  const boundary = !!node.data.focusBoundary || node.type === 'inputNode' || node.type === 'outputNode';
  const editable = !boundary && !!local?.sourceEdits && !local.reviewing && !local.page.greyed
    && (hasDeclarationEdit(node.data) || node.data.sourceEditCapabilities?.some(cap => cap.command === 'delete_unary_op'));
  return <aside className="node-inspector" aria-label="Node inspector" onKeyDownCapture={event => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); if (section !== 'details') setSection('details'); else onBack(); }
  }}>
    <header className="node-inspector-header">
      <div className="node-inspector-heading">
        <button type="button" onClick={onBack} aria-label="Back to graph"><ArrowLeft size={15} aria-hidden="true" />Back to graph</button>
        <span>Node inspector</span>
      </div>
      {title && <h2>{title}</h2>}
      <div className="node-inspector-tools" role="group" aria-label="Component actions">
        {!boundary && explanations?.model && <ExplainNodeButton action={explainAction} />}
        {explanations && local?.sourceAction !== 'copy' && <NodeActionButton label="Discuss in chat" icon={MessageCircle} onClick={() => {
          window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'report-selection' } }));
          window.dispatchEvent(new CustomEvent('tensorviz-agent-compose'));
        }} />}
        {node.data.captureRef && local?.captures && <NodeActionButton label="Captured calls & tensors" icon={ListTree} onClick={() => setSection('capture')} />}
        {onOpenInside && <NodeActionButton label="Open inside" icon={Expand} onClick={onOpenInside} />}
        {editable && <NodeActionButton label="Edit source" icon={Pencil} onClick={() => setSection('edit')} />}
      </div>
    </header>
    <div className="node-inspector-body" tabIndex={0}>
      {editing && editable ? <>
        <button type="button" onClick={() => setSection('details')} aria-label="Back to details"><ArrowLeft size={14} aria-hidden="true" />Details</button>
        <p className="node-inspector-edit-purpose">Changes prepare a source diff. Review and accept it before any source is written.</p>
        <LocalSourceEditor key={node.id} nodeId={node.id} data={node.data} />
      </> : section === 'capture' && node.data.captureRef && local?.captures ? <CaptureDetails inline initiallyOpen hideTrigger
        key={`${node.data.captureRef.captureId}:${node.id}`} reference={node.data.captureRef}
        label={title ?? ''} client={local.captures} stale={local.page.greyed || local.page.syncState === 'stale'}
        onClose={() => setSection('details')} /> : <>
      {!node.data.focusBoundary && <BlockSummary node={node} nodes={nodes} edges={edges} showIdentity={false} />}
      <NodeUnderstanding key={node.id} node={node} nodes={nodes} edges={edges} onNavigate={onNavigate} hideActions detailsLabel="Source and configuration"
        explanationExecution={explainAction.execution} />
      </>}
    </div>

  </aside>;
}

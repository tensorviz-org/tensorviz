import React from 'react';
import type { ModelEdge, ModelNode, ModelNodeData } from '../types';
import { BLOCK_LABELS, BLOCK_ORDER, classifyNode, moduleComposition } from './blockFamily';
import { BlockGlyph, blockTitle } from './BlockIdentity';
import './blockSummary.css';

/** The exact class or operator the family was resolved from, for the inspector. */
export function blockIdentity(data: ModelNodeData): string | undefined {
  const declared = data.parameters?._class;
  if (typeof declared === 'string' && declared) return declared;
  // A captured call carries its qualified class; a source module its template.
  if (data.captureRef?.kind === 'invocation' && data.layerType?.name) return data.layerType.name;
  if (data.layerType?.id === 'module' && data.moduleTemplateId) return String(data.moduleTemplateId);
  return data.layerType?.name || data.layerType?.id || undefined;
}

/** Family glyph and name, exact class, and a composite's verified composition. */
export function BlockSummary({ node, nodes, edges, showIdentity = true }: { node: ModelNode; nodes: readonly ModelNode[]; edges: readonly ModelEdge[]; showIdentity?: boolean }) {
  const block = classifyNode(node);
  const identity = block.kind === 'input' || block.kind === 'output' ? undefined : blockIdentity(node.data);
  const composition = block.kind === 'composite' || block.kind === 'container'
    ? moduleComposition(node.id, [...nodes], [...edges]) : [];
  return <div className="graph-block-summary" data-block={block.kind}>
    <span className="graph-block-family"><BlockGlyph kind={block.kind} size={14} />{blockTitle(block)}</span>
    {showIdentity && identity && <code className="graph-block-class" title={identity}>{identity}</code>}
    {composition.length > 0 && <span className="graph-block-composition" title="From the child blocks recorded in this graph">
      {composition.map((kind, index) => <React.Fragment key={`${kind}-${index}`}>{index > 0 && <span aria-hidden="true"> → </span>}
        <span className="graph-block-step"><BlockGlyph kind={kind} size={12} />{BLOCK_LABELS[kind]}</span></React.Fragment>)}
    </span>}
  </div>;
}

/** A compact key containing only the families shown in the current graph. */
export function FamilyLegend({ nodes }: { nodes: readonly ModelNode[] }) {
  const kinds = React.useMemo(() => {
    const present = new Set(nodes.filter(node => !node.hidden).map(node => classifyNode(node).kind));
    return BLOCK_ORDER.filter(kind => present.has(kind));
  }, [nodes]);
  if (!kinds.length) return null;
  return <details className="graph-family-legend">
    <summary aria-label={`Block families: ${kinds.map(kind => BLOCK_LABELS[kind]).join(', ')}`}>
      Blocks<span className="graph-family-legend-glyphs" aria-hidden="true">{kinds.map(kind => <BlockGlyph key={kind} kind={kind} size={12} />)}</span>
    </summary>
    <ul>{kinds.map(kind => <li key={kind}><BlockGlyph kind={kind} size={13} />{BLOCK_LABELS[kind]}</li>)}</ul>
  </details>;
}

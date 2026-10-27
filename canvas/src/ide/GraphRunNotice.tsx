import React from 'react';
import type { GraphSetupAction, ProjectionPayload } from './protocol.generated';
import type { LocalBrowserPageState } from './localBrowserState';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';
import { showsPreviousRunContext } from './graphDebugPresentation';
import type { GraphDocumentV2 } from '../types/graphDocument';

export function graphFailureNode(projection: ProjectionPayload | null): string | undefined {
  if (!projection?.traced || projection.greyed) return undefined;
  const document = projection.document as { semantic?: { metadata?: { capture?: { failureNodeId?: string } }; nodes?: { node_id: string }[] } };
  const id = document?.semantic?.metadata?.capture?.failureNodeId ?? projection.traceError?.nodeId;
  return id && document?.semantic?.nodes?.some(node => node.node_id === id) ? id : undefined;
}

/** The module path of the node where capture stopped, when the graph still names it. */
export function failingLayerLabel(projection: ProjectionPayload | null): string | undefined {
  const id = graphFailureNode(projection);
  if (!id || !projection) return undefined;
  const nodes = (projection.document as { semantic?: { nodes?: { node_id: string; op?: string; params?: Record<string, unknown>; metadata?: Record<string, unknown> }[] } }).semantic?.nodes;
  const node = nodes?.find(item => item.node_id === id);
  const candidates = [node?.params?._module_path, node?.metadata?.captureLabel, node?.metadata?.module_path, id];
  const label = candidates.find(value => typeof value === 'string' && value.trim().length > 0);
  return typeof label === 'string' ? label : undefined;
}

export function GraphRunNotice({ page, connected, onAction, standalone = false }: { page: LocalBrowserPageState; connected: boolean; standalone?: boolean;
  onAction(action: GraphSetupAction): void }) {
  const state = page.opening;
  if (state?.phase !== 'failed' || state.failure?.kind === 'setup') return null;
  const nodeId = state.hasCapture ? graphFailureNode(page.projection) : undefined;
  const layer = state.hasCapture ? failingLayerLabel(page.projection) : undefined;
  const document = page.projection?.document as GraphDocumentV2 | undefined;
  const previousContext = showsPreviousRunContext(page.projection, document?.semantic?.nodes?.length ?? 0);
  return <div className="graph-run-notice" role="status">
    <div><strong>{state.failure?.kind === 'environment' ? 'Environment needs attention' : layer ? `Capture stopped at ${layer}` : 'Capture stopped'}</strong>
      <span>{state.detail}</span>
      {previousContext && <small>Dashed context is from the previous run. Changed parameters are observations, not a proven cause.</small>}
    </div>
    {nodeId && <button type="button" onClick={() => dispatchBuilderCommand('focus-node', { nodeId })}>Show failure</button>}
    {!standalone && state.failure?.kind === 'environment' && <button type="button" disabled={!connected}
      onClick={() => onAction({ kind: 'environment' })}>Python environment</button>}
    {!standalone && state.failure?.kind === 'source' && <button type="button" disabled={!connected}
      onClick={() => onAction({ kind: 'save_sources' })}>Save & retry</button>}
    <button type="button" disabled={!connected} onClick={() => onAction({ kind: 'agent_help' })}>
      {standalone ? 'Copy MCP investigation request' : 'Use Agent'}</button>
  </div>;
}

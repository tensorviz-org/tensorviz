/* GENERATED from packages/contracts/src/graphSetupWorkflow.ts; npm run protocol:sync. */
import type { GraphOpeningState } from './protocol.generated';

export type GraphSetupNextStep =
  | { kind: 'wait'; tool: 'get_graph_setup'; waitMs: 10000 }
  | { kind: 'environment'; tool: 'configure_graph_environment' }
  | { kind: 'inputs'; tool: 'suggest_graph_inputs' }
  | { kind: 'execution_approval' }
  | { kind: 'complete' }
  | { kind: 'blocked'; reason: 'failed' | 'cancelled' }
  | { kind: 'source_only' };

export interface GraphSetupWorkflow {
  stages: {
    python: 'unchecked' | 'pending' | 'ready';
    inputs: 'unchecked' | 'needed' | 'saved';
    capture: 'not_started' | 'running' | 'awaiting_approval' | 'ready' | 'failed' | 'cancelled';
  };
  nextStep: GraphSetupNextStep;
}

/** One projection for host tools and UI. Drafts and agent text are not readiness evidence. */
export function graphSetupWorkflow(state: GraphOpeningState): GraphSetupWorkflow {
  const python = state.environment?.reason === 'ready' ? 'ready' : state.environment ? 'pending' : 'unchecked';
  const invocationChecked = state.phase === 'capturing' || state.phase === 'ready'
    || (state.phase === 'awaiting_execution' && !!state.execution && state.execution.kind !== 'prepare')
    || (state.phase === 'failed' && (state.failure?.kind === 'capture' || state.failure?.kind === 'execution'));
  const inputs = state.configFileExists === false ? 'needed' : invocationChecked ? 'saved'
    : state.phase === 'needs_setup' && python !== 'pending' ? 'needed' : 'unchecked';
  const capture = state.phase === 'ready' ? state.hasCapture ? 'ready' : 'failed'
    : state.phase === 'awaiting_execution' ? 'awaiting_approval'
      : state.phase === 'capturing' ? 'running'
        : state.phase === 'failed' ? 'failed'
          : state.phase === 'cancelled' ? 'cancelled' : 'not_started';
  const stages: GraphSetupWorkflow['stages'] = { python, inputs, capture };
  if (state.phase === 'source_only') return { stages, nextStep: { kind: 'source_only' } };
  if (state.phase === 'cancelled') return { stages, nextStep: { kind: 'blocked', reason: 'cancelled' } };
  if (state.phase === 'failed' || (state.phase === 'ready' && !state.hasCapture)) return { stages, nextStep: { kind: 'blocked', reason: 'failed' } };
  if (state.phase === 'ready') return { stages, nextStep: { kind: 'complete' } };
  if (state.phase === 'awaiting_execution') return { stages, nextStep: { kind: 'execution_approval' } };
  if (state.phase === 'resolving' || state.phase === 'saving' || state.phase === 'capturing') {
    return { stages, nextStep: { kind: 'wait', tool: 'get_graph_setup', waitMs: 10000 } };
  }
  return { stages, nextStep: python === 'pending'
    ? { kind: 'environment', tool: 'configure_graph_environment' } : { kind: 'inputs', tool: 'suggest_graph_inputs' } };
}

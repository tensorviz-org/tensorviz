import React from 'react';
import { NodeExplanationContext, sameNodeExplanationResult } from './NodeExplanationContext';
import type { ModelExplanationState } from '../ide/nodeExplanation.generated';

export type ExplainPhase = 'idle' | 'preparing' | 'working' | 'complete' | 'error' | 'stopped';
export interface NodeExplanationAction {
  phase: ExplainPhase;
  detail?: string;
  disabledReason?: string;
  execution?: ModelExplanationState['execution'];
  run(): void;
}
type Attempt = { scope?: string; requestId?: string; phase: ExplainPhase; detail?: string;
  pass?: ModelExplanationState; closed?: boolean };

/** Own a click by its exact request ID, independently of the current selection. */
export function useNodeExplanationAction(nodeId?: string): NodeExplanationAction {
  const context = React.useContext(NodeExplanationContext);
  const model = context?.model, scope = context?.scope;
  const owner = model && 'currentRequestId' in model ? model.currentRequestId ?? null : model?.state?.requestId;
  const [attempts, setAttempts] = React.useState<Record<string, Attempt>>({});
  const latest = React.useRef(attempts);
  const update = React.useCallback((next: Record<string, Attempt>) => { latest.current = next; setAttempts(next); }, []);
  React.useEffect(() => {
    let next = latest.current;
    for (const [id, attempt] of Object.entries(latest.current)) {
      if (attempt.scope !== scope) {
        if (next === latest.current) next = { ...next };
        delete next[id]; continue;
      }
      if (attempt.closed) continue;
      if (owner !== undefined && owner !== attempt.requestId && !model?.disabledReason
        && (attempt.phase === 'preparing' || attempt.phase === 'working')) {
        if (next === latest.current) next = { ...next };
        next[id] = { ...attempt, phase: 'stopped', detail: 'Another explanation replaced this request.', closed: true };
        continue;
      }
      const pass = model?.state && model.state.requestId === attempt.requestId ? model.state : undefined;
      const failure = model?.startFailure && model.startFailure.requestId === attempt.requestId ? model.startFailure : undefined;
      if (!pass && !failure && model?.disabledReason && (attempt.phase === 'preparing' || attempt.phase === 'working')) {
        if (next === latest.current) next = { ...next };
        next[id] = { ...attempt, phase: 'error', detail: 'The explanation was interrupted before completion.', closed: true };
        continue;
      }
      if (!pass && !failure) continue;
      const receipt = context?.verifiedResults?.[id];
      const verified = [receipt && receipt.requestId === attempt.requestId ? receipt.result : undefined,
        context?.states[id], context?.annotations?.[id]].some(result => result?.status === 'ready'
        && result.requestId === attempt.requestId && result.nodeId === id
        && !!result.explanation && result.revision === pass?.revision);
      const phase: ExplainPhase = failure || pass?.status === 'unavailable' ? 'error'
        : pass?.status === 'cancelled' ? 'stopped' : pass?.status === 'ready' && pass.completed > 0 && verified ? 'complete' : 'working';
      const detail = failure?.detail ?? (pass?.status === 'ready' && !verified ? 'Verifying the explanation against the current graph…' : pass?.detail);
      if (phase !== attempt.phase || detail !== attempt.detail || pass !== attempt.pass) {
        if (next === latest.current) next = { ...next };
        next[id] = { ...attempt, phase, detail, pass };
      }
    }
    if (next !== latest.current) update(next);
  }, [scope, owner, model?.state, model?.startFailure, model?.disabledReason, context?.states, context?.annotations, context?.verifiedResults, update]);
  const stored = nodeId ? attempts[nodeId] : undefined;
  const attempt = stored?.scope === scope ? stored : undefined;
  const cached = nodeId && [context?.states[nodeId], context?.annotations?.[nodeId]].some(result => result?.status === 'ready' && !!result.explanation);
  const phase = attempt?.phase ?? (cached ? 'complete' : 'idle');
  const busy = phase === 'preparing' || phase === 'working';
  const disabledReason = !busy ? model?.disabledReason
    || (model?.state?.status === 'loading' && model.state.requestId !== attempt?.requestId ? 'Another explanation is running.' : undefined) : undefined;
  const state = nodeId ? context?.states[nodeId] : undefined;
  const annotation = nodeId ? context?.annotations?.[nodeId] : undefined;
  const visible = state?.status === 'ready' && state.explanation ? state
    : annotation?.status === 'ready' && (!state || state.revision === annotation.revision) ? annotation : undefined;
  const receipt = nodeId ? context?.verifiedResults?.[nodeId] : undefined;
  const execution = receipt && (sameNodeExplanationResult(visible, receipt.result)
    || sameNodeExplanationResult(visible, receipt.adoption)) ? receipt.execution : undefined;
  return { phase, detail: attempt?.detail, disabledReason, execution,
    run: () => {
      if (!nodeId || !model) return;
      const current = latest.current[nodeId];
      if (busy) {
        // A delayed Stop callback belongs to the attempt rendered on that button.
        if (!attempt?.requestId || current?.requestId !== attempt.requestId || current.scope !== scope || current.closed
          || (current.phase !== 'preparing' && current.phase !== 'working')) return;
        update({ ...latest.current, [nodeId]: { ...current, phase: 'stopped', detail: 'Explanation stopped.', closed: true } });
        model.cancel(attempt.requestId); return;
      }
      if (disabledReason) return;
      let requestId: string | void;
      try { requestId = model.start([nodeId]); } catch { requestId = undefined; }
      const next = { ...latest.current }; delete next[nodeId];
      next[nodeId] = { scope, requestId: typeof requestId === 'string' ? requestId : undefined,
        phase: typeof requestId === 'string' ? 'preparing' : 'error',
        ...(typeof requestId === 'string' ? {} : { detail: 'The explanation could not be started. Check the current graph and agent connection.' }) };
      while (Object.keys(next).length > 64) delete next[Object.keys(next)[0]];
      update(next);
    } };
}

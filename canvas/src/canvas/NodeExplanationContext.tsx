import React from 'react';
import type { ModelExplanationState, NodeExplanationState } from '../ide/nodeExplanation.generated';
import type { CanvasAgentAvailability, CanvasAgentSettings } from '../ide/protocol.generated';

/** Local verified pass provenance; never substitutes for selection/citation authority. */
export interface NodeExplanationResultReceipt {
  requestId: string;
  result: NodeExplanationState;
  adoption?: NodeExplanationState;
  execution?: ModelExplanationState['execution'];
}

export function sameNodeExplanationResult(left?: NodeExplanationState, right?: NodeExplanationState): boolean {
  return !!left && !!right && left.status === 'ready' && right.status === 'ready'
    && !!left.explanation && !!right.explanation && left.nodeId === right.nodeId
    && left.revision === right.revision && left.requestId === right.requestId
    && JSON.stringify(left.explanation) === JSON.stringify(right.explanation);
}

/** Host-confirmed, revision-scoped interpretations. Never stored in node data. */
export interface NodeExplanationContextValue {
  /** Current local graph identity for transient action feedback, never wire data. */
  scope?: string;
  states: Readonly<Record<string, NodeExplanationState>>;
  /** Retained ready annotations survive transient selection/documentation reads. */
  annotations?: Readonly<Record<string, NodeExplanationState>>;
  verifiedResults?: Readonly<Record<string, NodeExplanationResultReceipt>>;
  /** Explicit model enrichment changes presentation only, never graph evidence. */
  annotationsVisible?: boolean;
  /** Independent node attempts; the view controller below is a separate entry. */
  nodeJobs?: {
    currentRequestIds: Readonly<Record<string, string>>;
    requests: Readonly<Record<string, {
      cancelled?: boolean;
      /** Verified completion for this job, independent of newer visible output. */
      result?: NodeExplanationState;
      state?: ModelExplanationState;
      startFailure?: { requestId: string; detail: string };
    }>>;
    start(nodeId: string): string | void;
    cancel(requestId: string): void;
  };
  model?: {
    /** Owns the explicit attempt, including asynchronous revision preflight. */
    currentRequestId?: string;
    state?: ModelExplanationState;
    startFailure?: { requestId: string; detail: string };
    disabledReason?: string;
    configuration?: {
      availability?: CanvasAgentAvailability;
      settings: CanvasAgentSettings;
      followsChat: boolean;
      disabled?: boolean;
      choose(settings: CanvasAgentSettings | undefined): void;
      discover(refresh?: boolean): void;
    };
    start(nodeIds: string[]): string | void;
    cancel(requestId?: string): void;
    toggle(): void;
  };
  retry(state: NodeExplanationState): void;
  reveal(state: NodeExplanationState, citationId: string): void;
}

export const NodeExplanationContext = React.createContext<NodeExplanationContextValue | null>(null);

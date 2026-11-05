import React from 'react';
import type { ModelExplanationState, NodeExplanationState } from '../ide/nodeExplanation.generated';

/** Host-confirmed, revision-scoped interpretations. Never stored in node data. */
export interface NodeExplanationContextValue {
  states: Readonly<Record<string, NodeExplanationState>>;
  /** Retained ready annotations survive transient selection/documentation reads. */
  annotations?: Readonly<Record<string, NodeExplanationState>>;
  /** Explicit model enrichment changes presentation only, never graph evidence. */
  annotationsVisible?: boolean;
  model?: {
    state?: ModelExplanationState;
    disabledReason?: string;
    start(nodeIds: string[]): void;
    cancel(): void;
    toggle(): void;
  };
  retry(state: NodeExplanationState): void;
  reveal(state: NodeExplanationState, citationId: string): void;
}

export const NodeExplanationContext = React.createContext<NodeExplanationContextValue | null>(null);

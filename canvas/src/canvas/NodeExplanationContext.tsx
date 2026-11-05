import React from 'react';
import type { NodeExplanationState } from '../ide/nodeExplanation.generated';

/** Host-confirmed, revision-scoped interpretations. Never stored in node data. */
export interface NodeExplanationContextValue {
  states: Readonly<Record<string, NodeExplanationState>>;
  retry(state: NodeExplanationState): void;
  reveal(state: NodeExplanationState, citationId: string): void;
}

export const NodeExplanationContext = React.createContext<NodeExplanationContextValue | null>(null);

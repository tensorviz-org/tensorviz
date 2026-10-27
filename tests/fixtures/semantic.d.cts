import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { ResolvedExecutionPlan, CapturePreparation } from '@tensorviz/graph-host/executionPlanCore';
import type { AppliedSourceObservation } from '@tensorviz/contracts/protocol';
export function appliedFixture(): AppliedSourceObservation;
export function canonicalFixture(attemptId?: string): {
  capture: CaptureSnapshot & { planDigest: string; preparationDigest: null; automatic: null };
  plan: ResolvedExecutionPlan;
};
export function automaticFixture(attemptId?: string): {
  capture: CaptureSnapshot & { planDigest: null; preparationDigest: string; automatic: NonNullable<CaptureSnapshot['automatic']> };
  preparation: CapturePreparation;
};

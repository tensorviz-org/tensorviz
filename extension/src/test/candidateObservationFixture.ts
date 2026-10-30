/** Synthetic history/transport fixture. Real numerics are checked by the CPU integration suite. */
import type { CandidateEvidence } from '@tensorviz/graph-host/candidateExecution';
import type { CandidateExecution } from '../executionPlan';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

export function candidateObservationFixture(id = 'candidate:observations'): CandidateEvidence {
  const { capture, plan } = canonicalFixture(id);
  const digest = 'a'.repeat(64);
  return { execution: { id: plan.id, plan } as CandidateExecution, capture, baseCurrent: true, candidateCurrent: true,
    observations: { version: 1, plan: structuredClone(plan), planDigest: capture.planDigest!, status: 'passed',
      setup: { version: 1, resolvedPlan: structuredClone(plan), planDigest: capture.planDigest,
        sourceDigest: plan.source.files.find(file => file.path === plan.source.modelFile)!.digest,
        inputs: { digest: capture.inputDigest }, state: { digest: capture.stateDigest } },
      execution: { version: 1, outputs: { digest, containers: [], truncated: [], leaves: [0, 1, 2].map(i => ({
        path: `$[${i}]`, kind: 'tensor', shape: [1], dtype: 'float32', numel: 1, digest, nonfinite: 0,
        values: { encoding: 'base64', byteOrder: 'little', data: 'AAAAQA==' } })) } } } };
}

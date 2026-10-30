import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';
import type { CandidateEvidence } from '@tensorviz/graph-host/candidateExecution';
import type { CandidateExecution } from '@tensorviz/graph-host/executionPlanCore';
import type { CheckpointEvidence } from '@tensorviz/graph-host/checkpointEvidence';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';

/** Failed load has no capture, but retains its exact candidate source/resource. */
export function candidateCheckpointFixture(): CandidateEvidence & { checkpoint: CheckpointEvidence } {
  const { plan } = canonicalFixture('candidate:checkpoint');
  plan.execution = executionSpecSchema.parse({ inputs: { args: [{ shape: [1, 3], dtype: 'float32', fill: 'ones' }] },
    checkpoint: { file: 'weights.pt', strict: true } });
  const source = plan.source.files.find(item => item.path === plan.source.modelFile)!;
  plan.source.files.push({ path: 'weights.pt', digest: 'b'.repeat(64), role: 'native', documentVersion: null });
  const registered = { kind: 'parameter' as const, shape: [2, 3], dtype: 'torch.float32',
    objectGroup: 0, storageGroup: 0, storageOffset: 0, stride: [3, 1], unavailable: null };
  return { execution: { id: plan.id, plan } as CandidateExecution, baseCurrent: true, candidateCurrent: false,
    checkpoint: { version: 1, sourceDigest: source.digest,
      checkpoint: { file: 'weights.pt', digest: 'b'.repeat(64), strict: true },
      scope: 'registered_state_before_load', loaderSemantics: 'not_inferred', loadStatus: 'failed',
      complete: true, registeredComplete: true, storedComplete: true, omissions: [],
      entries: ['left.weight', 'right.weight'].map(name => ({ name, registered: { ...registered },
        stored: { ...registered, kind: 'stored', shape: [2, 4], stride: [4, 1] },
        presence: 'both', shape: 'different', dtype: 'same' })) } };
}

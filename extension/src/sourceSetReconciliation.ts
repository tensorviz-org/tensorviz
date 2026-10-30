/** Compare accepted canonical captures after all reviewed source files were applied.
 * No recipe postcondition, filesystem access or model execution is implied. */
import { isDeepStrictEqual } from 'node:util';
import { compareCaptures } from '@tensorviz/contracts/captureComparison';
import type { AppliedSourceFile, AppliedSourceSetComparison } from '@tensorviz/contracts/protocol';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';

export interface AppliedSourceSetEvidence { version: 2; base: CaptureSnapshot; files: AppliedSourceFile[] }

function reason(input: AppliedSourceSetEvidence, current: CaptureSnapshot): AppliedSourceSetComparison['reason'] {
  const before = input.base.source, after = current.source;
  if (before.workspaceSessionId !== after.workspaceSessionId || before.workspaceId !== after.workspaceId
    || before.modelId !== after.modelId || before.modelFile !== after.modelFile || before.className !== after.className) return 'different_model';
  if (input.base.id === current.id || input.base.attemptId === current.attemptId) return 'capture_not_new';
  if (before.sourceMode !== 'saved' || after.sourceMode !== 'saved') return 'source_unavailable';
  const reviewed = new Map(input.files.map(file => [file.file, file]));
  const actual = input.files.map(file => after.files.find(item => item.path === file.file));
  if (actual.some(file => !file || file.role !== 'source')) return 'source_unavailable';
  if (!actual.every((file, index) => file!.digest === input.files[index].afterDigest)) {
    return actual.every((file, index) => file!.digest === input.files[index].beforeDigest) ? 'source_reverted' : 'source_changed';
  }
  const files = (source: CaptureSnapshot['source'], expected: boolean) => source.files.map(file =>
    [file.path, file.role, expected ? reviewed.get(file.path)?.afterDigest ?? file.digest : file.digest])
    .sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
  if (!isDeepStrictEqual(files(before, true), files(after, false)) || before.configDigest !== after.configDigest) return 'dependencies_changed';
  if (before.environmentDigest !== after.environmentDigest) return 'environment_changed';
  return 'reviewed_source';
}

export function reconcileAppliedSourceSet(input: AppliedSourceSetEvidence, current: CaptureSnapshot): AppliedSourceSetComparison {
  const result: AppliedSourceSetComparison = { baseCaptureId: input.base.id, captureId: current.id, sourceId: current.source.id,
    dependencyScope: input.base.source.dependencyScope === 'complete' && current.source.dependencyScope === 'complete' ? 'complete' : 'partial',
    reason: reason(input, current), inputs: 'unavailable', state: 'unavailable', semanticPostconditions: 'not_checked', comparison: null };
  if (result.reason !== 'reviewed_source') return result;
  const identity = (a: string | null, b: string | null) => a === null || b === null ? 'unavailable' : a === b ? 'same' : 'different';
  result.inputs = identity(input.base.inputDigest, current.inputDigest);
  result.state = identity(input.base.stateDigest, current.stateDigest);
  const diff = compareCaptures(input.base, current), fields = (rows: typeof diff.changes) => rows.reduce((sum, row) => sum + row.fields.length, 0);
  result.comparison = { equivalent: diff.equivalent, coverageComplete: diff.coverageComplete,
    changedFieldCount: fields(diff.changes), observedFieldCount: fields(diff.observations), unresolvedFieldCount: fields(diff.unresolved),
    unmatchedRecordCount: Object.values(diff.correspondence.unmatched).reduce((sum, kinds) =>
      sum + Object.values(kinds).reduce((count, ids) => count + ids.length, 0), 0),
    uncertainMatchCount: diff.correspondence.uncertain.length, connectionChangeCount: diff.valueUses.changes.length,
    unresolvedConnectionCount: diff.valueUses.unresolved.length };
  return result;
}

/** Host-only applicability of a resolved saved invocation. No browser-supplied paths. */
import { isDeepStrictEqual } from 'node:util';
import type { RunSpec } from '@tensorviz/contracts/protocol';
import type { ValidationPreparation } from './validationController';

export interface ExecutionGuard {
  readonly spec: RunSpec;
  /** Re-read saved bytes, configuration and the selected environment without imports. */
  current(signal: AbortSignal): Promise<boolean>;
}

/** Request IDs change on each preflight; executable applicability must not. Opening
 * or releasing an unchanged saved file in VS Code is harmless. Revisions of
 * loaded documents, physical file identity and saved bytes remain checked. */
export function samePreparedExecution(before: ValidationPreparation, after: ValidationPreparation): boolean {
  if (before.resolution.status !== 'resolved' || after.resolution.status !== 'resolved') return false;
  const a = before.resolution.spec, b = after.resolution.spec;
  const identity = ({ runId: _run, generation: _generation, ...applicability }: RunSpec['identity']) => applicability;
  const files = (spec: RunSpec) => spec.files.map(({ documentVersion: _version, ...file }) => file);
  if (!isDeepStrictEqual(identity(a.identity), identity(b.identity))
    || !isDeepStrictEqual(a.profile, b.profile) || !isDeepStrictEqual(a.interpreter, b.interpreter)
    || a.dependencyScope !== b.dependencyScope || !isDeepStrictEqual(files(a), files(b))) return false;
  return sameExecutionGuards(before.guards, after.guards);
}

/** Shared physical-file and observed-buffer guard for profiles and candidate replay. */
export function sameExecutionGuards(before: Record<string, unknown>, after: Record<string, unknown>): boolean {
  const entries = Object.entries(before);
  if (entries.length !== Object.keys(after).length) return false;
  return entries.every(([file, guard]) => {
    const next = after[file];
    if (!Array.isArray(guard) || !Array.isArray(next)) return false;
    // Directories carry canonical path + physical id; files add digest/version/dirty.
    if (guard.length === 2) return isDeepStrictEqual(guard, next);
    return guard.length === 5 && next.length === 5
      && isDeepStrictEqual(guard.slice(0, 3), next.slice(0, 3))
      && (guard[3] == null || next[3] == null || guard[3] === next[3]) && !guard[4] && !next[4];
  });
}

export function savedExecutionGuard(prepared: ValidationPreparation,
  refresh: (signal: AbortSignal) => Promise<ValidationPreparation>): ExecutionGuard {
  if (prepared.resolution.status !== 'resolved') throw new Error('A resolved execution is required.');
  const baseline = structuredClone(prepared);
  return {
    spec: structuredClone(prepared.resolution.spec),
    current: async signal => {
      if (signal.aborted) return false;
      try {
        const now = await refresh(signal);
        return !signal.aborted && samePreparedExecution(baseline, now);
      } catch { return false; }
    },
  };
}

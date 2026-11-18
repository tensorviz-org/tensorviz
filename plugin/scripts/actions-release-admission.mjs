// GitHub's signed artifact provenance replaces reviewer-managed signing keys.
// Execution evidence is read live from the pinned trusted workflow, never inferred
// from caller-supplied success booleans. GUI acceptance is outside this scope.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {contentHash} from './source-release-admission.mjs';
import {canonicalReleaseMigration, verifyReleaseMigration} from './release-migration.mjs';

export const unifiedWorkflow = '.github/workflows/release-unified.yml';
export const requiredReleaseJobs = Object.freeze(['configuration', 'renderer', 'extension-and-docs', 'assemble', 'bundle',
  ...['darwin-arm64', 'darwin-x64', 'linux-arm64', 'linux-x64', 'win32-arm64', 'win32-x64'].map(p => `native-${p}`),
  'migration-darwin-arm64', 'migration-linux-x64']);

export function validateActionsReleaseRun(manifest, admission, policy, run, jobs) {
  const migration = canonicalReleaseMigration(policy.version);
  if (policy.schemaVersion !== 1 || policy.enabled !== true || policy.repository !== 'tensorviz-org/tensorviz'
      || policy.evidence?.method !== 'github-actions' || policy.evidence.workflow !== unifiedWorkflow
      || policy.version !== manifest.version || manifest.repository !== policy.repository
      || policy.previousVersion !== migration.from
      || policy.previousCatalogCommit !== migration.previousExport) throw Error('Reviewed Actions release policy required');
  if (admission?.schemaVersion !== 2 || admission.method !== 'github-actions' || !Number.isSafeInteger(admission.runId)
      || admission.runId < 1 || !Number.isSafeInteger(admission.runAttempt) || admission.runAttempt < 1
      || manifest.build?.runId !== admission.runId || manifest.build?.runAttempt !== admission.runAttempt
      || manifest.build?.workflow !== unifiedWorkflow) throw Error('Actions evidence identity mismatch');
  if (run.id !== admission.runId || run.run_attempt !== admission.runAttempt || run.head_sha !== manifest.sourceRevision
      || run.repository?.full_name !== policy.repository || run.head_repository?.full_name !== policy.repository
      || run.path !== unifiedWorkflow || run.event !== 'workflow_dispatch' || run.head_branch !== 'main'
      || run.status !== 'completed' || run.conclusion !== 'success') throw Error('Candidate was not produced by a successful trusted main workflow');
  for (const name of requiredReleaseJobs) {
    const matching = jobs.filter(job => job.name === name);
    const attempts = new Set();
    for (const job of matching) {
      if (job.run_id !== run.id || job.head_sha !== run.head_sha || !Number.isSafeInteger(job.run_attempt)
          || job.run_attempt < 1 || job.run_attempt > run.run_attempt || attempts.has(job.run_attempt))
        throw Error(`Required release job provenance differs: ${name}`);
      attempts.add(job.run_attempt);
    }
    // GitHub reruns only selected jobs and their dependants. Successful jobs
    // inherited from this same immutable run remain required evidence.
    const latest = matching.reduce((selected, job) => !selected || job.run_attempt > selected.run_attempt ? job : selected, null);
    if (!latest || latest.status !== 'completed' || latest.conclusion !== 'success'
        || (name === 'bundle' && latest.run_attempt !== run.run_attempt))
      throw Error(`Required release job did not succeed: ${name}`);
  }
}

export async function verifyActionsReleaseAdmission(manifest, manifestBytes, admission, policy, release, {command = execFileSync} = {}) {
  const api = endpoint => JSON.parse(command('gh', ['api', '--hostname', 'github.com', `repos/tensorviz-org/tensorviz/${endpoint}`], {encoding: 'utf8', maxBuffer: 4 * 1024 * 1024}));
  if (!Number.isSafeInteger(admission?.runId) || admission.runId < 1 || !Number.isSafeInteger(admission.runAttempt) || admission.runAttempt < 1) throw Error('Invalid candidate run identity');
  const run = api(`actions/runs/${admission.runId}`);
  const jobs = api(`actions/runs/${admission.runId}/jobs?filter=all&per_page=100`);
  if (jobs.total_count > 100) throw Error('Unexpected release job inventory');
  validateActionsReleaseRun(manifest, admission, policy, run, jobs.jobs);
  if (release.version !== manifest.version || release.sourceRevision !== manifest.sourceRevision) throw Error('Migration release identity mismatch');
  verifyReleaseMigration({previous: {version: policy.previousVersion}, initialMain: policy.previousCatalogCommit,
    release, receipts: admission.migrationReceipts});
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-provenance-'));
  try {
    const file = path.join(temporary, 'unified-release.json');
    await fs.writeFile(file, manifestBytes, {flag: 'wx', mode: 0o600});
    const proofs = JSON.parse(command('gh', ['attestation', 'verify', file, '--repo', policy.repository,
      '--signer-workflow', `${policy.repository}/${unifiedWorkflow}`, '--signer-digest', manifest.sourceRevision,
      '--source-ref', 'refs/heads/main', '--source-digest', manifest.sourceRevision, '--deny-self-hosted-runners', '--format', 'json'],
    {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024}));
    const invocation = `https://github.com/${policy.repository}/actions/runs/${admission.runId}/attempts/${admission.runAttempt}`;
    if (!Array.isArray(proofs) || !proofs.some(proof => {
      const statement = proof.verificationResult?.statement;
      return statement?.predicate?.runDetails?.metadata?.invocationId === invocation
        && statement.subject?.some(subject => subject.digest?.sha256 === contentHash(manifestBytes));
    })) throw Error('Signed provenance does not identify this exact candidate run and manifest');
    return {method: 'github-actions', runId: run.id, runAttempt: run.run_attempt, jobs: requiredReleaseJobs.length,
      scope: 'six native targets; Codex/Claude upgrade and rollback on darwin-arm64 and linux-x64; packaged extension and docs'};
  } finally { await fs.rm(temporary, {recursive: true, force: true}); }
}

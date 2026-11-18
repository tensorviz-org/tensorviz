// Trusted-main preflight; this step has no publisher credentials and changes no refs.
import {execFileSync} from 'node:child_process';
import {appendFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {unifiedVersionFiles, validateUnifiedSourceVersions} from './unified-source-versions.mjs';
import {unifiedWorkflow} from './actions-release-admission.mjs';
import {canonicalReleaseMigration} from './release-migration.mjs';

export function unifiedReleasePreflight({source, head, publish, candidateRunId, repository, ref, event}, {command = execFileSync} = {}) {
  if (repository !== 'tensorviz-org/tensorviz' || ref !== 'refs/heads/main' || event !== 'workflow_dispatch'
      || !/^[a-f0-9]{40}$/.test(head ?? '') || typeof publish !== 'boolean') throw Error('Manual trusted main release required');
  source = publish ? source : head;
  if (!/^[a-f0-9]{40}$/.test(source ?? '') || (publish && !/^[1-9]\d*$/.test(candidateRunId ?? ''))) throw Error('Exact candidate source and run required');
  const git = args => command('git', args, {encoding: 'utf8', maxBuffer: 4 * 1024 * 1024}).trim();
  command('git', ['merge-base', '--is-ancestor', source, head], {stdio: 'pipe'});
  const policy = JSON.parse(git(['show', `${source}:plugin/release/public-launch-policy.json`]));
  const migration = canonicalReleaseMigration(policy.version);
  validateUnifiedSourceVersions(Object.fromEntries(unifiedVersionFiles.map(file => [file, JSON.parse(git(['show', `${source}:${file}`]))])), policy.version);
  if (policy.schemaVersion !== 1 || policy.repository !== repository || policy.previousVersion !== migration.from
      || policy.previousCatalogCommit !== migration.previousExport) throw Error('Canonical release policy mismatch');
  if (publish) {
    if (!policy.enabled || !policy.channels?.enabled || policy.evidence?.method !== 'github-actions'
        || policy.evidence.workflow !== unifiedWorkflow) throw Error('Reviewed automated publication policy not enabled');
    const run = JSON.parse(command('gh', ['api', '--hostname', 'github.com', `repos/${repository}/actions/runs/${candidateRunId}`], {encoding: 'utf8'}));
    if (run.head_sha !== source || run.path !== unifiedWorkflow || run.head_branch !== 'main'
        || run.repository?.full_name !== repository || run.head_repository?.full_name !== repository
        || run.event !== 'workflow_dispatch' || run.status !== 'completed' || run.conclusion !== 'success') throw Error('Trusted successful candidate run required');
  }
  return {source, version: policy.version, publish, candidateRunId: publish ? candidateRunId : ''};
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = unifiedReleasePreflight({source: process.env.RELEASE_SOURCE, head: process.env.GITHUB_SHA,
    publish: process.env.RELEASE_PUBLISH === 'true', candidateRunId: process.env.CANDIDATE_RUN_ID,
    repository: process.env.GITHUB_REPOSITORY, ref: process.env.GITHUB_REF, event: process.env.GITHUB_EVENT_NAME});
  appendFileSync(process.env.GITHUB_OUTPUT, `source=${result.source}\nversion=${result.version}\npublish=${result.publish}\ncandidate_run_id=${result.candidateRunId}\n`);
}

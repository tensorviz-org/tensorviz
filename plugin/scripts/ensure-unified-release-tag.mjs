// This runs only after the full candidate bundle succeeds. Never move an existing ref.
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

function githubApi(method, endpoint, fields) {
  const args = ['api', '--hostname', 'github.com', '--method', method, endpoint];
  for (const [key, value] of Object.entries(fields ?? {})) args.push('--raw-field', `${key}=${value}`);
  const result = spawnSync('gh', args, {encoding: 'utf8', maxBuffer: 1024 * 1024});
  if (result.error) throw Error('Release tag API could not be started');
  if (result.status !== 0) {
    let body;
    try { body = JSON.parse(result.stdout); } catch {}
    if (method === 'GET' && /\(HTTP 404\)\s*$/.test(result.stderr ?? '') && body?.message === 'Not Found') return null;
    throw Error('Release tag API failed; no ref may be overwritten');
  }
  try { return JSON.parse(result.stdout); } catch { throw Error('Unexpected release tag API response'); }
}

export function ensureUnifiedReleaseTag({repository, version, source}, api = githubApi) {
  if (repository !== 'tensorviz-org/tensorviz' || !/^[1-9]\d*\.\d+\.\d+$/.test(version ?? '') || !/^[a-f0-9]{40}$/.test(source ?? '')) throw Error('Exact canonical release tag identity required');
  const ref = `refs/tags/v${version}`;
  const valid = value => value?.ref === ref && value?.object?.type === 'commit' && value.object.sha === source;
  const existing = api('GET', `repos/${repository}/git/ref/tags/v${version}`);
  if (existing !== null) {
    if (!valid(existing)) throw Error('Existing release tag must be a lightweight tag at the exact candidate commit');
    return {ref, source, created: false};
  }
  const created = api('POST', `repos/${repository}/git/refs`, {ref, sha: source});
  if (!valid(created)) throw Error('Unexpected created release tag response');
  return {ref, source, created: true};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(ensureUnifiedReleaseTag({repository: process.env.GITHUB_REPOSITORY,
    version: process.env.TENSORVIZ_RELEASE_VERSION, source: process.env.RELEASE_SOURCE})));
}

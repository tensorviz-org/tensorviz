import * as vscode from 'vscode';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { ReviewSession, ReviewSnapshot } from './reviewSnapshots';
import { SOURCE_SET_LIMITS } from './workspaceSources';

export interface PullRequestAddress { owner: string; repo: string; number: number }
interface PullRequest {
  number: number; title: string; html_url: string;
  base: { sha: string; repo: { full_name: string } };
  head: { sha: string; repo: { full_name: string } | null };
}
export function parsePullRequestUrl(value: string): PullRequestAddress {
  let url: URL;
  try { url = new URL(value.trim()); } catch { throw new Error('Enter a GitHub pull request URL.'); }
  const match = /^\/([\w.-]+)\/([\w.-]+)\/pull\/([1-9]\d*)(?:\/(?:files|commits|changes))?\/?$/.exec(url.pathname);
  if (url.protocol !== 'https:' || url.hostname !== 'github.com' || url.port || url.username || url.password || !match) throw new Error('Enter a https://github.com/owner/repository/pull/number URL.');
  return { owner: match[1], repo: match[2], number: Number(match[3]) };
}
export function githubRepository(remote: string): string | undefined {
  const match = /^(?:https:\/\/github\.com\/|git@github\.com:)([\w.-]+\/[\w.-]+?)(?:\.git)?\/?$/.exec(remote.trim());
  return match?.[1];
}
export type GitHubRead = <T>(route: string) => Promise<T>;

/** Credentials are requested through VS Code and never enter URLs, logs or disk. */
export function githubReader(): GitHubRead {
  let token: string | undefined;
  return async <T>(route: string): Promise<T> => {
    if (!route.startsWith('/repos/')) throw new Error('Invalid GitHub review resource.');
    const read = () => fetch(`https://api.github.com${route}`, { headers: { Accept: 'application/vnd.github+json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}) }, redirect: 'error', signal: AbortSignal.timeout(30_000) });
    let response = await read();
    if ((response.status === 401 || response.status === 404 || response.status === 403) && !token) {
      const existing = await vscode.authentication.getSession('github', ['repo'], { silent: true });
      token = existing?.accessToken;
      if (!token) {
        try { token = (await promisify(execFile)('gh', ['auth', 'token', '--hostname', 'github.com'], { timeout: 5000, maxBuffer: 16_384 })).stdout.trim() || undefined; }
        catch { /* GitHub CLI is optional. Its output is never logged or surfaced. */ }
      }
      if (!token) token = (await vscode.authentication.getSession('github', ['repo'], { createIfNone: { detail: 'Read the pull request and its source for graph review.' } })).accessToken;
      response = await read();
    }
    if (!response.ok) throw new Error(response.status === 404 ? 'Pull request source was not found or your GitHub account cannot access it.'
      : response.status === 403 || response.status === 429 ? 'GitHub refused the request or its rate limit was reached. Try again after access or rate limits are resolved.'
      : `GitHub review request failed (${response.status}).`);
    return await response.json() as T;
  };
}

const validSha = (sha: string) => /^[a-f0-9]{40}$/.test(sha);
const safeRepo = (repo: string) => /^[\w.-]+\/[\w.-]+$/.test(repo);
export async function githubSnapshot(repository: string, commit: string, read: GitHubRead, cache: string): Promise<ReviewSnapshot> {
  if (!safeRepo(repository) || !validSha(commit)) throw new Error('GitHub returned an invalid revision identity.');
  const tree = await read<{ sha: string; tree: { path: string; type: string; mode: string; sha: string; size?: number }[]; truncated: boolean }>(`/repos/${repository}/git/trees/${commit}?recursive=1`);
  const files = new Map<string, string>(), large = new Set<string>();
  for (const item of tree.tree) {
    if (item.type !== 'blob' || item.mode === '120000' || !validSha(item.sha)) continue;
    if (item.path.startsWith('/') || item.path.split('/').some(part => ['..', '.', ''].includes(part)) || item.path.includes('\\')) continue;
    files.set(item.path, item.sha);
    if ((item.size ?? 0) > SOURCE_SET_LIMITS.maxSourceBytes) large.add(item.path);
  }
  // Source cache is partitioned by repository; reads still require a successful
  // authenticated tree request, including when all blobs are already cached.
  const directory = path.join(cache, createHash('sha256').update(repository).digest('hex'));
  const bodies = new Map<string, Promise<string>>();
  return { label: commit.slice(0, 7), commit, files, truncated: tree.truncated, unavailable: large,
    read: file => {
      const sha = files.get(file);
      if (!sha || large.has(file)) return Promise.reject(new Error(`Source is unavailable or exceeds the review size limit: ${file}`));
      let body = bodies.get(sha);
      if (!body) {
        body = (async () => {
          const cacheFile = path.join(directory, sha);
          let bytes: Buffer | undefined;
          try { bytes = await fs.readFile(cacheFile); } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
          const digest = (value: Buffer) => createHash('sha1').update(`blob ${value.length}\0`).update(value).digest('hex');
          if (!bytes || digest(bytes) !== sha) {
            const blob = await read<{ sha: string; content: string; encoding: string; size: number }>(`/repos/${repository}/git/blobs/${sha}`);
            if (blob.sha !== sha || blob.encoding !== 'base64' || blob.size > SOURCE_SET_LIMITS.maxSourceBytes) throw new Error('GitHub returned an invalid source blob.');
            bytes = Buffer.from(blob.content, 'base64');
            if (bytes.length !== blob.size || digest(bytes) !== sha) throw new Error('GitHub source digest did not match the pinned revision.');
            await fs.mkdir(directory, { recursive: true, mode: 0o700 });
            await fs.writeFile(cacheFile, bytes, { mode: 0o600 });
          }
          return bytes.toString('utf8');
        })();
        bodies.set(sha, body);
      }
      return body;
    } };
}

export async function loadPullRequest(address: PullRequestAddress, read: GitHubRead, cache: string,
  revisions?: { before: string; after: string }): Promise<ReviewSession> {
  const repository = `${address.owner}/${address.repo}`;
  const route = `/repos/${repository}/pulls/${address.number}`;
  const pr = await read<PullRequest>(route);
  if (pr.number !== address.number || pr.base.repo.full_name.toLowerCase() !== repository.toLowerCase()
    || !validSha(pr.base.sha) || !validSha(pr.head.sha)) throw new Error('GitHub returned a different pull request or invalid revisions.');
  const comparison = revisions ? { merge_base_commit: { sha: revisions.before } }
    : await read<{ merge_base_commit: { sha: string } }>(`/repos/${repository}/compare/${pr.base.sha}...${pr.head.sha}`);
  const head = revisions?.after ?? pr.head.sha;
  const [before, after] = await Promise.all([
    githubSnapshot(repository, comparison.merge_base_commit.sha, read, cache),
    githubSnapshot(pr.head.repo?.full_name ?? repository, head, read, cache),
  ]);
  return { title: `${repository} #${pr.number} · ${pr.title}`, before, after,
    refresh: () => loadPullRequest(address, read, cache),
    updated: async () => { const current = await read<PullRequest>(route); return current.head.sha !== head || current.base.sha !== pr.base.sha; },
    locate: uri => {
      if (uri.scheme !== 'pr') return undefined;
      try {
        const value = JSON.parse(uri.query);
        if (value.prNumber !== pr.number || value.headCommit !== head || value.baseCommit !== before.commit) return undefined;
        const file = value.isBase && value.previousFileName ? value.previousFileName : value.fileName;
        const side = value.isBase ? 'before' : 'after';
        return typeof file === 'string' && (side === 'before' ? before : after).files.has(file) ? { file, side } : undefined;
      } catch { return undefined; }
    } };
}

export function choosePullRequest(repository: string | undefined, read: GitHubRead): Promise<PullRequestAddress | undefined> {
  return new Promise(resolve => {
    const picker = vscode.window.createQuickPick<vscode.QuickPickItem & { url: string }>();
    picker.title = 'Open PR in tensorViz';
    picker.placeholder = 'Paste a GitHub PR URL or select a repository pull request';
    picker.matchOnDescription = true;
    let items: (vscode.QuickPickItem & { url: string })[] = [], finished = false;
    const render = () => {
      let pasted: typeof items = [];
      try { const address = parsePullRequestUrl(picker.value); pasted = [{ label: `Open ${address.owner}/${address.repo} #${address.number}`, url: picker.value, alwaysShow: true }]; } catch { /* The picker also filters repository titles. */ }
      picker.items = [...pasted, ...items];
    };
    const finish = (address?: PullRequestAddress) => {
      if (finished) return;
      finished = true; subscriptions.forEach(item => item.dispose()); picker.dispose(); resolve(address);
    };
    const subscriptions = [picker.onDidHide(() => finish()), picker.onDidChangeValue(render), picker.onDidAccept(() => {
      const choice = picker.selectedItems[0];
      try { finish(parsePullRequestUrl(choice?.url ?? picker.value)); }
      catch { picker.title = 'Enter a complete GitHub PR URL or select a pull request'; }
    })];
    picker.show();
    if (repository) {
      picker.busy = true;
      void read<PullRequest[]>(`/repos/${repository}/pulls?state=open&sort=updated&per_page=100`).then(prs => {
        if (finished) return;
        items = prs.map(pr => ({ label: `#${pr.number} ${pr.title}`, description: repository, url: pr.html_url })); render();
      }, () => { if (!finished) picker.title = 'Repository PRs unavailable — paste a GitHub PR URL'; })
        .finally(() => { if (!finished) picker.busy = false; });
    }
  });
}

// Durable write-ahead records live in a separate, permanently retained draft.
// No tokens, logs, source refs, or release assets are changed by this journal.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {contentHash} from './source-release-admission.mjs';
import {githubTransport} from './publish-source-release.mjs';

export function githubPublicationJournal(identity, {transport} = {}) {
  if (!/^[1-9]\d*\.\d+\.\d+$/.test(identity.version ?? '') || !/^[a-f0-9]{40}$/.test(identity.sourceRevision ?? '')) throw Error('Exact journal identity required');
  const tag = `publication-journal-v${identity.version}-${identity.sourceRevision}`;
  return {
    async intent(channel, submission) {
      if (!['vscode-marketplace', 'open-vsx', 'docs'].includes(channel)
          || submission.version !== identity.version || submission.sourceRevision !== identity.sourceRevision
          || !/^[a-f0-9]{64}$/.test(submission.vsixSha256 ?? '')) throw Error('Journal submission identity mismatch');
      const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-journal-'));
      try {
        const api = transport ?? githubTransport(temporary);
        const repo = await api.repository();
        if (repo.full_name !== 'tensorviz-org/tensorviz' || repo.private || repo.archived) throw Error('Canonical journal repository required');
        let release = await api.release(tag);
        if (!release) release = await api.create({tag_name: tag, target_commitish: identity.sourceRevision,
          draft: true, prerelease: true, name: `Publication journal ${identity.version}`,
          body: 'Durable publication intents. Retain this draft and its assets for retries; never publish, delete or reset it.'});
        if (!release.draft || release.tag_name !== tag || release.target_commitish !== identity.sourceRevision) throw Error('Journal draft provenance mismatch');
        const name = `${channel}.json`;
        const assets = await api.assets(release.id);
        if (assets.some(asset => asset.name === name)) throw Error('Prior submission is pending or uncertain; inspect registry instead of re-uploading');
        const bytes = Buffer.from(JSON.stringify({schemaVersion: 1, channel, version: identity.version,
          sourceRevision: identity.sourceRevision, vsixSha256: submission.vsixSha256, state: 'submission-started'}) + '\n');
        await fs.writeFile(path.join(temporary, name), bytes, {flag: 'wx', mode: 0o600});
        await api.upload(tag, name, bytes);
        const hosted = (await api.assets(release.id)).filter(asset => asset.name === name);
        if (hosted.length !== 1 || contentHash(await api.download(hosted[0].id)) !== contentHash(bytes)) throw Error('Durable journal readback failed; submission blocked');
      } finally { await fs.rm(temporary, {recursive: true, force: true}); }
    },
  };
}

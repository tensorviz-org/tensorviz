import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { validateWindowsPE } from './windows-bootstrap.mjs';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
export async function verifyBootstrap(archive, { arch, version, sourceRevision }) {
  const base = path.join(path.dirname(archive), `tensorviz-bootstrap-${arch}`);
  const binary = `${base}.exe`, notices = `${base}.NOTICES.md`;
  for (const file of [binary, notices, `${binary}.json`]) {
    const stat = await fs.lstat(file);
    if (!stat.isFile() || stat.size > 128 * 1024 * 1024) throw new Error('Unsafe bootstrap input');
  }
  const receipt = JSON.parse(await fs.readFile(`${binary}.json`, 'utf8'));
  const bytes = await fs.readFile(binary), noticeBytes = await fs.readFile(notices);
  validateWindowsPE(bytes, arch);
  if (receipt.formatVersion !== 1 || receipt.arch !== arch || receipt.version !== version
    || receipt.sourceRevision !== sourceRevision || !/^[a-f0-9]{40}$/.test(sourceRevision)
    || receipt.sha256 !== hash(bytes) || receipt.bytes !== bytes.length
    || !noticeBytes.length || receipt.noticesSha256 !== hash(noticeBytes)) throw new Error('Bootstrap provenance mismatch');
  return { binary, notices, receipt };
}

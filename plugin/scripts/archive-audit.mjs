import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

/** Python's standard library streams member hashes; no archive paths are extracted. */
export function auditArchive(file, kind) {
  const python = process.env.TENSORVIZ_COMPILER_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  return JSON.parse(execFileSync(python, [fileURLToPath(new URL('./archive-audit.py', import.meta.url)), kind, file],
    { encoding: 'utf8', maxBuffer: 2 * 1024 * 1024 }));
}

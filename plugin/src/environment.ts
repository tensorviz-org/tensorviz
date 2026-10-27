import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { environmentExecutable, runEnvironmentHelper, type HelperResult } from '@tensorviz/graph-host/projectEnvironmentCore';
import { digest, HostError } from './workspace';

export type EnvironmentSelection = { interpreter: string; source: 'explicit' | 'saved' | 'project' | 'managed' };
export type EnvironmentResolution = EnvironmentSelection | { source: 'selection_required'; candidates: string[]; selected?: EnvironmentSelection; detail: string };

export function applicationData(): string {
  return process.platform === 'darwin' ? path.join(os.homedir(), 'Library', 'Application Support', 'TensorViz', 'standalone')
    : process.platform === 'win32' ? path.join(process.env.LOCALAPPDATA ?? path.join(os.homedir(), 'AppData', 'Local'), 'TensorViz', 'standalone')
      : path.join(process.env.XDG_DATA_HOME ?? path.join(os.homedir(), '.local', 'share'), 'tensorviz', 'standalone');
}

export class Environments {
  constructor(readonly backend: string, readonly storage: string = applicationData()) {}

  private choiceFile(root: string): string { return path.join(this.storage, 'workspaces', digest(root), 'environment.json'); }

  async resolve(root: string, explicit?: string): Promise<EnvironmentResolution> {
    let saved: string | undefined;
    let managed = false;
    if (!explicit) {
      try {
        const value = JSON.parse(await fs.readFile(this.choiceFile(root), 'utf8'));
        if (value.workspaceRoot === root && typeof value.interpreter === 'string') {
          saved = value.interpreter;
          managed = value.managed === true;
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw new HostError('ENVIRONMENT_SELECTION_INVALID', 'The saved environment choice cannot be read. Supply an explicit interpreter to replace it.');
      }
    }
    const selected = explicit ?? saved;
    if (selected) {
      const interpreter = environmentExecutable(selected);
      if (!interpreter) {
        if (!path.isAbsolute(selected)) throw new HostError('ENVIRONMENT_UNAVAILABLE', 'Supply an absolute Python executable or environment path.');
        return { source: 'selection_required', selected: { interpreter: selected, source: explicit ? 'explicit' : managed ? 'managed' : 'saved' },
          candidates: await this.candidates(root), detail: 'The selected Python is no longer available. Choose an existing environment; no other Python was substituted.' };
      }
      return { interpreter, source: explicit ? 'explicit' : managed ? 'managed' : 'saved' };
    }
    const candidates = await this.candidates(root);
    return { source: 'selection_required', candidates,
      detail: 'Choose Python for this model. Compatibility will be checked before use. An explicit managed CPU environment is also available through MCP.' };
  }

  async candidates(root: string): Promise<string[]> {
    const paths = ['.venv', 'venv', 'env'].map(name => path.join(root, name));
    paths.push(...[process.env.VIRTUAL_ENV, process.env.CONDA_PREFIX].filter((value): value is string => !!value), ...await this.previous());
    return [...new Set(paths.map(environmentExecutable).filter((value): value is string => !!value))].slice(0, 16);
  }

  /** Available past choices are suggestions, never another project's implicit selection. */
  private async previous(): Promise<string[]> {
    const directory = path.join(this.storage, 'workspaces');
    let entries;
    try { entries = await fs.readdir(directory, { withFileTypes: true }); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []; throw error; }
    const candidates = new Set<string>();
    for (const entry of entries.filter(item => item.isDirectory()).sort((a, b) => a.name.localeCompare(b.name)).slice(0, 128)) {
      try {
        const file = path.join(directory, entry.name, 'environment.json');
        const stat = await fs.lstat(file);
        if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 16384) continue;
        const value = JSON.parse(await fs.readFile(file, 'utf8'));
        // Managed project packages are intentionally isolated per checkout.
        if (value.managed || typeof value.workspaceRoot !== 'string' || digest(value.workspaceRoot) !== entry.name
          || typeof value.interpreter !== 'string' || !path.isAbsolute(value.interpreter)) continue;
        const interpreter = environmentExecutable(value.interpreter);
        if (interpreter) candidates.add(interpreter);
        if (candidates.size >= 16) break;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error;
      }
    }
    return [...candidates];
  }

  /** Remember only a selection that the host has inspected with operation approval. */
  async remember(root: string, interpreter: string, managed = false): Promise<void> {
    const file = this.choiceFile(root);
    await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
    const temporary = `${file}.${process.pid}.tmp`;
    await fs.writeFile(temporary, JSON.stringify({ workspaceRoot: root, interpreter, managed }) + '\n', { mode: 0o600 });
    await fs.rename(temporary, file);
  }

  async executableFingerprint(interpreter: string): Promise<string> {
    const real = await fs.realpath(interpreter), stat = await fs.stat(interpreter);
    return digest(JSON.stringify([interpreter, real, stat.dev, stat.ino, stat.size, stat.mtimeMs]));
  }

  async fingerprint(interpreter: string, managerPython?: string): Promise<string> {
    const result = await this.inspect(interpreter, new AbortController().signal, managerPython);
    if (!result.ok || !result.fingerprint) throw new HostError('ENVIRONMENT_UNAVAILABLE', 'The selected Python inventory could not be read. Inspect or choose an existing environment.');
    return result.fingerprint;
  }

  async inspect(interpreter: string, signal: AbortSignal, managerPython?: string): Promise<HelperResult> {
    const result = await runEnvironmentHelper(managerPython ?? interpreter, this.backend,
      [...(managerPython ? ['--target-python', interpreter] : []), 'inspect-capture'], () => signal.aborted, true, true);
    if (signal.aborted) throw new HostError('CANCELLED', 'Environment inspection was cancelled.');
    return result;
  }
}

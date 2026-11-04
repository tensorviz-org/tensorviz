/** Environment facts and additive-plan subprocess, independent of host UI. */
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { isModelImportDiagnostic, type GraphEnvironmentCandidate, type ModelImportDiagnostic } from '@tensorviz/contracts/protocol';
import { backendPythonArgs, nativeBackend } from './backendPython';

export interface HelperResult {
  ok: boolean;
  code?: string;
  message?: string;
  /** Bounded evidence retained from the already-authorized model import probe. */
  diagnostic?: ModelImportDiagnostic;
  /** Sanitized unavailable requirement, never raw pip output or index URLs. */
  requirement?: string;
  python?: string;
  torch?: string;
  cuda?: string | null;
  pip?: boolean;
  missing?: string[];
  incompatible?: string[];
  /** Exact project declarations behind each model-library conflict. */
  incompatibleRequirements?: string[];
  compatibilityPackages?: string[];
  interpreter?: string;
  digest?: string;
  fingerprint?: string;
  /** `previous` is the installed version an in-place replacement changes. */
  packages?: Array<{ name: string; version: string; previous?: string }>;
  /** The plan changes Torch, its companions or CUDA wheels. */
  replacesTorch?: boolean;
  builtFromSource?: string[];
  requirements?: string[];
  sourcePackages?: string[];
  manifestDigest?: string;
  missingModules?: Array<{ module: string; requirement: string | null; optional?: boolean }>;
  /** Missing imports already declared with a source or condition the installer cannot handle. */
  unsupportedDeclaredModules?: string[];
  filesChecked?: number;
  detail?: string;
  truncated?: boolean;
  environment?: { executable: string; prefix: string; python: string; platform: string; architecture: string;
    torch: string | null; pip: string | null; implementation?: string; abi?: string | null };
  /** `inspect-candidate`: the static model check beside the capture facts; absent when the model could not be checked. */
  model?: { missingModules: Array<{ module: string; requirement: string | null; optional?: boolean }>; incompatible: string[] };
}
export type EnvironmentRunner = (python: string, backendDir: string, args: string[],
  cancelled: () => boolean, isolated?: boolean, parentPipe?: boolean) => Promise<HelperResult>;

export type EnvironmentHint = Pick<GraphEnvironmentCandidate, 'interpreter' | 'source' | 'selected' | 'pythonSelected' | 'managed'>;

/** Only schema-checked worker diagnostics cross into agent/browser state. */
export function parseEnvironmentHelperResult(stdout: string): HelperResult {
  try {
    const result = JSON.parse(stdout.trim().split('\n').pop() ?? '') as HelperResult;
    if (!result || typeof result.ok !== 'boolean') throw new Error();
    if (result.diagnostic !== undefined && !isModelImportDiagnostic(result.diagnostic)) delete result.diagnostic;
    return result;
  } catch {
    return { ok: false, message: 'The selected Python could not complete its compatibility check.' };
  }
}

/** Two bounded metadata probes at a time; neither Torch nor the model is imported.
 *  With `model`, each probe also runs the static model check (`inspect-candidate`),
 *  so an opening can rank every candidate by what the model still needs. */
export async function inspectEnvironmentCandidates(hints: EnvironmentHint[], options: {
  backendDir: string; managerPython?: string; signal: AbortSignal; runner?: EnvironmentRunner;
  cancelled?: () => boolean;
  model?: { workspaceFolder: string; file: string; missingModules?: string[] };
}): Promise<GraphEnvironmentCandidate[]> {
  const action = options.model ? ['inspect-candidate', options.model.workspaceFolder,
    path.join(options.model.workspaceFolder, options.model.file), JSON.stringify(options.model.missingModules ?? [])] : ['inspect-capture'];
  const paths = new Map<string, EnvironmentHint>();
  for (const hint of hints) {
    const key = path.normalize(hint.interpreter), first = paths.get(key);
    if (first) Object.assign(first, hint.selected ? { selected: true } : {}, hint.pythonSelected ? { pythonSelected: true } : {});
    else if (paths.size < 16) paths.set(key, { ...hint });
  }
  const unique = [...paths.values()];
  const rows: GraphEnvironmentCandidate[] = new Array(unique.length);
  let next = 0;
  const inspect = async () => {
    while (next < unique.length && !options.signal.aborted && !options.cancelled?.()) {
      const index = next++, hint = unique[index];
      const id = createHash('sha256').update(hint.interpreter).digest('hex').slice(0, 24);
      let result: HelperResult;
      try {
        result = await (options.runner ?? runEnvironmentHelper)(options.managerPython ?? hint.interpreter, options.backendDir,
          [...(options.managerPython ? ['--target-python', hint.interpreter] : []), ...action],
          () => options.signal.aborted || !!options.cancelled?.(), true, true);
      } catch { result = { ok: false, message: 'This Python environment could not be inspected.' }; }
      options.signal.throwIfAborted();
      const env = result.environment;
      const model = result.ok ? result.model : undefined;
      const modelMissing = (model?.missingModules ?? []).filter(item => !item.optional).map(item => (item.requirement ?? item.module).slice(0, 1024));
      rows[index] = { ...hint, source: hint.source.slice(0, 256), id, status: !result.ok ? 'unavailable' : result.incompatible?.length ? 'incompatible'
        : result.missing?.length ? 'missing_libraries' : 'libraries_present',
        ...(result.python ? { python: result.python.slice(0, 256) } : {}),
        ...(env ? { prefix: env.prefix, torch: env.torch?.slice(0, 256) ?? null, implementation: env.implementation?.slice(0, 256),
          abi: env.abi?.slice(0, 256) ?? null, architecture: env.architecture.slice(0, 256) } : result.torch ? { torch: result.torch.slice(0, 256) } : {}),
        ...(result.fingerprint ? { fingerprint: result.fingerprint } : {}),
        ...(result.missing?.length ? { missing: result.missing.slice(0, 32).map(item => item.slice(0, 1024)) } : {}),
        ...(result.incompatible?.length ? { incompatible: result.incompatible.slice(0, 32).map(item => item.slice(0, 1024)) } : {}),
        ...(!result.ok ? { detail: (result.message ?? 'Python is not available.').slice(0, 1024) } : {}),
        ...(model ? { modelChecked: true as const,
          ...(modelMissing.length ? { modelMissing: modelMissing.slice(0, 32) } : {}),
          ...(model.incompatible.length ? { modelIncompatible: model.incompatible.slice(0, 32).map(item => item.slice(0, 1024)) } : {}) } : {}),
      };
    }
  };
  await Promise.all([inspect(), inspect()]);
  options.signal.throwIfAborted();
  if (options.cancelled?.()) return [];
  // Different aliases can refer to one environment. Venvs sharing a base executable
  // remain separate: it is the resolved prefix, not the symlinked binary, that matters.
  const prefixes = new Map<string, GraphEnvironmentCandidate>();
  return rows.filter(row => {
    let key = row.prefix ?? row.interpreter;
    try { if (row.prefix) key = fs.realpathSync(row.prefix); } catch { /* Unavailable metadata stays visible. */ }
    const first = prefixes.get(key);
    if (first) {
      Object.assign(first, row.selected ? { selected: true } : {}, row.pythonSelected ? { pythonSelected: true } : {});
      return false;
    }
    prefixes.set(key, row); return true;
  });
}

/** Accept executable paths and both Conda/venv directory layouts, including Windows. */
export function environmentExecutable(value: string): string | undefined {
  const candidate = value.startsWith('~/') ? path.join(os.homedir(), value.slice(2)) : value;
  if (!path.isAbsolute(candidate)) return undefined;
  try {
    if (fs.statSync(candidate).isFile()) return candidate;
    for (const relative of process.platform === 'win32'
      ? ['python.exe', 'Scripts/python.exe'] : ['bin/python', 'bin/python3']) {
      const executable = path.join(candidate, relative);
      if (fs.statSync(executable, { throwIfNoEntry: false })?.isFile()) return executable;
    }
  } catch { /* A removed environment is offered again through the picker. */ }
  return undefined;
}

/** Kill the setup process and pip children together, including when the user cancels. */
export function createEnvironmentRunner(entryPoint: string): EnvironmentRunner {
  return (python, backendDir, args, cancelled, isolated = false, parentPipe = false) => new Promise(resolve => {
  if (cancelled()) { resolve({ ok: false, message: 'Environment setup cancelled.' }); return; }
  // Native probes must load the compiled worker in the selected interpreter.
  // Metadata and package operations still belong to the separate manager.
  if (nativeBackend(backendDir) && args[0] === '--target-python'
    && ['probe-capture', 'probe-model'].includes(args[2])) {
    python = args[1]; args = args.slice(2);
  }
  const child = spawn(python, backendPythonArgs(backendDir, entryPoint, args, isolated), {
    cwd: backendDir, detached: process.platform !== 'win32', windowsHide: true,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1', KMP_DUPLICATE_LIB_OK: 'TRUE', TENSORVIZ_PARENT_PIPE: parentPipe ? '1' : undefined },
    stdio: [parentPipe ? 'pipe' : 'ignore', 'pipe', 'pipe'],
  });
  let stdout = '';
  let settled = false;
  const finish = (result: HelperResult) => {
    if (settled) return;
    settled = true;
    clearTimeout(timeout); clearInterval(poll);
    resolve(result);
  };
  const stop = (message: string) => {
    if (child.pid) {
      try {
        if (process.platform === 'win32') {
          spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true }).unref();
        } else process.kill(-child.pid, 'SIGKILL');
      } catch { /* Already exited. */ }
    }
    finish({ ok: false, message });
  };
  const action = args[0] === '--target-python' ? args[2] : args[0];
  const timeout = setTimeout(() => stop('Environment setup timed out. Check the environment and retry.'),
    action === 'inspect-capture' || action === 'inspect-runtime' ? 15_000 : action === 'inspect-candidate' ? 30_000
      : action.startsWith('inspect') || action.startsWith('probe') ? 90_000 : 20 * 60_000);
  const poll = setInterval(() => { if (cancelled()) stop('Environment setup cancelled.'); }, 200);
  child.stdout!.on('data', data => { stdout = (stdout + String(data)).slice(-128_000); });
  child.stderr!.resume(); // Import/pip output can contain private configuration; do not log it.
  child.on('error', () => finish({ ok: false, message: 'Could not start the selected Python. Choose another environment.' }));
  child.on('close', () => finish(parseEnvironmentHelperResult(stdout)));
});
}

export const runEnvironmentHelper = createEnvironmentRunner(path.join('services', 'ide', 'environment_setup.py'));

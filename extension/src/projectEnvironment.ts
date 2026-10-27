/** User-selected project Python, independent of the tool service runtime. */
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { BackendInterpreterOptions, BackendInterpreterResult, managedInterpreter } from './preflight';
import { userInterpreter } from './worker';
import { environmentExecutable, createEnvironmentRunner, inspectEnvironmentCandidates, type EnvironmentHint, type EnvironmentRunner } from '@tensorviz/graph-host/projectEnvironmentCore';
import type { GraphEnvironmentCandidate } from '@tensorviz/contracts/protocol';
export { environmentExecutable, type EnvironmentRunner } from '@tensorviz/graph-host/projectEnvironmentCore';

export const runEnvironmentHelper = createEnvironmentRunner('extension_environment.py');

export const ENVIRONMENT_KEY = 'tensorviz.selectedEnvironment';
export const ENVIRONMENTS_KEY = 'tensorviz.selectedEnvironments';
/** `explicit`: the user chose it (picker, Browse, Canvas selection, a created environment or an
 * approved compatibility copy). A choice tensorViz saved itself, or one saved before this flag
 * existed, may be replaced at opening by an environment that is ready for the model. */
export type EnvironmentChoice = ({ kind: 'project'; interpreter: string } | { kind: 'managed' }) & { explicit?: true };
type Context = Pick<vscode.ExtensionContext, 'workspaceState' | 'globalState' | 'globalStorageUri'>;
export type EnvironmentResult = BackendInterpreterResult | { ok: false; cancelled: true; message: string; detail: string };

/** An explicit choice stays authoritative, even when broken. Other roots and
 * global environments are alternatives, never an automatic execution choice. */
export function automaticProjectCandidate(candidates: GraphEnvironmentCandidate[]): GraphEnvironmentCandidate | undefined {
  const selected = candidates.find(candidate => candidate.selected);
  if (selected) return selected;
  const pythonSelected = candidates.find(candidate => candidate.pythonSelected && !candidate.managed);
  if (pythonSelected) return pythonSelected;
  const local = candidates.filter(candidate => !candidate.managed
    && ['Model’s project folder', 'Owning workspace folder'].includes(candidate.source));
  return local.length === 1 ? local[0] : undefined;
}

/** Capture libraries present and nothing missing or mismatched for the model. Metadata
 * only: the opening still runs its model and worker checks on the environment it uses. */
export function readyForModel(candidate: GraphEnvironmentCandidate): boolean {
  return candidate.status === 'libraries_present' && candidate.modelChecked === true
    && !candidate.modelMissing?.length && !candidate.modelIncompatible?.length;
}

/** Libraries to add, plus three per version change: replacing an installed version can
 * break whatever else uses the environment. Undefined when it should not be recommended:
 * unavailable, conflicting capture libraries (including an old Python), no Torch or no model check. */
export function environmentCost(candidate: GraphEnvironmentCandidate): number | undefined {
  if (candidate.status === 'unavailable' || candidate.status === 'incompatible' || !candidate.torch || candidate.modelChecked !== true) return undefined;
  // The capture and model checks can both name one distribution (numpy, for example); it installs once.
  const distribution = (requirement: string) => (/^\s*([A-Za-z0-9][A-Za-z0-9._-]*)/.exec(requirement)?.[1] ?? requirement).toLowerCase().replace(/[-_.]+/g, '-');
  const install = new Set([...candidate.missing ?? [], ...candidate.modelMissing ?? []].map(distribution));
  return install.size + 3 * (candidate.modelIncompatible?.length ?? 0);
}

const LOCAL_SOURCES = ['Model’s project folder', 'Owning workspace folder', 'Previously selected'];
function preference(candidate: GraphEnvironmentCandidate): number {
  if (candidate.selected) return 0;
  if (candidate.pythonSelected) return 1;
  if (candidate.managed) return 6;
  const local = LOCAL_SOURCES.indexOf(candidate.source);
  return local < 0 ? 5 : 2 + local;
}

/** Fewest changes first; ties prefer the saved choice, the VS Code selection and the model's
 * own folders. The shared tensorViz-managed runtime and unusable environments come last. */
export function rankEnvironmentCandidates(candidates: GraphEnvironmentCandidate[]): GraphEnvironmentCandidate[] {
  const key = (candidate: GraphEnvironmentCandidate) => [environmentCost(candidate)
    ?? (candidate.status === 'unavailable' ? Number.MAX_SAFE_INTEGER : Number.MAX_SAFE_INTEGER - 1), preference(candidate)];
  return candidates.map((candidate, index) => ({ candidate, index, key: key(candidate) }))
    .sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1] || a.index - b.index).map(item => item.candidate);
}

export function environmentFolder(resource?: vscode.Uri): vscode.WorkspaceFolder | undefined {
  const folders = vscode.workspace.workspaceFolders ?? [];
  return resource ? vscode.workspace.getWorkspaceFolder(resource)
    ?? folders.find(folder => folder.uri.fsPath === resource.fsPath) : folders[0];
}

/** The old picker belonged to the first folder. Never migrate its answer into another root. */
export function savedEnvironment(context: Pick<Context, 'workspaceState'>, resource?: vscode.Uri): EnvironmentChoice | undefined {
  const scope = environmentFolder(resource)?.uri.fsPath;
  if (!scope) return undefined;
  const choices = context.workspaceState.get<Record<string, EnvironmentChoice>>(ENVIRONMENTS_KEY);
  if (choices) return choices[scope];
  return scope === vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
    ? context.workspaceState.get<EnvironmentChoice>(ENVIRONMENT_KEY) : undefined;
}

/** Selections name host-local runtimes and belong only to the model's owning folder. */
export async function rememberEnvironment(context: Context, choice: EnvironmentChoice, resource?: vscode.Uri): Promise<void> {
  const scope = environmentFolder(resource)?.uri.fsPath;
  if (!scope) throw new Error('Select a model in a workspace folder before choosing its Python environment.');
  const choices = { ...context.workspaceState.get<Record<string, EnvironmentChoice>>(ENVIRONMENTS_KEY) };
  const legacy = context.workspaceState.get<EnvironmentChoice>(ENVIRONMENT_KEY);
  const first = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (legacy && first && !choices[first]) choices[first] = legacy;
  choices[scope] = choice;
  await context.workspaceState.update(ENVIRONMENTS_KEY, choices);
  await context.workspaceState.update(ENVIRONMENT_KEY, undefined);
}

/** One read-only choice for profile resolution, capture and candidate execution.
 * A saved choice is authoritative; checking a ready managed runtime never provisions it.
 * Explicit missing paths must fail in that environment, not silently switch Python. */
export async function selectedExecutionInterpreter(context: Context, resource: vscode.Uri,
  readyManaged: () => Promise<string | undefined>): Promise<string | undefined> {
  return (await selectedExecutionEnvironment(context, resource, readyManaged)).interpreter;
}

export async function selectedExecutionEnvironment(context: Context, resource: vscode.Uri,
  readyManaged: () => Promise<string | undefined>): Promise<{ interpreter?: string;
    source: 'project' | 'managed' | 'setting' | 'python-extension' }> {
  const selected = savedEnvironment(context, resource);
  if (selected?.kind === 'project') return { interpreter: selected.interpreter, source: 'project' };
  if (selected?.kind === 'managed') return { interpreter: await readyManaged(), source: 'managed' };
  const configured = vscode.workspace.getConfiguration('tensorviz', resource).get<string>('pythonPath', '').trim();
  if (configured) return { interpreter: environmentExecutable(configured), source: 'setting' };
  const active = await userInterpreter(resource);
  return active ? { interpreter: active, source: 'python-extension' }
    : { interpreter: await readyManaged(), source: 'managed' };
}

/** Only bounded, host-local sources. Probes read metadata and never import the model. */
export async function discoverEnvironments(context: Context, backendDir: string, options: {
  resource?: vscode.Uri; workspaceRoot?: string; signal: AbortSignal; managerPython?: string; runner?: EnvironmentRunner;
  readyManaged?: () => Promise<string | undefined>;
  cancelled?: () => boolean;
  /** Also check each candidate for this model's libraries (opening only). */
  model?: { workspaceFolder: string; file: string; missingModules?: string[] };
}): Promise<GraphEnvironmentCandidate[]> {
  if (!vscode.workspace.isTrusted) throw new Error('Trust this workspace before checking Python environments.');
  const resource = options.resource;
  const root = environmentFolder(resource)?.uri;
  const configured = vscode.workspace.getConfiguration('tensorviz', resource ?? root).get<string>('pythonPath', '').trim();
  const saved = savedEnvironment(context, resource);
  const [active, known, managed] = await Promise.all([
    resource || root ? userInterpreter(resource ?? root!) : Promise.resolve(undefined),
    (async () => {
      try {
        const api = await vscode.extensions.getExtension('ms-python.python')?.activate() as {
          environments?: { known?: ReadonlyArray<{ path?: string; executable?: { uri?: vscode.Uri } }> } } | undefined;
        return (api?.environments?.known ?? []).slice(0, 64).map(env => env.executable?.uri?.fsPath ?? env.path ?? '');
      } catch { return []; }
    })(),
    options.readyManaged?.() ?? Promise.resolve(undefined),
  ]);
  options.signal.throwIfAborted();
  const hints: EnvironmentHint[] = [];
  const add = (value: string | undefined, source: string, explicit = false, flags: Partial<EnvironmentHint> = {}) => {
    if (!value) return;
    const interpreter = environmentExecutable(value) ?? (explicit && path.isAbsolute(value) ? value : undefined);
    if (interpreter) hints.push({ interpreter, source, ...flags });
  };
  if (saved?.kind === 'project') add(saved.interpreter, 'Saved for this folder', true, { selected: true });
  add(configured, 'tensorViz Python setting', true, !saved ? { selected: true } : {});
  add(active, 'Selected in VS Code', true, { pythonSelected: true });
  const projectRoot = options.workspaceRoot ?? root?.fsPath;
  if (projectRoot) for (const name of ['.venv', 'venv', 'env']) add(path.join(projectRoot, name), 'Model’s project folder');
  if (root && root.fsPath !== projectRoot) for (const name of ['.venv', 'venv', 'env']) add(path.join(root.fsPath, name), 'Owning workspace folder');
  for (const folder of (vscode.workspace.workspaceFolders ?? []).slice(0, 16)) {
    if (folder.uri.fsPath !== root?.fsPath) for (const name of ['.venv', 'venv', 'env']) add(path.join(folder.uri.fsPath, name), `Other workspace folder: ${path.basename(folder.uri.fsPath)}`);
  }
  for (const value of known) add(value, 'Known to the Python extension');
  add(process.env.VIRTUAL_ENV, 'Active virtual environment'); add(process.env.CONDA_PREFIX, 'Active Conda environment');
  for (const choice of Object.values(context.workspaceState.get<Record<string, EnvironmentChoice>>(ENVIRONMENTS_KEY) ?? {}).slice(0, 16)) {
    if (choice.kind === 'project') add(choice.interpreter, 'Previously selected');
  }
  const managedHint: EnvironmentHint | undefined = managed ? { interpreter: managed, source: 'tensorViz-managed fallback', managed: true,
    ...(saved?.kind === 'managed' ? { selected: true } : {}) } : undefined;
  if (managedHint) saved?.kind === 'managed' ? hints.unshift(managedHint) : hints.push(managedHint);
  // Reserve the last slot for the ready managed fallback when many environments exist.
  const bounded = managedHint && !managedHint.selected ? [...hints.slice(0, 15), managedHint] : hints;
  return inspectEnvironmentCandidates(bounded, { ...options, backendDir, runner: options.runner ?? runEnvironmentHelper });
}

async function pickEnvironment(context: Context, backendDir: string, options: Parameters<typeof discoverEnvironments>[2] & { selectionOnly?: true }): Promise<EnvironmentChoice | undefined> {
  const candidates = await discoverEnvironments(context, backendDir, options);
  if (options.cancelled?.()) return undefined;
  const items: Array<vscode.QuickPickItem & { choice?: EnvironmentChoice; action?: string }> = candidates.map(candidate => ({
    label: candidate.interpreter,
    description: [candidate.selected ? 'Used by tensorViz' : candidate.source, candidate.pythonSelected ? 'Selected in VS Code' : ''].filter(Boolean).join(' · '),
    detail: [candidate.python ? `Python ${candidate.python}` : '', candidate.torch ? `Torch ${candidate.torch}` : '',
      candidate.status === 'libraries_present' ? 'Capture libraries present' : candidate.status === 'missing_libraries'
        ? `Needs ${candidate.missing?.length} capture libraries: ${candidate.missing?.join(', ')}`
        : candidate.status === 'incompatible' ? `Incompatible: ${candidate.incompatible?.join(', ')}` : 'Not available'].filter(Boolean).join(' · '),
    choice: candidate.managed ? { kind: 'managed', explicit: true } : { kind: 'project', interpreter: candidate.interpreter, explicit: true },
  }));
  items.push({ label: 'Enter Python or environment path…', action: 'path' },
    { label: 'Use a separate tensorViz environment', description: options.selectionOnly ? 'Select the tensorViz fallback' : 'Download only if needed; asks before installing',
      detail: options.selectionOnly ? 'Library checks and setup continue in tensorViz after selection.'
        : 'Includes tensorViz and PyTorch. Opening a graph detects and offers to install its missing model libraries here.', choice: { kind: 'managed', explicit: true } });
  const picked = await vscode.window.showQuickPick(items, {
    title: 'Choose the Python environment for tensorViz',
    placeHolder: 'Reuse your project environment to avoid installing PyTorch again',
    matchOnDescription: true,
  });
  if (!picked) return undefined;
  if (picked.choice) return picked.choice;
  const value = await vscode.window.showInputBox({ title: 'Python environment',
    prompt: 'Absolute path to the Python executable or a Conda/venv folder', ignoreFocusOut: true });
  if (!value) return undefined;
  const executable = environmentExecutable(value.trim());
  if (!executable) {
    await vscode.window.showErrorMessage('No Python executable was found at that path. Select the environment again.');
    return undefined;
  }
  return { kind: 'project', interpreter: executable, explicit: true };
}


const cancellation = (): Extract<EnvironmentResult, { ok: false }> => ({ ok: false, cancelled: true, message: 'Environment selection cancelled.', detail: '' });

/** Browse saves a choice; Canvas owns subsequent checks, repair and progress. */
export async function selectProjectEnvironment(context: Context, backendDir: string,
  options: NonNullable<Parameters<typeof resolveCaptureEnvironment>[2]> = {},
): Promise<{ ok: true; banner: string } | Extract<EnvironmentResult, { ok: false }>> {
  if (!vscode.workspace.isTrusted) return { ok: false, message: 'Trust this workspace before selecting its Python environment.', detail: '' };
  if (!environmentFolder(options.resource)) return { ok: false, message: 'Select a model in a workspace folder before choosing its Python environment.', detail: '' };
  const cancelled = options.isCancelled ?? (() => false);
  if (cancelled()) return cancellation();
  const choice = await (options.picker?.() ?? pickEnvironment(context, backendDir, { ...options, cancelled,
    selectionOnly: true, signal: new AbortController().signal,
    runner: (python, backend, args, probeCancelled, ...rest) => (options.runner ?? runEnvironmentHelper)(python, backend, args,
      () => cancelled() || probeCancelled(), ...rest),
    readyManaged: async () => { const result = await options.runtime?.check(); return result?.state === 'ready' ? result.interpreter : undefined; },
  }));
  if (!choice || cancelled()) return cancellation();
  await rememberEnvironment(context, choice, options.resource);
  return { ok: true, banner: choice.kind === 'managed' ? 'tensorViz fallback selected. Library checks and setup continue in tensorViz.'
    : `Python selected: ${choice.interpreter}. Library checks and setup continue in tensorViz.` };
}

/** Check only the isolated capture worker, then review missing worker libraries. */
export async function resolveCaptureEnvironment(
  context: Context, backendDir: string,
  options: BackendInterpreterOptions & { forcePicker?: boolean; runner?: EnvironmentRunner;
    picker?: () => Promise<EnvironmentChoice | undefined>; managerPython?: string; resource?: vscode.Uri; workspaceRoot?: string } = {},
): Promise<EnvironmentResult> {
  if (!vscode.workspace.isTrusted) return { ok: false, message: 'Trust this workspace before selecting or running its Python environment.', detail: '' };
  if (!environmentFolder(options.resource)) return { ok: false, message: 'Select a model in a workspace folder before choosing its Python environment.', detail: '' };
  const cancelled = options.isCancelled ?? (() => false);
  const run: EnvironmentRunner = (python, backend, args, ...rest) => (options.runner ?? runEnvironmentHelper)(
    options.managerPython ?? python, backend, options.managerPython ? ['--target-python', python, ...args] : args, ...rest);
  let choice = options.forcePicker ? undefined : savedEnvironment(context, options.resource);
  for (;;) {
    if (cancelled()) return cancellation();
    choice ??= await (options.picker?.() ?? pickEnvironment(context, backendDir, { ...options, cancelled, signal: new AbortController().signal,
      runner: (python, backend, args, probeCancelled, ...rest) => (options.runner ?? runEnvironmentHelper)(python, backend, args,
        () => cancelled() || probeCancelled(), ...rest),
      readyManaged: async () => { const result = await options.runtime?.check(); return result?.state === 'ready' ? result.interpreter : undefined; },
    }));
    if (!choice || cancelled()) return cancellation();
    if (choice.kind === 'managed') {
      const result = await managedInterpreter(options);
      if (cancelled() || result.cancelled) return cancellation();
      if (!result.interpreter) return { ok: false, message: 'The separate tensorViz environment is not ready.', detail: result.note, repairRuntime: result.repairable };
      await rememberEnvironment(context, choice, options.resource);
      return { ok: true, interpreter: result.interpreter, banner: `python OK: ${result.interpreter} — tensorViz managed runtime` };
    }
    options.report?.('Checking the selected environment…');
    const info = await run(choice.interpreter, backendDir, ['inspect-capture'], cancelled);
    if (cancelled()) return cancellation();
    const probe = info.ok ? await run(choice.interpreter, backendDir, ['probe-capture'], cancelled) : info;
    if (cancelled()) return cancellation();
    if (probe.ok && !info.missing?.length && !info.incompatible?.length) {
      await rememberEnvironment(context, choice, options.resource);
      return { ok: true, interpreter: choice.interpreter,
        banner: `python OK: ${choice.interpreter} — Python ${info.python}, Torch ${probe.torch}${probe.cuda ? `, CUDA ${probe.cuda}` : ''}` };
    }
    const action = await vscode.window.showWarningMessage(
      info.incompatible?.length ? 'Some installed versions are incompatible with tensorViz.'
        : probe.message ?? 'This environment needs capture libraries before opening the model.',
      { modal: true, detail: `${choice.interpreter}\n${info.torch ? `Existing Torch: ${info.torch}.` : info.ok ? 'PyTorch is missing; installing it can be a large download.' : ''}\n` +
        (info.incompatible?.length ? `Incompatible versions: ${info.incompatible.join(', ')}.\n` : '') +
        (info.missing?.length ? `Missing libraries: ${info.missing.join(', ')}.\n` : '') +
        (info.ok && !info.pip && !options.managerPython ? 'The tool installer is unavailable. Choose another environment or use MCP to inspect setup.\n' : '') +
        'Review checks compatible wheels without changing installed packages. You can also choose another environment.' },
      ...(info.ok && (info.pip || options.managerPython) && !info.incompatible?.length ? ['Review missing libraries'] : []), 'Choose environment');
    if (cancelled()) return cancellation();
    if (action === 'Choose environment') { choice = undefined; continue; }
    if (action !== 'Review missing libraries') return cancellation();
    await fs.promises.mkdir(context.globalStorageUri.fsPath, { recursive: true });
    const directory = await fs.promises.mkdtemp(path.join(context.globalStorageUri.fsPath, 'environment-plan-'));
    try {
      options.report?.('Resolving missing libraries while preserving installed versions…');
      const plan = await run(choice.interpreter, backendDir, ['plan-capture', directory], cancelled);
      if (cancelled()) return cancellation();
      if (!plan.ok || !plan.digest || !plan.packages?.length) {
        await vscode.window.showErrorMessage(plan.message ?? 'No additive install can fix this environment. Choose another environment or repair it manually.');
        return cancellation();
      }
      const consent = await vscode.window.showInformationMessage(
        `Install ${plan.packages.length} missing libraries in this environment?`,
        { modal: true, detail: `${choice.interpreter}\n\n${plan.packages.map(p => `${p.name} ${p.version}`).join('\n')}\n\n` +
          'Existing packages, including Torch/CUDA, will not be replaced. This downloads public PyPI wheels. Cancellation can leave some new libraries installed.' },
        'Install missing libraries', 'Cancel');
      if (consent !== 'Install missing libraries' || cancelled()) return cancellation();
      options.report?.('Installing the approved missing libraries…');
      const installed = await run(choice.interpreter, backendDir, ['install', directory, plan.digest], cancelled);
      if (cancelled()) return cancellation();
      if (!installed.ok) return { ok: false, message: installed.message ?? 'The missing libraries could not be installed.', detail: 'Some new libraries may have been installed. Existing packages were not scheduled for replacement.' };
      const verified = await run(choice.interpreter, backendDir, ['probe-capture'], cancelled);
      if (cancelled()) return cancellation();
      if (!verified.ok) return { ok: false, message: verified.message ?? 'The capture worker still cannot load.', detail: 'Choose another environment or use MCP to investigate this runtime.' };
      await rememberEnvironment(context, choice, options.resource);
      return { ok: true, interpreter: choice.interpreter,
        banner: `python OK: ${choice.interpreter} — Torch ${verified.torch}${verified.cuda ? `, CUDA ${verified.cuda}` : ''}; missing libraries installed` };
    } finally { await fs.promises.rm(directory, { recursive: true, force: true }); }
  }
}

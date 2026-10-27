import { isHostFile } from './workspaceFiles';
import { createHash } from 'node:crypto';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { isGraphSetupDiscovery, isGraphSetupDraft, type GraphSetupDraft, type LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { multifileProjectionBody, sha256Text, vscodeSourceIO, type SourceIO } from './workspaceSources';
import { readSetupFile, saveSetupFiles, openSetupHelper, SetupFileError, type PreparedGraphSetup, type SetupFileSnapshot } from './graphSetupFiles';
import type { GraphSetupPreparation } from '@tensorviz/graph-host/graphOpening';
import type { ProfileLookupOptions } from '@tensorviz/graph-host/profileValidation';
import type { LocalBrowserTraceSupport } from './localBrowserSync';
import { translateProfile } from '@tensorviz/graph-host/validationProfileBridge';
import { executionFailure } from '@tensorviz/graph-host/executionFailure';
import { executionDiagnosticDetail } from './executionDiagnostic';
import type { RunSpec } from '@tensorviz/contracts/protocol';

interface Snapshot {
  target: LocalBrowserTargetIdentity;
  sources: Awaited<ReturnType<typeof multifileProjectionBody>>;
  config?: SetupFileSnapshot;
  runtimeDraft?: GraphSetupDraft;
  configDefaults?: boolean;
}
export interface GraphSetupHostOptions {
  origin(): string | undefined;
  token(): string | undefined;
  current(target: LocalBrowserTargetIdentity): boolean;
  profiles?: ProfileLookupOptions;
  sourceIO?: (root: string) => SourceIO;
  fetchImpl?: typeof fetch;
}
const same = (a: LocalBrowserTargetIdentity, b: LocalBrowserTargetIdentity) => a.workspaceFolder === b.workspaceFolder
  && a.file === b.file && a.cls === b.cls && a.sourceDigest === b.sourceDigest && a.documentVersion === b.documentVersion;

/** Resolve and prepare setup against the exact source/config snapshot shown by the form. */
export class GraphSetupHost {
  private snapshot?: Snapshot;
  constructor(private readonly options: GraphSetupHostOptions) {}

  private check(target: LocalBrowserTargetIdentity, signal: AbortSignal): void {
    signal.throwIfAborted();
    if (!this.options.current(target)) throw new SetupFileError('The selected source changed. Reopen setup.');
  }

  private async sources(target: LocalBrowserTargetIdentity) {
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, target.file)));
    return multifileProjectionBody({ io: (this.options.sourceIO ?? vscodeSourceIO)(target.workspaceFolder),
      file: target.file, cls: target.cls, source: document.getText(), documentVersion: document.version });
  }

  private async post(endpoint: string, body: unknown, signal: AbortSignal): Promise<Record<string, any>> {
    const origin = this.options.origin(), token = this.options.token();
    if (!origin || !token) throw new SetupFileError('The local backend is unavailable. Reopen the graph.');
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    const timeout = setTimeout(abort, 30000);
    try {
      const response = await (this.options.fetchImpl ?? fetch)(`${origin}/api/ide/graph-setup/${endpoint}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': token },
        body: JSON.stringify(body), signal: controller.signal,
      });
      if (!response.ok) throw new SetupFileError('Setup could not be prepared. Check required fields and existing tensorviz.toml in VS Code.');
      const text = await response.text();
      if (Buffer.byteLength(text) > 512 * 1024) throw new SetupFileError('Setup exceeds the supported size. Use a smaller explicit profile.');
      const value = JSON.parse(text);
      if (!value || typeof value !== 'object' || Array.isArray(value)) throw new SetupFileError('Invalid setup response.');
      return value;
    } finally {
      clearTimeout(timeout);
      signal.removeEventListener('abort', abort);
    }
  }

  async discover(target: LocalBrowserTargetIdentity, signal: AbortSignal, saved?: GraphSetupDraft): Promise<GraphSetupPreparation> {
    this.check(target, signal);
    const sources = await this.sources(target);
    this.check(target, signal);
    if (sources.root.source_digest !== target.sourceDigest) throw new SetupFileError('The selected source changed.');
    const { restoredDraft, ...discovery } = await this.post('discover', { ...sources, ...(saved ? { profile: saved.profile } : {}) }, signal);
    if (!isGraphSetupDiscovery(discovery) || discovery.sourceDigest !== target.sourceDigest) throw new SetupFileError('Model setup is unavailable for this source snapshot.');
    if (restoredDraft != null && (!saved || !isGraphSetupDraft(restoredDraft) || !restoredDraft.configObject
      || restoredDraft.configObject.index >= discovery.configs.length || restoredDraft.profile.model !== discovery.model)) {
      throw new SetupFileError('The saved configuration fields could not be restored.');
    }
    const snapshot: Snapshot = { target: { ...target }, sources,
      ...(discovery.resolution?.runtimeInputs ? { runtimeDraft: discovery.resolution.draft,
        configDefaults: discovery.init === null && !discovery.resolution.draft.profile.factory } : {}) };
    const requirements: string[] = [];
    try { snapshot.config = await readSetupFile(target.workspaceFolder); }
    catch (error) { requirements.push(error instanceof SetupFileError ? error.message : 'The setup file could not be read. Check tensorviz.toml in VS Code.'); }
    this.check(target, signal);
    this.snapshot = snapshot;
    const dirty = vscode.workspace.textDocuments.some(doc => doc.isDirty && isHostFile(doc.uri) && this.inside(target.workspaceFolder, doc.uri.fsPath)
      && (/\.pyw?$/i.test(doc.uri.fsPath) || path.basename(doc.uri.fsPath) === 'tensorviz.toml'));
    if (dirty) requirements.push('Save the model and changed setup/dependency files before capturing.');
    return { discovery, ...(discovery.resolution ? { draft: discovery.resolution.draft } : {}),
      ...(restoredDraft ? { restoredDraft: { ...restoredDraft, ...(saved?.profileName ? { profileName: saved.profileName } : {}) } } : {}),
      ...(requirements.length ? { requirements } : {}) };
  }

  async inferInputs(target: LocalBrowserTargetIdentity, draft: GraphSetupDraft, signal: AbortSignal,
    trace: LocalBrowserTraceSupport): Promise<{ draft?: GraphSetupDraft; detail: string; missingModules?: string[]; importFailure?: true }> {
    const snapshot = this.snapshot;
    this.check(target, signal);
    const unavailable = { detail: 'An input example could not be determined. Complete the input fields to capture this model.' };
    if (!snapshot || !same(snapshot.target, target) || !snapshot.runtimeDraft || draft.configObject
      || JSON.stringify(snapshot.runtimeDraft) !== JSON.stringify(draft)) return unavailable;
    const translated = translateProfile(draft.profile as unknown as RunSpec['profile'], 'example');
    if (!translated.ok || !await trace.trusted()) return unavailable;
    this.check(target, signal);
    const beforeLaunch = async () => {
      this.check(target, signal);
      const fresh = await this.sources(target);
      const config = await readSetupFile(target.workspaceFolder);
      this.check(target, signal);
      return vscode.workspace.isTrusted && this.sourceIdentity(fresh) === this.sourceIdentity(snapshot.sources)
        && JSON.stringify(config) === JSON.stringify(snapshot.config);
    };
    const abort = () => trace.worker.cancel();
    signal.addEventListener('abort', abort, { once: true });
    try {
      if (!await beforeLaunch()) return unavailable;
      const result = await trace.worker.run({ file: path.join(target.workspaceFolder, target.file), cls: target.cls,
        workspace: target.workspaceFolder, execution: translated.run.execution, prepareInputs: true,
        ...(snapshot.configDefaults ? { prepareConfigDefaults: true } : {}),
        timeoutMs: 30000, beforeLaunch });
      this.check(target, signal);
      if (!result.ok || !result.executionCurrent || !await result.recheckExecution?.(signal) || !await beforeLaunch()) {
        return result.ok ? unavailable : { detail: executionDiagnosticDetail(result.envelope?.error?.diagnostic, target.cls, result.envelope?.error?.node_id)
          ?? 'Input preparation failed. Check the selected Python environment and model configuration.',
          missingModules: executionFailure(result.envelope?.error?.diagnostic).missingModules,
          importFailure: executionFailure(result.envelope?.error?.diagnostic).importFailure };
      }
      const proposal = (result.envelope.result as { graphExample?: { binding?: unknown; detail?: unknown; configDefaults?: unknown } } | null)?.graphExample;
      if (proposal?.configDefaults !== undefined && (proposal.configDefaults !== true || !snapshot.configDefaults)) return unavailable;
      const inferred = { ...draft, ...(proposal?.configDefaults ? { configDefaults: true as const } : {}),
        profile: { ...draft.profile, inputs: proposal?.binding } };
      if (!proposal?.binding || typeof proposal.detail !== 'string' || proposal.detail.length > 512 || !isGraphSetupDraft(inferred)) return unavailable;
      this.check(target, signal);
      return { draft: inferred, detail: proposal.detail };
    } finally { signal.removeEventListener('abort', abort); }
  }

  private sourceIdentity(body: Snapshot['sources']): string {
    return JSON.stringify([body.root.source_digest, body.truncated, body.sources.map(entry => [entry.file, entry.digest])]);
  }

  async save(target: LocalBrowserTargetIdentity, draft: GraphSetupDraft, signal: AbortSignal): Promise<{ requirements?: string[] }> {
    const snapshot = this.snapshot;
    this.check(target, signal);
    if (!snapshot || !same(snapshot.target, target) || !snapshot.config) {
      return { requirements: ['Save or discard changes to tensorviz.toml, then reopen setup.'] };
    }
    const fresh = await this.sources(target);
    if (this.sourceIdentity(fresh) !== this.sourceIdentity(snapshot.sources)) return { requirements: ['Workspace source changed while setup was open. Reopen setup before saving.'] };
    this.check(target, signal);
    const result = await this.post('profile', { sources: snapshot.sources, config: snapshot.config.text,
      profile: draft.profile, ...(draft.profileName ? { profile_name: draft.profileName } : {}),
      ...(draft.configObject ? { config_object: draft.configObject } : {}),
      ...(draft.configDefaults ? { config_defaults: true } : {}) }, signal);
    this.check(target, signal);
    if (result.ok === false && Array.isArray(result.requirements)) return { requirements: result.requirements.slice(0, 64).map(item =>
      `${Array.isArray(item.field) ? item.field.join('.') : 'setup'}: ${typeof item.reason === 'string' ? item.reason : 'Complete this field.'}`.slice(0, 1024)) };
    if (result.ok !== true || !/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(result.profile)
      || typeof result.config !== 'string' || Buffer.byteLength(result.config) > 262144
      || result.factory !== null && (!result.factory || typeof result.factory.file !== 'string'
        || typeof result.factory.source !== 'string' || Buffer.byteLength(result.factory.source) > 131072)) {
      throw new SetupFileError('The prepared setup files are invalid.');
    }
    await saveSetupFiles(target.workspaceFolder, snapshot.config, result as unknown as PreparedGraphSetup,
      () => this.options.current(target), signal);
    this.check(target, signal);
    await this.options.profiles?.remember?.(target.workspaceFolder, target, result.profile);
    this.snapshot = undefined;
    return {};
  }

  async factoryTemplate(target: LocalBrowserTargetIdentity, draft: GraphSetupDraft | undefined, signal: AbortSignal): Promise<GraphSetupDraft> {
    this.check(target, signal);
    const prepared = await this.discover(target, signal);
    const [module, cls] = prepared.discovery.model.split(':');
    if (!module?.split('.').every(part => /^[A-Za-z_]\w*$/.test(part)) || !/^[A-Za-z_]\w*$/.test(cls)) {
      throw new SetupFileError('This model needs a hand-written factory with an importable class.');
    }
    const fields = prepared.discovery.init?.filter(field => field.required).map(field => field.name).join(', ') || 'your model configuration';
    const source = `# Complete this factory, then save it and continue graph setup.\n\ndef make_model():\n    from ${module} import ${cls}\n    # Provide ${fields}. Return an instance of ${cls}.\n    raise NotImplementedError("Complete the model factory before capturing")\n`;
    const file = `.tensorviz/setup_${cls}_${createHash('sha256').update(source).digest('hex').slice(0, 12)}.py`;
    await openSetupHelper(target.workspaceFolder, file, source, () => this.options.current(target), signal);
    return { ...draft, configObject: undefined, configDefaults: undefined, profile: { kind: 'model', model: prepared.discovery.model,
      mode: 'eval', device: 'cpu', grad_enabled: false, inputs: { args: [], kwargs: {} }, ...draft?.profile,
      init: undefined, factory: { file, callable: 'make_model', kwargs: {} } } };
  }

  private inside(root: string, file: string): boolean {
    const relative = path.relative(root, file);
    return !!relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
  }

  /** Explicit Save & Open saves only Python/setup files in the selected workspace. */
  async saveSources(target: LocalBrowserTargetIdentity, signal: AbortSignal): Promise<LocalBrowserTargetIdentity | undefined> {
    if (!vscode.workspace.isTrusted) return undefined;
    for (const document of vscode.workspace.textDocuments) {
      signal.throwIfAborted();
      if (document.isDirty && isHostFile(document.uri) && this.inside(target.workspaceFolder, document.uri.fsPath)
        && (/\.pyw?$/i.test(document.uri.fsPath) || path.basename(document.uri.fsPath) === 'tensorviz.toml')) {
        if (!await document.save()) return undefined;
      }
    }
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, target.file)));
    if (document.isDirty) return undefined;
    return { ...target, sourceDigest: sha256Text(document.getText()), documentVersion: document.version };
  }
}

/** Profile names are preferences; execution values remain in the project file. */
export function graphProfilePreferenceKey(root: string, target: { file: string; cls: string }): string {
  return `graphProfile.${createHash('sha256').update(JSON.stringify([root, target.file, target.cls])).digest('hex')}`;
}

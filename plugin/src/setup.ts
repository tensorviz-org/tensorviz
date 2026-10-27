import type { GraphSetupPreparation } from '@tensorviz/graph-host/graphOpening';
import { isGraphSetupDiscovery, isGraphSetupDraft, type GraphSetupDraft, type LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { PreparedGraphSetup } from '@tensorviz/graph-host/graphOpening';
import type { MultifileProjectionBody } from '@tensorviz/graph-host/workspaceSourceCore';
import { translateProfile } from '@tensorviz/graph-host/validationProfileBridge';
import type { RunSpec } from '@tensorviz/contracts/protocol';
import type { TraceRequest, TraceOutcome } from '@tensorviz/graph-host/traceWorkerCore';
import path from 'node:path';
import { ApprovalGate } from './approval';
import { readSetup, sameSetup, saveSetup, type SetupSnapshot } from './setupFiles';
import { digest, HostError, savedFile } from './workspace';

interface Snapshot {
  target: LocalBrowserTargetIdentity; sources: MultifileProjectionBody; config: SetupSnapshot;
  runtimeDraft?: GraphSetupDraft; configDefaults?: boolean;
}
export interface SetupOptions {
  current(target: LocalBrowserTargetIdentity): boolean;
  sources(target: LocalBrowserTargetIdentity): Promise<MultifileProjectionBody>;
  post(route: string, body: unknown, signal: AbortSignal): Promise<Record<string, any>>;
  interpreter(): string;
  approvals: ApprovalGate;
  remember(profile: string): Promise<void>;
  prepare(request: TraceRequest, signal: AbortSignal): Promise<TraceOutcome>;
}
const identity = (sources: MultifileProjectionBody) => JSON.stringify([sources.root, sources.truncated, sources.sources.map(entry => [entry.file, entry.digest])]);

/** Shares deterministic source discovery/profile preparation with the VS Code host. */
export class StandaloneSetup {
  private snapshot?: Snapshot;
  constructor(private readonly options: SetupOptions) {}

  private check(target: LocalBrowserTargetIdentity, signal: AbortSignal): void {
    signal.throwIfAborted();
    if (!this.options.current(target)) throw new HostError('SOURCE_CHANGED', 'The saved model changed. Reopen setup.');
  }

  async discover(target: LocalBrowserTargetIdentity, signal: AbortSignal, saved?: GraphSetupDraft): Promise<GraphSetupPreparation> {
    this.check(target, signal);
    const sources = await this.options.sources(target);
    const inputs = saved?.profile.inputs as { factory?: { file?: string } } | undefined;
    for (const helper of [(saved?.profile.factory as { file?: string } | undefined)?.file, inputs?.factory?.file]) {
      if (helper && !sources.sources.some(entry => entry.file === helper)) {
        const file = await savedFile(target.workspaceFolder, helper);
        sources.sources.push({ file: file.file, source: file.text, digest: file.digest });
      }
    }
    const config = await readSetup(target.workspaceFolder);
    const { restoredDraft, ...discovery } = await this.options.post('/api/ide/graph-setup/discover', {
      ...sources, ...(saved ? { profile: saved.profile } : {}),
    }, signal);
    if (!isGraphSetupDiscovery(discovery) || discovery.sourceDigest !== target.sourceDigest
      || (restoredDraft != null && !isGraphSetupDraft(restoredDraft))) throw new HostError('SETUP_UNAVAILABLE', 'Setup discovery did not match the current source.');
    this.check(target, signal);
    this.snapshot = { target, sources, config,
      ...(discovery.resolution?.runtimeInputs ? { runtimeDraft: discovery.resolution.draft,
        configDefaults: discovery.init === null && !discovery.resolution.draft.profile.factory } : {}) };
    return { discovery, ...(discovery.resolution ? { draft: discovery.resolution.draft } : {}),
      ...(restoredDraft ? { restoredDraft: { ...restoredDraft, ...(saved?.profileName ? { profileName: saved.profileName } : {}) } } : {}) };
  }

  private async unchanged(snapshot: Snapshot, signal: AbortSignal): Promise<boolean> {
    this.check(snapshot.target, signal);
    const fresh = await this.options.sources(snapshot.target);
    // Generated helpers are included explicitly in saved-profile discovery.
    for (const entry of snapshot.sources.sources) {
      if (!fresh.sources.some(item => item.file === entry.file)) {
        const file = await savedFile(snapshot.target.workspaceFolder, entry.file);
        fresh.sources.push({ file: file.file, source: file.text, digest: file.digest });
      }
    }
    return identity(fresh) === identity(snapshot.sources)
      && sameSetup(snapshot.config, await readSetup(snapshot.target.workspaceFolder));
  }

  async save(target: LocalBrowserTargetIdentity, draft: GraphSetupDraft, signal: AbortSignal): Promise<{ requirements?: string[] }> {
    this.check(target, signal);
    const snapshot = this.snapshot;
    if (!snapshot || JSON.stringify(snapshot.target) !== JSON.stringify(target) || !await this.unchanged(snapshot, signal)) {
      throw new HostError('SETUP_CHANGED', 'Source or configuration changed. Reopen setup before saving.');
    }
    const prepared = await this.options.post('/api/ide/graph-setup/profile', { sources: snapshot.sources,
      config: snapshot.config.text, profile: draft.profile, ...(draft.profileName ? { profile_name: draft.profileName } : {}),
      ...(draft.configObject ? { config_object: draft.configObject } : {}), ...(draft.configDefaults ? { config_defaults: true } : {}) }, signal);
    if (prepared.ok === false && Array.isArray(prepared.requirements)) return { requirements: prepared.requirements.slice(0, 64).map((item: any) =>
      `${Array.isArray(item.field) ? item.field.join('.') : 'setup'}: ${typeof item.reason === 'string' ? item.reason : 'Complete this field.'}`.slice(0, 1024)) };
    if (prepared.ok !== true || typeof prepared.profile !== 'string' || !/^[A-Za-z_]\w{0,63}$/.test(prepared.profile)
      || typeof prepared.config !== 'string' || Buffer.byteLength(prepared.config) > 262144
      || prepared.factory !== null && (!prepared.factory || typeof prepared.factory.file !== 'string'
        || typeof prepared.factory.source !== 'string' || Buffer.byteLength(prepared.factory.source) > 131072)) {
      throw new HostError('INVALID_SETUP', 'The prepared setup files are invalid.');
    }
    await this.options.approvals.approve({ operation: 'save_setup', workspaceRoot: target.workspaceFolder,
      interpreter: this.options.interpreter(), detail: 'Save this selected execution profile and optional generated helper, then capture under your TensorViz permission. Other profiles and comments are preserved.',
      facts: { sourceDigest: target.sourceDigest, beforeDigest: digest(snapshot.config.text), profile: prepared.profile,
        files: [{ file: 'tensorviz.toml', content: prepared.config }, ...(prepared.factory ? [prepared.factory] : [])] } }, signal);
    await saveSetup(target.workspaceFolder, snapshot.config, prepared as unknown as PreparedGraphSetup,
      signal, () => this.unchanged(snapshot, signal));
    await this.options.remember(prepared.profile);
    this.snapshot = undefined;
    return {};
  }

  async inferInputs(target: LocalBrowserTargetIdentity, draft: GraphSetupDraft, signal: AbortSignal): Promise<{ draft?: GraphSetupDraft; detail: string }> {
    const snapshot = this.snapshot;
    this.check(target, signal);
    const unavailable = { detail: 'A deterministic input example is unavailable. Complete the input fields from repository evidence.' };
    if (!snapshot?.runtimeDraft || draft.configObject || JSON.stringify(snapshot.runtimeDraft) !== JSON.stringify(draft)) return unavailable;
    const translated = translateProfile(draft.profile as unknown as RunSpec['profile'], 'example');
    if (!translated.ok || !await this.unchanged(snapshot, signal)) return unavailable;
    await this.options.approvals.approve({ operation: 'prepare_inputs', workspaceRoot: target.workspaceFolder,
      interpreter: this.options.interpreter(), detail: 'Import and construct this exact model in an isolated worker to inspect declared input dimensions. No forward pass runs. Project construction can execute code; this is not an OS sandbox.',
      facts: { sourceDigest: target.sourceDigest, sources: identity(snapshot.sources), profile: draft.profile, timeoutMs: 30000 } }, signal);
    if (!await this.unchanged(snapshot, signal)) throw new HostError('SOURCE_CHANGED', 'Setup changed during input-preparation approval.');
    const result = await this.options.prepare({ file: path.join(target.workspaceFolder, target.file), cls: target.cls,
      workspace: target.workspaceFolder, execution: translated.run.execution, prepareInputs: true,
      ...(snapshot.configDefaults ? { prepareConfigDefaults: true } : {}), timeoutMs: 30000,
      beforeLaunch: () => this.unchanged(snapshot, signal) }, signal);
    if (!result.ok || !result.executionCurrent || !await result.recheckExecution?.(signal) || !await this.unchanged(snapshot, signal)) return unavailable;
    const proposal = (result.envelope.result as { graphExample?: { binding?: unknown; detail?: unknown; configDefaults?: unknown } } | null)?.graphExample;
    if (proposal?.configDefaults !== undefined && (proposal.configDefaults !== true || !snapshot.configDefaults)) return unavailable;
    const inferred = { ...draft, ...(proposal?.configDefaults ? { configDefaults: true as const } : {}), profile: { ...draft.profile, inputs: proposal?.binding } };
    return proposal?.binding && typeof proposal.detail === 'string' && proposal.detail.length <= 512 && isGraphSetupDraft(inferred)
      ? { draft: inferred, detail: proposal.detail } : unavailable;
  }
}

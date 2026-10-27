import { matchesHostFile } from './workspaceFiles';
/** On-demand checks never save a document or execute the user's model. */
import * as path from 'node:path';
import * as vscode from 'vscode';
import { AppliedSourceReceipts, type AppliedSourceReceipt } from './appliedSourceReceipts';
import { reconcileAppliedCapture } from './sourceReconciliation';
import { reviewResourceGuard } from './sourceReviewResources';
import { sha256 } from './sourceChanges';
import type { LocalBrowserTargetIdentity, AppliedSourceObservation } from '@tensorviz/contracts/protocol';
export type { AppliedSourceObservation } from '@tensorviz/contracts/protocol';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { ResolvedExecutionPlan } from './executionPlan';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';

export interface AppliedCapture {
  capture: CaptureSnapshot;
  plan?: ResolvedExecutionPlan;
  startedAt: number;
}
export interface AppliedEditorState {
  source: AppliedSourceObservation['source'];
  current(): boolean;
}
interface Options {
  origin(): string | undefined;
  token(): string | undefined;
  trusted(): boolean;
  live(target: LocalBrowserTargetIdentity): AppliedCapture | undefined;
  checkSetup?(target: LocalBrowserTargetIdentity, live: AppliedCapture, signal: AbortSignal): Promise<boolean | undefined>;
  readSource?: typeof readAppliedSource;
  captureGuard?: typeof appliedCaptureGuard;
  reconcile?: typeof reconcileAppliedCapture;
  fetchImpl?: typeof fetch;
}

/** Compare the current buffer, and only call it saved when disk bytes agree.
 * Reopening is intentional: a closed TextDocument no longer follows the file. */
export async function readAppliedSource(target: LocalBrowserTargetIdentity, receipt: AppliedSourceReceipt,
  signal: AbortSignal): Promise<AppliedEditorState> {
  signal.throwIfAborted();
  const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, receipt.file)));
  signal.throwIfAborted();
  const text = document.getText(), version = document.version, dirty = document.isDirty, digest = sha256(text);
  const source = digest === receipt.beforeDigest ? 'reverted' : digest !== receipt.afterDigest ? 'changed'
    : dirty ? 'reviewed_unsaved' : 'reviewed_saved';
  const disk = source === 'reviewed_saved' ? await reviewResourceGuard(target.workspaceFolder,
    [{ path: receipt.file, digest }], signal, absolute => vscode.workspace.textDocuments.some(doc => matchesHostFile(doc.uri, absolute) && doc.isDirty)) : () => true;
  const current = () => !signal.aborted && !document.isClosed && document.version === version
    && document.isDirty === dirty && sha256(document.getText()) === digest && disk();
  return { source, current };
}

/** Pin every tracked dependency, including binary resources, across async checks. */
export function appliedCaptureGuard(target: LocalBrowserTargetIdentity, live: AppliedCapture, signal: AbortSignal): Promise<() => boolean> {
  return reviewResourceGuard(target.workspaceFolder, live.capture.source.files, signal,
    absolute => vscode.workspace.textDocuments.some(doc => matchesHostFile(doc.uri, absolute) && doc.isDirty));
}

/** No cached current verdict: each explicit read checks again. Only the active
 * read can publish checked evidence; revocation and newer reads cancel old work. */
export class AppliedSourceInspector {
  private active?: { requestId: string; controller: AbortController };
  constructor(private readonly receipts: AppliedSourceReceipts, private readonly options: Options) {}

  cancel(requestId?: string): void {
    if (!requestId || this.active?.requestId === requestId) this.active?.controller.abort();
  }

  async inspect(requestId: string, signalFromCaller?: AbortSignal): Promise<AppliedSourceObservation> {
    if (signalFromCaller?.aborted) throw new GraphToolError('CANCELLED', 'Applied-source inspection was cancelled.');
    this.cancel();
    const entry = this.receipts.read(requestId);
    if (!entry) throw new GraphToolError('APPLIED_SOURCE_UNAVAILABLE', 'This proposal has no retained confirmed Apply receipt.');
    const controller = new AbortController(), operation = { requestId, controller };
    this.active = operation;
    const { signal } = controller;
    const abortFromCaller = () => controller.abort();
    signalFromCaller?.addEventListener('abort', abortFromCaller, { once: true });
    let rejectAbort!: (error: Error) => void;
    const cancelled = new Promise<never>((_resolve, reject) => { rejectAbort = reject; });
    const abort = () => rejectAbort(new Error('cancelled'));
    signal.addEventListener('abort', abort, { once: true });
    const wait = <T>(work: Promise<T>) => Promise.race([work, cancelled]);
    const timer = setTimeout(() => controller.abort(), 90_000);
    const result: AppliedSourceObservation = { receipt: entry.receipt, checkedAt: Date.now(), source: 'unavailable',
      captureFreshness: 'unavailable', reconciliation: null, unavailable: 'source_unavailable' };
    const origin = this.options.origin(), token = this.options.token();
    const owned = () => !signal.aborted && this.active === operation && this.options.trusted()
      && this.receipts.current(requestId, entry.reservation)
      && this.options.origin() === origin && this.options.token() === token;
    let editor: AppliedEditorState | undefined;
    const stale = () => {
      if (editor && !editor.current()) result.source = 'unavailable';
      result.captureFreshness = 'stale'; result.reconciliation = null; result.unavailable = 'check_stale';
    };
    try {
      if (!owned()) throw new Error('unavailable');
      editor = await wait((this.options.readSource ?? readAppliedSource)(entry.target, entry.receipt, signal));
      result.source = editor.source;
      if (!owned() || !editor.current()) { stale(); return result; }
      if (editor.source !== 'reviewed_saved') {
        result.unavailable = editor.source === 'reviewed_unsaved' ? 'source_not_saved'
          : editor.source === 'unavailable' ? 'source_unavailable' : 'source_changed';
        result.captureFreshness = 'stale'; return result;
      }
      if (!entry.evidence) { result.unavailable = 'evidence_unavailable'; return result; }
      const live = this.options.live(entry.target);
      if (!live || live.startedAt < entry.receipt.appliedAt || live.capture.id === entry.evidence.base.id) {
        result.unavailable = 'capture_unavailable'; return result;
      }
      // Freeze the chosen run, then reselect its identity after every async boundary.
      const selected = structuredClone(live);
      const selectedCurrent = () => {
        const now = this.options.live(entry.target);
        return owned() && editor!.current() && now?.capture.id === selected.capture.id
          && now.capture.source.id === selected.capture.source.id && now.startedAt === selected.startedAt;
      };
      const resources = await wait((this.options.captureGuard ?? appliedCaptureGuard)(entry.target, selected, signal));
      const current = () => selectedCurrent() && resources();
      if (!current()) { stale(); return result; }
      const before = await wait(this.options.checkSetup?.(entry.target, selected, signal) ?? Promise.resolve(undefined));
      if (!current() || before === false) { stale(); return result; }
      if (!origin || !token) { result.unavailable = 'check_failed'; return result; }
      const reconciliation = await wait((this.options.reconcile ?? reconcileAppliedCapture)(entry.evidence, selected.capture,
        { origin, token, signal, fetchImpl: this.options.fetchImpl }));
      if (!current()) { stale(); return result; }
      const after = await wait(this.options.checkSetup?.(entry.target, selected, signal) ?? Promise.resolve(undefined));
      if (!current() || after === false || after !== before) { stale(); return result; }
      result.captureFreshness = before === true && after === true ? 'current' : 'unavailable';
      result.reconciliation = reconciliation; result.unavailable = null;
      return result;
    } catch {
      // No arbitrary filesystem/model/backend errors enter transport summaries.
      if (!owned() || editor && !editor.current()) stale();
      else result.unavailable = editor ? 'check_failed' : 'source_unavailable';
      return result;
    } finally {
      clearTimeout(timer); signal.removeEventListener('abort', abort);
      signalFromCaller?.removeEventListener('abort', abortFromCaller); result.checkedAt = Date.now();
      if (this.active === operation) this.active = undefined;
    }
  }
}

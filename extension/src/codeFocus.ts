import * as vscode from 'vscode';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { CodeFocusOptions, DEFAULT_CODE_FOCUS, isCodeFocusOptions,
  LocalBrowserTargetIdentity, LocalBrowserViewSelectionMessage } from '@tensorviz/contracts/protocol';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import type { LocalBrowserSync } from './localBrowserSync';

type Lines = [number, number];
interface FocusSpan {
  file: string;
  file_digest: string;
  focus: { class: Lines[]; function: Lines[]; node: Lines[]; parameter?: null | {
    declaration: Lines[]; uses: Lines[]; declarationFunctions: Lines[]; useFunctions: Lines[]; limitation: string;
  } };
}
interface Plan { digest: string; ranges: Lines[]; highlights: Lines[] }
type Selection = Pick<LocalBrowserViewSelectionMessage, 'viewId' | 'target' | 'nodeIds' | 'edgeIds' | 'tooLarge'>;

function validLines(value: unknown, count: number): value is Lines[] {
  return Array.isArray(value) && value.length <= 512 && value.every(range => Array.isArray(range) && range.length === 2
    && range.every(Number.isSafeInteger) && range[0] >= 1 && range[1] >= range[0] && range[1] <= count);
}

/** Merge overlaps before taking the complement so multi-selection cannot dim
 * another selected node, and an empty/unmapped selection never dims everything. */
export function mergeFocusLines(ranges: Lines[]): Lines[] {
  const merged: Lines[] = [];
  for (const range of [...ranges].sort((a, b) => a[0] - b[0])) {
    const last = merged[merged.length - 1];
    if (last && range[0] <= last[1] + 1) last[1] = Math.max(last[1], range[1]);
    else merged.push([...range]);
  }
  return merged;
}

export class CodeFocus implements vscode.Disposable {
  private settings = this.readOptions();
  private selection?: Selection;
  private generation = 0;
  private plans = new Map<string, Plan>();
  private decorated = new Set<vscode.TextEditor>();
  private dim?: vscode.TextEditorDecorationType;
  private readonly highlight = vscode.window.createTextEditorDecorationType({ isWholeLine: true,
    backgroundColor: new vscode.ThemeColor('editor.rangeHighlightBackground'),
    borderColor: new vscode.ThemeColor('focusBorder'), borderWidth: '0 0 0 2px', borderStyle: 'solid' });
  private readonly subscriptions: vscode.Disposable[];
  private saving = Promise.resolve();
  private writes = 0;
  private pending?: { selection: Selection; reveal: boolean | 'always' };

  constructor(private readonly context: (target: LocalBrowserTargetIdentity) => ReturnType<LocalBrowserSync['focusContext']>,
    private readonly publish: (message: LocalBrowserHostPublish) => unknown) {
    this.subscriptions = [
      vscode.window.onDidChangeVisibleTextEditors(() => this.paint()),
      vscode.workspace.onDidChangeTextDocument(event => {
        if (event.contentChanges?.length !== 0 && this.plans.has(event.document.uri.fsPath)) this.clear(undefined, 'Source changed. Waiting for a current graph.');
      }),
      vscode.workspace.onDidChangeConfiguration(event => {
        if (this.writes || !event.affectsConfiguration('tensorviz.codeFocus')) return;
        this.settings = this.readOptions();
        if (this.selection) void this.select(this.selection, false);
      }),
    ];
  }

  private readOptions(): CodeFocusOptions {
    const options = vscode.workspace.getConfiguration('tensorviz').get<unknown>('codeFocus');
    return isCodeFocusOptions(options) ? { ...options } : { ...DEFAULT_CODE_FOCUS };
  }

  async configure(selection: Selection, options: CodeFocusOptions): Promise<void> {
    this.settings = { ...options };
    this.writes++;
    // Serialize settings writes so quick scope changes persist in click order.
    this.saving = this.saving.then(() => vscode.workspace.getConfiguration('tensorviz')
      .update('codeFocus', options, vscode.ConfigurationTarget.Global)).catch(() => {
        void vscode.window.showWarningMessage('Code focus works for this session, but its preferences could not be saved.');
      }).finally(() => { this.writes--; });
    await this.select(selection, 'always');
  }

  private report(selection: Selection, detail: string, parameter = false): void {
    this.publish({ type: 'codeFocusState', viewId: selection.viewId, target: selection.target,
      options: { ...this.settings }, detail, parameter });
  }

  /** Reapply options that arrived while a projection was still in flight. */
  retry(): Promise<void> {
    const pending = this.pending;
    if (!pending || !this.context(pending.selection.target)) return Promise.resolve();
    return this.select(pending.selection, pending.reveal);
  }

  async select(selection: Selection, reveal: boolean | 'always' = true): Promise<void> {
    const previous = this.selection;
    const current = this.context(selection.target);
    if (!current) {
      this.pending = { selection, reveal };
      this.report(selection, 'Current source context is unavailable for this selection.');
      return;
    }
    this.pending = undefined;
    reveal = reveal === 'always' || (reveal && (!previous || previous.viewId !== selection.viewId
      || JSON.stringify(previous.target) !== JSON.stringify(selection.target)
      || previous.nodeIds.join('\0') !== selection.nodeIds.join('\0')));
    const generation = ++this.generation;
    this.erase();
    this.selection = selection;
    const spans = selection.nodeIds.map(id => current?.projection.spanmap[id] as FocusSpan | undefined);
    const parameter = spans.some(span => span?.focus && 'parameter' in span.focus);
    if (!this.settings.enabled) { this.report(selection, 'Focus code is off.', parameter); return; }
    if (selection.tooLarge || !selection.nodeIds.length || selection.edgeIds.length) {
      this.report(selection, 'Select graph nodes to focus their source.'); return;
    }
    if (!current || spans.some(span => !span?.focus)) {
      this.report(selection, 'Current source context is unavailable for this selection.', parameter); return;
    }
    const plans = new Map<string, Plan>();
    let detail = parameter ? 'Direct parameter references in the declaring class; runtime aliases are not resolved.' : 'Selected source context.';
    try {
      for (const span of spans as FocusSpan[]) {
        if (span.file.includes('\\') || path.isAbsolute(span.file) || span.file.split('/').some(part => !part || part === '.' || part === '..')
          || current.dependencies.get(span.file) !== span.file_digest) throw new Error('unverified');
        const file = path.join(selection.target.workspaceFolder, span.file);
        const document = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
        if (createHash('sha256').update(document.getText()).digest('hex') !== span.file_digest) throw new Error('stale');
        const focus = span.focus, state = focus.parameter;
        let highlights = focus.node, functions = focus.function;
        if (!('parameter' in focus) && this.settings.references !== 'both') {
          detail = 'This selection is not a parameter, so reference options stay on the selected expression.';
        }
        if ('parameter' in focus) {
          if (!state) throw new Error('unverified');
          highlights = [...(this.settings.references !== 'uses' ? state.declaration : []),
            ...(this.settings.references !== 'declaration' ? state.uses : [])];
          functions = [...(this.settings.references !== 'uses' ? state.declarationFunctions : []),
            ...(this.settings.references !== 'declaration' ? state.useFunctions : [])];
          detail = state.limitation;
        }
        const ranges = this.settings.scope === 'class' ? focus.class : this.settings.scope === 'function' ? functions : highlights;
        if (!ranges.length || !validLines(ranges, document.lineCount) || !validLines(highlights, document.lineCount)) throw new Error('unverified');
        const plan = plans.get(file) ?? { digest: span.file_digest, ranges: [], highlights: [] };
        plan.ranges.push(...ranges); plan.highlights.push(...highlights); plans.set(file, plan);
      }
      if (generation !== this.generation || !this.context(selection.target)) return;
      // Reveal one file for a single selection without stealing keyboard focus
      // from Canvas. Multi-selection decorates its already visible source files.
      if (reveal && spans.length === 1) {
        const [file, plan] = [...plans][0];
        const document = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
        if (generation !== this.generation) return;
        const editor = await vscode.window.showTextDocument(document, {
          viewColumn: vscode.ViewColumn.One, preserveFocus: true, preview: false });
        if (generation !== this.generation) return;
        const first = plan.highlights[0] ?? plan.ranges[0];
        editor.revealRange(new vscode.Range(first[0] - 1, 0, first[1] - 1, 0), vscode.TextEditorRevealType.InCenterIfOutsideViewport);
      }
      if (generation !== this.generation || !this.context(selection.target)) return;
      this.plans = plans;
      this.dim?.dispose();
      this.dim = vscode.window.createTextEditorDecorationType({ opacity: String(this.settings.opacity / 100),
        rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed });
      this.paint();
      if (generation === this.generation) this.report(selection, detail, parameter);
    } catch {
      if (generation !== this.generation) return;
      this.erase();
      this.report(selection, 'Source context is unavailable or changed. No code is dimmed.', parameter);
    }
  }

  private paint(): void {
    if (!this.dim || !this.plans.size) return;
    if (!this.selection || !this.context(this.selection.target)) { this.clear(); return; }
    for (const editor of vscode.window.visibleTextEditors) {
      const plan = this.plans.get(editor.document.uri.fsPath);
      if (!plan) continue;
      const text = editor.document.getText();
      if (createHash('sha256').update(text).digest('hex') !== plan.digest) { this.clear(); return; }
      const lines = text.split(/\r?\n/);
      const visible = mergeFocusLines(plan.ranges), dim: vscode.Range[] = [];
      let next = 1;
      const range = ([start, end]: Lines) => new vscode.Range(start - 1, 0, end - 1, lines[end - 1].length);
      for (const [start, end] of visible) {
        if (next < start) dim.push(range([next, start - 1]));
        next = end + 1;
      }
      if (next <= editor.document.lineCount) dim.push(range([next, editor.document.lineCount]));
      editor.setDecorations(this.dim, dim);
      editor.setDecorations(this.highlight, mergeFocusLines(plan.highlights).map(range));
      this.decorated.add(editor);
    }
  }

  private erase(): void {
    for (const editor of this.decorated) {
      if (this.dim) editor.setDecorations(this.dim, []);
      editor.setDecorations(this.highlight, []);
    }
    this.decorated.clear(); this.plans.clear();
  }

  clear(viewId?: string, detail = 'Select a node to focus its source.'): void {
    if (viewId && this.selection?.viewId !== viewId) return;
    this.pending = undefined;
    this.generation++;
    this.erase();
    if (this.selection) this.report(this.selection, detail);
    this.selection = undefined;
  }

  disconnectBrowsers(nativeViewId?: string): void {
    if (!nativeViewId || this.selection?.viewId !== nativeViewId) this.clear();
  }

  dispose(): void {
    this.clear(); this.dim?.dispose(); this.highlight.dispose();
    this.subscriptions.forEach(subscription => subscription.dispose());
  }
}

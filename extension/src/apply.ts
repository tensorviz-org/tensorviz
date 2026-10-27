import * as vscode from 'vscode';

import type { BackendEndpoint } from './projection';

/** One replacement, as `/api/ide/apply` returns it: 1-based lines, 0-based
 *  columns — the spanmap's convention, not vscode.Position's. */
export interface SourceEdit {
  range: { start: [number, number]; end: [number, number] };
  newText: string;
}

/** An operation the source cannot express, and why. */
export interface Refusal {
  op_id: string;
  reason: string;
  message: string;
  span?: { file: string; start: [number, number]; end: [number, number] } | null;
}

export interface ApplyResponse {
  edits: SourceEdit[];
  refused: Refusal[];
  spanmap: Record<string, unknown>;
  source_digest: string;
}

/** What the applier needs to know about the projection the ops were composed
 *  against. Supplied per call by the host source-review controller. */
export interface ApplyTarget {
  uri: vscode.Uri;
  cls: string;
  /** Workspace-relative, forward slashes — what the spanmap's `file` says. */
  rel: string;
  spanmap: Record<string, unknown>;
  /** The digest of the buffer the spans were computed from. */
  sourceDigest: string;
}

export interface ApplyOutcome {
  applied: number;
  refused: Refusal[];
  /** True when the buffer had moved on and the backend refused the whole batch. */
  stale: boolean;
}

const REQUEST_TIMEOUT_MS = 10_000;

/** A refusal reason as a sentence a person can act on.
 *
 * The backend's own message already says what happened and names the thing;
 * this adds the one word that says whose problem it is, because "refused" on
 * its own reads as a failure of the product rather than a property of the code.
 */
const HEADLINE: Record<string, string> = {
  control_flow_region: 'That layer is inside an if/for/while',
  dynamic_construction: 'That layer has no declaration to edit',
  non_literal_value: 'That value is an expression, not a number',
  unsupported_operation: 'That edit has no single-statement form',
  malformed_operation: 'That edit arrived without something it needs',
};

/**
 * Canvas edits, written back into the user's file (task.md M5).
 *
 * The backend computes the patch and this applies it, which is the division the
 * safety rule requires: FastAPI never writes a workspace file (§8.5), and a
 * `WorkspaceEdit` is what puts the change in the undo stack, in the dirty
 * buffer, and in front of the user before it reaches disk.
 *
 * Refusals are shown, never swallowed. A canvas operation that cannot be made
 * as a source edit must say so and reveal the line it is about — the alternative
 * is a canvas that quietly disagrees with the file it claims to be a view of.
 */
export class EditApplier {
  constructor(
    private readonly backend: BackendEndpoint,
    private readonly channel: vscode.OutputChannel,
    /** Injected so the tests can assert what the user was told without a UI. */
    private readonly notify: (level: 'warn' | 'error', message: string) => void = (
      level,
      message,
    ) => {
      void (level === 'error'
        ? vscode.window.showErrorMessage(message)
        : vscode.window.showWarningMessage(message));
    },
    private readonly fetchImpl: typeof fetch = globalThis.fetch,
  ) {}

  /** The edits `ops` imply, without touching the buffer.
   *
   * Separate from `apply` because a Quick Fix hands its `WorkspaceEdit` to VS
   * Code rather than applying it: that is what gives the user the native preview
   * and puts the change in their undo stack as their own edit.
   */
  async plan(target: ApplyTarget, ops: unknown[]): Promise<ApplyResponse | undefined> {
    const origin = this.backend.origin();
    if (!origin || ops.length === 0) {
      return undefined;
    }
    const document = await vscode.workspace.openTextDocument(target.uri);
    return this.request(origin, {
      file: target.rel,
      cls: target.cls,
      source: document.getText(),
      source_digest: target.sourceDigest,
      spanmap: target.spanmap,
      ops,
    });
  }

  /** Ask the backend for the edits `ops` imply, and apply them. */
  async apply(target: ApplyTarget, ops: unknown[]): Promise<ApplyOutcome> {
    let planned: ApplyResponse | undefined;
    try {
      planned = await this.plan(target, ops);
    } catch (error) {
      if (error instanceof StaleSpanmap) {
        // Not the user's mistake and not worth a modal: the buffer changed while
        // they were dragging, and the next projection is already on its way.
        this.channel.appendLine(`[apply] ${target.rel}#${target.cls}: spans were stale`);
        this.notify('warn', 'tensorViz: the file changed while that edit was in flight. Try it again.');
        return { applied: 0, refused: [], stale: true };
      }
      this.channel.appendLine(`[apply] ${target.rel}#${target.cls}: ${String(error)}`);
      this.notify('error', `tensorViz could not apply that edit: ${String(error)}`);
      return { applied: 0, refused: [], stale: false };
    }
    if (!planned) {
      return { applied: 0, refused: [], stale: false };
    }
    const response = planned;

    if (response.edits.length > 0) {
      const edit = new vscode.WorkspaceEdit();
      for (const change of response.edits) {
        edit.replace(target.uri, toRange(change.range), change.newText);
      }
      const ok = await vscode.workspace.applyEdit(edit);
      if (!ok) {
        this.notify('error', 'tensorViz: the editor rejected that edit.');
        return { applied: 0, refused: response.refused, stale: false };
      }
      this.channel.appendLine(
        `[apply] ${target.rel}#${target.cls}: ${response.edits.length} edit(s)`,
      );
    }

    for (const refusal of response.refused) {
      await this.report(target, refusal);
    }
    return { applied: response.edits.length, refused: response.refused, stale: false };
  }

  /** Say why, and put the cursor on the code the refusal is about. */
  private async report(target: ApplyTarget, refusal: Refusal): Promise<void> {
    const headline = HEADLINE[refusal.reason] ?? 'That edit cannot be made here';
    this.channel.appendLine(`[apply] refused (${refusal.reason}): ${refusal.message}`);
    this.notify('warn', `tensorViz: ${headline}. ${refusal.message}`);
    if (!refusal.span) {
      return;
    }
    const range = toRange({ start: refusal.span.start, end: refusal.span.end });
    const editor = await vscode.window.showTextDocument(target.uri, {
      viewColumn: vscode.ViewColumn.One,
      preserveFocus: false,
      preview: false,
    });
    editor.selection = new vscode.Selection(range.start, range.end);
    editor.revealRange(range, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
  }

  private async request(origin: string, body: unknown): Promise<ApplyResponse> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await this.fetchImpl(`${origin}/api/ide/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...this.backend.headers() },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await response.text();
      const parsed = text ? JSON.parse(text) : undefined;
      if (response.status === 409) {
        throw new StaleSpanmap();
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${text.slice(0, 300)}`);
      }
      return parsed as ApplyResponse;
    } finally {
      clearTimeout(timer);
    }
  }
}

/** The spanmap the ops were composed against is not this buffer's. */
export class StaleSpanmap extends Error {
  constructor() {
    super('spanmap_stale');
  }
}

/** A `[line, column]` pair from the backend as a vscode Range.
 *  Spans count lines from 1 (Python's own convention); vscode counts from 0. */
export function toRange(range: { start: [number, number]; end: [number, number] }): vscode.Range {
  return new vscode.Range(
    Math.max(0, range.start[0] - 1),
    Math.max(0, range.start[1]),
    Math.max(0, range.end[0] - 1),
    Math.max(0, range.end[1]),
  );
}

import * as path from 'node:path';

import * as vscode from 'vscode';

import type { ApplyTarget } from './apply';
import type { TraceFailure } from './diagnostics';

/**
 * Python tracebacks in the terminal, turned into places on the graph (task.md P3).
 *
 * The user runs `python train.py`, it raises, and the terminal prints eight
 * frames. Exactly one of them is theirs; the rest are torch's, and they are true
 * and useless — nobody fixes `torch/nn/modules/linear.py`. The work the IDE
 * removes is the work of reading past those frames, opening the one file that is
 * the user's, and then working out which box on the canvas that line is. So a
 * frame in the user's own workspace becomes a link that does all three: it opens
 * the file at the line, selects the node whose span covers it, and files the
 * exception as a `TraceFailure` diagnostic so the failure survives the terminal
 * being cleared — and is there for "Fix with agent" to seed from.
 *
 * **Frames outside the workspace get no link at all**, which is the decision this
 * file exists to hold. Linking them would be easy and would look richer; it would
 * also put a "Show on graph" underline under a file the user cannot edit and a
 * line that has no node, and the second time that happens they stop trusting the
 * underline. A link is a promise that something useful is on the other side.
 *
 * Nothing here touches `vscode` at runtime. The provider is handed a
 * `TracebackHost` — reveal, focus, report — because the effects are the half that
 * needs an extension host and the parsing is the half that needs tests, and
 * splitting them is what lets `tracebackLinks.test.ts` drive a whole traceback
 * through this file in a plain mocha process.
 */

/** One `File "...", line N, in fn` the terminal printed. */
export interface TracebackFrame {
  /** Absolute, as Python prints it. */
  file: string;
  /** 1-based, as Python counts and as the spanmap counts. */
  line: number;
  /** The function name, when the frame carried one (`<module>` counts). */
  fn?: string;
}

/**
 * The exception a traceback ended with, remembered per traceback.
 *
 * A mutable object rather than a field on the provider because of the shape of
 * `TerminalLinkProvider`: VS Code calls `provideTerminalLinks` once per rendered
 * LINE, in order, and the frames arrive *before* the `RuntimeError: ...` line
 * that says what went wrong. The link therefore cannot be built with the
 * exception in it — it has to hold a reference to the state its own traceback is
 * still filling in, and read it at activation, which is always after the whole
 * traceback has been printed.
 *
 * A fresh instance per `Traceback (most recent call last):` is what keeps a link
 * from an older traceback pointing at the exception that traceback actually
 * ended with, rather than at whatever the terminal printed most recently.
 */
export interface TracebackState {
  exception?: { type: string; message: string };
}

/** What activating a link needs from the editor. Injected; see the file docstring. */
export interface TracebackHost {
  /** The open folder, absolute. `undefined` when no folder is open — in which case
   *  nothing is "inside the workspace" and no frame is ever linked. */
  workspaceRoot(): string | undefined;
  /** The class the canvas is currently projecting, or undefined when none is. This
   *  is the only spanmap in the window, so it is the only file whose lines can be
   *  resolved to a node id. */
  target(): ApplyTarget | undefined;
  /** Open the file and put the cursor on the line. */
  reveal(frame: TracebackFrame): void | Promise<void>;
  /** Select the node on the canvas (`focusNode` over the bridge). */
  focusNode(nodeId: string): void;
  /** File the exception against the frame's line, in the trace collection. */
  reportTrace(frame: TracebackFrame, failure: TraceFailure): void;
}

/** A terminal link that knows which frame it came from. `nodeId` is resolved when
 *  the link is BUILT rather than when it is activated, so the tooltip can promise
 *  the graph only when there is a node to show. */
export interface TracebackLink extends vscode.TerminalLink {
  frame: TracebackFrame;
  nodeId?: string;
  state: TracebackState;
}

/** `  File "/ws/models/resid.py", line 41, in forward` — the one line shape every
 *  CPython traceback frame has. `, in <fn>` is optional: a frame printed by a
 *  bare `traceback.print_tb` omits it, and the file and line are the useful part
 *  either way. */
const FRAME = /File "([^"]+)", line (\d+)(?:, in (\S+))?/g;

/** The header. Its only job here is to start a new `TracebackState`. */
const TRACEBACK_HEADER = /^Traceback \(most recent call last\):\s*$/;

/**
 * `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x8 and 4x4)`.
 *
 * Anchored at column zero, because that is where CPython prints the exception and
 * where nothing else in a traceback is, and restricted to names that END in one of
 * the four suffixes Python's own hierarchy uses. Matching any `Word: text` at
 * column zero would swallow half of pytest's output and file it as an exception.
 * Dotted names are allowed for `torch.cuda.OutOfMemoryError` and friends.
 */
const EXCEPTION = /^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))(?::\s?(.*))?\s*$/;

/**
 * The node whose span covers `line` (1-based), innermost first.
 *
 * `file` filters by the spanmap's own workspace-relative path, so a projection
 * that spans two files cannot answer a line number from the wrong one.
 */
export function nodeAtLine(
  spanmap: Record<string, unknown>,
  line: number,
  file?: string,
): string | undefined {
  let best: string | undefined;
  let bestWidth = Number.MAX_SAFE_INTEGER;
  for (const [nodeId, raw] of Object.entries(spanmap)) {
    const span = raw as { file?: unknown; start?: unknown; end?: unknown } | undefined;
    const start = Array.isArray(span?.start) ? Number(span?.start[0]) : NaN;
    const end = Array.isArray(span?.end) ? Number(span?.end[0]) : NaN;
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      continue;
    }
    if (file !== undefined && typeof span?.file === 'string' && span.file !== file) {
      continue;
    }
    if (line < start || line > end) {
      continue;
    }
    const width = end - start;
    if (width < bestWidth) {
      best = nodeId;
      bestWidth = width;
    }
  }
  return best;
}

/**
 * Is this frame's file one the user can edit?
 *
 * Inside the open folder, and not inside an installed package. The second half
 * matters because a virtualenv usually lives *in* the workspace
 * (`.venv/lib/python3.11/site-packages/torch/...`), so "inside the workspace" on
 * its own would link exactly the torch frames this provider exists to stay
 * quiet about.
 */
export function isUserFrame(root: string | undefined, file: string): boolean {
  if (!root) {
    return false;
  }
  const resolved = path.resolve(file);
  const base = path.resolve(root);
  if (resolved !== base && !resolved.startsWith(base + path.sep)) {
    return false;
  }
  const segments = resolved.slice(base.length).split(path.sep);
  return !segments.includes('site-packages') && !segments.includes('dist-packages');
}

/** Every traceback frame one printed line carries, in the order they appear. */
export function framesIn(line: string): Array<TracebackFrame & { startIndex: number; length: number }> {
  const found: Array<TracebackFrame & { startIndex: number; length: number }> = [];
  FRAME.lastIndex = 0;
  for (let match = FRAME.exec(line); match; match = FRAME.exec(line)) {
    found.push({
      file: match[1],
      line: Number(match[2]),
      fn: match[3],
      // The whole `File "...", line N, in fn` is the link: it is what the reader
      // sees as "the location", and a link that covers only the path makes the
      // line number — the half that says WHERE — unclickable.
      startIndex: match.index,
      length: match[0].length,
    });
  }
  return found;
}

/**
 * `File "..." , line N` in the terminal → a link onto the graph.
 *
 * Registered once at activation; see `registerTracebackLinks` below.
 */
export class TracebackLinkProvider implements vscode.TerminalLinkProvider<TracebackLink> {
  /** Per-terminal, because two terminals can be printing two tracebacks at once
   *  and the exception of one is not the exception of the other. Weak so a closed
   *  terminal is not kept alive by its last traceback. */
  private readonly states = new WeakMap<object, TracebackState>();

  constructor(private readonly host: TracebackHost) {}

  provideTerminalLinks(context: vscode.TerminalLinkContext): TracebackLink[] {
    const line = context.line;
    const terminal = (context.terminal ?? this) as object;

    if (TRACEBACK_HEADER.test(line)) {
      // A new traceback: from here the frames belong to a new exception. The old
      // state object stays alive for the links that already reference it.
      this.states.set(terminal, {});
      return [];
    }

    const exception = EXCEPTION.exec(line);
    if (exception) {
      this.stateFor(terminal).exception = {
        type: exception[1],
        message: (exception[2] ?? '').trim(),
      };
      return [];
    }

    const root = this.host.workspaceRoot();
    const target = this.host.target();
    const state = this.stateFor(terminal);
    const links: TracebackLink[] = [];
    for (const frame of framesIn(line)) {
      if (!isUserFrame(root, frame.file)) {
        continue;
      }
      const nodeId =
        target && samePath(target.uri.fsPath, frame.file)
          ? nodeAtLine(target.spanmap, frame.line, target.rel)
          : undefined;
      links.push({
        startIndex: frame.startIndex,
        length: frame.length,
        // The tooltip is the promise, so it says only what this link can keep: a
        // frame with no node opens the file and nothing else, and saying "Show on
        // graph" there would be a lie the user only finds out by clicking.
        tooltip: nodeId ? 'Show on graph' : 'Open this frame',
        frame: { file: frame.file, line: frame.line, fn: frame.fn },
        nodeId,
        state,
      });
    }
    return links;
  }

  async handleTerminalLink(link: TracebackLink): Promise<void> {
    await this.host.reveal(link.frame);
    if (link.nodeId) {
      this.host.focusNode(link.nodeId);
    }
    const exception = link.state.exception;
    if (exception) {
      // The diagnostic is the durable half. The terminal scrolls away and is
      // cleared; a Problem on the line stays until the code changes, and it is
      // what "Fix with agent" reads to seed the composer.
      this.host.reportTrace(link.frame, {
        message: exception.message,
        line: link.frame.line,
        type: exception.type,
      });
    }
  }

  private stateFor(terminal: object): TracebackState {
    let state = this.states.get(terminal);
    if (!state) {
      state = {};
      this.states.set(terminal, state);
    }
    return state;
  }
}

/**
 * Register the provider with the window.
 *
 * A one-line wrapper so that `extension.ts` owns activation and this file owns the
 * behaviour, and so that the unit test can construct the provider without the
 * `vscode.window` this call needs.
 */
export function registerTracebackLinks(host: TracebackHost): vscode.Disposable {
  return vscode.window.registerTerminalLinkProvider(new TracebackLinkProvider(host));
}

/** Path equality that survives the two spellings of the same file a traceback and
 *  the editor can disagree on (a trailing `/./`, a relative frame). Case is left
 *  alone: the fork ships on macOS and Linux, and lowercasing would merge two real
 *  files on the second. */
function samePath(left: string, right: string): boolean {
  return path.resolve(left) === path.resolve(right);
}

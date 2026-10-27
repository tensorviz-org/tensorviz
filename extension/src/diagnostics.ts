import * as vscode from 'vscode';

/** Where a parse failed, as the backend reported it. */
export interface ParseFailure {
  message: string;
  /** 1-based, as Python's SyntaxError counts. */
  line: number;
  /** 0-based column. */
  column: number;
}

/** A failure that came from running the code, not from parsing it. */
export interface TraceFailure {
  message: string;
  /** 1-based, from the worker's own traceback frames. */
  line: number;
  /** What raised it, e.g. `RuntimeError` — prefixed onto the message so the
   *  Problems panel reads like Python's own output. */
  type?: string;
}

/** A class the worker could not construct, and what to do about it. */
export interface ConstructFailure {
  cls: string;
  message: string;
  /** 1-based, like the other two. */
  line: number;
  /** The `# tensorviz: init=` line a Quick Fix would insert. */
  skeleton: string;
  /** Whether the file already carries a declaration for this class. When it does the
   *  fix is to correct that line, not to insert a second one. */
  hasDeclaration: boolean;
}

/** Diagnostic code the Quick Fix provider matches on, so it offers the insertion only
 *  for this failure and not for every tensorViz diagnostic in the file. */
export const CONSTRUCT_FAILED_CODE = 'tensorviz.constructFailed';

/**
 * SyntaxErrors as editor diagnostics (task.md M3).
 *
 * The graph is a projection of a buffer that is, mid-keystroke, usually invalid.
 * Two things must therefore be true at once: the editor says exactly where the
 * parse stopped, and the canvas keeps the last projection that *did* parse
 * rather than blanking. This owns the first half; the current graph controller owns the
 * second, which is why `greyed` is a field on the projection and not a mode
 * here.
 *
 * One collection, keyed by file, so a fixed error clears without bookkeeping.
 */
export class ParseDiagnostics implements vscode.Disposable {
  private readonly collection: vscode.DiagnosticCollection;
  /** A second collection, because the two failures are independent and a shared
   *  one would make them clear each other: a file whose forward raises still
   *  parses, and fixing a SyntaxError must not erase yesterday's trace error. */
  private readonly traces: vscode.DiagnosticCollection;

  /** A third collection: a class that cannot be BUILT is independent of both a parse
   *  error and a forward that raised, and sharing one would make them clear each other. */
  private readonly constructs: vscode.DiagnosticCollection;
  /** What the Quick Fix would insert, per file. */
  private readonly skeletons = new Map<string, ConstructFailure>();

  constructor() {
    this.collection = vscode.languages.createDiagnosticCollection('tensorviz');
    this.traces = vscode.languages.createDiagnosticCollection('tensorviz-trace');
    this.constructs = vscode.languages.createDiagnosticCollection('tensorviz-construct');
  }

  report(uri: vscode.Uri, failure: ParseFailure): void {
    const line = Math.max(0, failure.line - 1);
    const column = Math.max(0, failure.column);
    const diagnostic = new vscode.Diagnostic(
      // To end-of-line: a SyntaxError's own column is where the parser gave up,
      // which is rarely where the mistake looks like it is, and a zero-width
      // squiggle is invisible.
      new vscode.Range(line, column, line, Number.MAX_SAFE_INTEGER),
      failure.message,
      vscode.DiagnosticSeverity.Error,
    );
    diagnostic.source = 'tensorViz';
    this.collection.set(uri, [diagnostic]);
  }

  /** A forward that raised, at the line it raised on (task.md M4).
   *
   * The line is the whole content of the diagnostic: "mat1 and mat2 shapes cannot
   * be multiplied (2x8 and 4x4)" is a perfectly precise message about a place it
   * does not name, and finding that place by hand is the work the IDE exists to
   * remove.
   */
  reportTrace(uri: vscode.Uri, failure: TraceFailure): void {
    const line = Math.max(0, failure.line - 1);
    const diagnostic = new vscode.Diagnostic(
      new vscode.Range(line, 0, line, Number.MAX_SAFE_INTEGER),
      failure.type ? `${failure.type}: ${failure.message}` : failure.message,
      vscode.DiagnosticSeverity.Error,
    );
    diagnostic.source = 'tensorViz trace';
    this.traces.set(uri, [diagnostic]);
  }

  /** A class that could not be built, on its own line (task.md P0).
   *
   * A third collection, for the same reason there are already two: a construction failure
   * and a forward that raised are independent, and a shared collection would make fixing
   * one silently erase the other. It also means the Quick Fix can find its own diagnostic
   * by code without walking everything else in the file.
   */
  reportConstruct(uri: vscode.Uri, failure: ConstructFailure): void {
    const line = Math.max(0, failure.line - 1);
    const diagnostic = new vscode.Diagnostic(
      new vscode.Range(line, 0, line, Number.MAX_SAFE_INTEGER),
      failure.message,
      vscode.DiagnosticSeverity.Warning,
    );
    diagnostic.source = 'tensorViz';
    diagnostic.code = CONSTRUCT_FAILED_CODE;
    this.constructs.set(uri, [diagnostic]);
    this.skeletons.set(uri.toString(), failure);
  }

  /** What a Quick Fix should insert for `uri`, if anything. */
  constructFailure(uri: vscode.Uri): ConstructFailure | undefined {
    return this.skeletons.get(uri.toString());
  }

  clearConstruct(uri: vscode.Uri): void {
    this.constructs.delete(uri);
    this.skeletons.delete(uri.toString());
  }

  clearTrace(uri: vscode.Uri): void {
    this.traces.delete(uri);
  }

  clear(uri: vscode.Uri): void {
    this.collection.delete(uri);
  }

  /** What the file currently shows. The Electron test reads this rather than
   *  screen-scraping the Problems panel; nothing in the product calls it. */
  get(uri: vscode.Uri): readonly vscode.Diagnostic[] {
    return [
      ...(this.collection.get(uri) ?? []),
      ...(this.traces.get(uri) ?? []),
      ...(this.constructs.get(uri) ?? []),
    ];
  }

  dispose(): void {
    this.collection.dispose();
    this.traces.dispose();
    this.constructs.dispose();
  }
}

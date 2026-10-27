/** A stand-in for the `vscode` module, for tests that run outside the extension host.
 *
 * Only the API surface the pure seams actually touch is here. Anything a test has not set up
 * throws rather than returning undefined: a stub that silently answers "nothing" turns a
 * genuine regression (preflight reading a different setting, say) into a green test.
 *
 * Loaded by bootstrap.ts, which points `require('vscode')` at the compiled copy of this file.
 * Tests import it by relative path and get the same module instance, so mutating `state` here
 * is what the code under test sees.
 */

import type { SourceIO } from '../workspaceSources';

export interface StubFolder {
  uri: { fsPath: string };
}

export interface StubExtension {
  activate(): Promise<unknown>;
}

export const state: {
  /** Keyed `"<section>.<key>"`, e.g. `"tensorviz.repoPath"`. */
  configuration: Map<string, unknown>;
  workspaceFolders: StubFolder[] | undefined;
  isTrusted: boolean;
  extensions: Map<string, StubExtension>;
  openDialogResult: { fsPath: string }[] | undefined;
  errorMessages: string[];
  /** Modal warnings the code under test raised, newest last. The trust prompt is
   *  one of these, and "was the user asked exactly once" is the thing to assert. */
  warningMessages: string[];
  /** What `showWarningMessage` answers, shifted one per call. `undefined` (an
   *  exhausted queue) stands for the user dismissing the dialog, which is a
   *  distinct outcome from declining and must stay distinguishable. */
  warningAnswers: Array<string | undefined>;
  /** Every `registerDebugAdapterTrackerFactory` call, in order. The debug type is
   *  the whole point of recording them: a tracker registered for a type no real
   *  session reports is invisible, and that is exactly how a tracker for the wrong
   *  debug type looked installed while no session ever called it. */
  debugTrackerFactories: Array<{ debugType: string; factory: StubTrackerFactory }>;
  /** VS Code SecretStorage as a plain map (task.md §6.5). The provider key lives here and
   *  nowhere else, so a test that finds it in `configuration` or in one of the message lists
   *  above has caught exactly the leak BYOK is supposed to prevent. */
  secrets: Map<string, string>;
  /** What `showInputBox` answers, shifted one per call. `undefined` — including an exhausted
   *  queue — is the user pressing Escape, which must stay distinguishable from an empty key. */
  inputBoxAnswers: Array<string | undefined>;
  /** Every `showInputBox` call's options, in order. `password` is the assertion that matters:
   *  an API key typed in clear text is on screen and in the screen recording. */
  inputBoxOptions: StubInputBoxOptions[];
  /** What `showQuickPick` answers, shifted one per call and matched against each item's
   *  `label`. `undefined` is Escape. */
  quickPickAnswers: Array<string | undefined>;
  /** Everyone listening to `workspace.onDidChangeConfiguration`. `fireConfigurationChange`
   *  is what a test uses to move a setting under code that is already running. */
  configurationListeners: Array<(event: StubConfigurationChange) => void>;
} = {
  configuration: new Map(),
  workspaceFolders: undefined,
  isTrusted: true,
  extensions: new Map(),
  openDialogResult: undefined,
  errorMessages: [],
  warningMessages: [],
  warningAnswers: [],
  debugTrackerFactories: [],
  secrets: new Map(),
  inputBoxAnswers: [],
  inputBoxOptions: [],
  quickPickAnswers: [],
  configurationListeners: [],
};

export function reset(): void {
  infoMessageDetails.length = 0;
  state.configuration = new Map();
  state.workspaceFolders = undefined;
  state.isTrusted = true;
  state.extensions = new Map();
  state.openDialogResult = undefined;
  state.errorMessages = [];
  state.warningMessages = [];
  state.warningAnswers = [];
  state.debugTrackerFactories = [];
  state.secrets = new Map();
  state.inputBoxAnswers = [];
  state.inputBoxOptions = [];
  state.quickPickAnswers = [];
  state.configurationListeners = [];
  infoResponder = undefined;
}

export const Uri = {
  [Symbol.hasInstance](value: unknown): boolean {
    return !!value && typeof value === 'object' && 'scheme' in value && 'fsPath' in value;
  },
  file(fsPath: string): { fsPath: string; toString(): string; scheme: string } {
    return { fsPath, scheme: 'file', toString: () => fsPath };
  },
};

/** A plain text tab, distinct from custom editors and diff inputs. */
export class TabInputText {
  constructor(readonly uri: ReturnType<typeof Uri.file>) {}
}

/** vscode.ConfigurationChangeEvent, narrowed to the one method callers use. */
export interface StubConfigurationChange {
  affectsConfiguration(section: string): boolean;
}

/** Move a setting and tell the listeners, the way VS Code does when the user edits
 *  settings.json. Two steps on purpose: code that caches a setting at registration
 *  time passes a test that only writes `state.configuration`, and fails this. */
export function fireConfigurationChange(key: string, value: unknown): void {
  state.configuration.set(key, value);
  const event: StubConfigurationChange = { affectsConfiguration: (section) => key.startsWith(section) };
  state.configurationListeners.slice().forEach((listener) => listener(event));
}

export const workspace = {
  onDidChangeConfiguration(listener: (event: StubConfigurationChange) => void): { dispose(): void } {
    state.configurationListeners.push(listener);
    return {
      dispose: () => {
        const index = state.configurationListeners.indexOf(listener);
        if (index >= 0) state.configurationListeners.splice(index, 1);
      },
    };
  },
  get textDocuments(): StubTextDocument[] {
    return [...editorState.documents.keys()].map(file => editorState.visibleTextEditors.find(editor => editor.document.uri.fsPath === file)?.document ?? stubDocument(file));
  },
  getConfiguration(section: string): { get<T>(key: string, fallback: T): T; update(key: string, value: unknown, target?: unknown): Promise<void> } {
    return {
      async update(key: string, value: unknown): Promise<void> {
        fireConfigurationChange(`${section}.${key}`, value);
      },
      get<T>(key: string, fallback: T): T {
        const value = state.configuration.get(`${section}.${key}`);
        return value === undefined ? fallback : (value as T);
      },
    };
  },
  get workspaceFolders(): StubFolder[] | undefined {
    return state.workspaceFolders;
  },
  /** VS Code's own Workspace Trust. Since task.md D20 this is the ONLY consent gate
   *  before tensorViz runs a workspace's code; the extension has no prompt of its own. */
  get isTrusted(): boolean {
    return state.isTrusted;
  },
  /** The folder containing a file, or undefined when no folder is open — which is
   *  the case the CodeLens provider has to handle by falling back to the file's
   *  own directory. */
  getWorkspaceFolder(uri: { fsPath: string }): StubFolder | undefined {
    return (state.workspaceFolders ?? []).find((folder) =>
      uri.fsPath.startsWith(`${folder.uri.fsPath}/`),
    );
  },
};

/** Enough of vscode.Range for a CodeLens to carry a position. */
export class Range {
  constructor(
    readonly startLine: number,
    readonly startCharacter: number,
    readonly endLine: number,
    readonly endCharacter: number,
  ) {}
  get start(): Position { return new Position(this.startLine, this.startCharacter); }
  get end(): Position { return new Position(this.endLine, this.endCharacter); }
}

/** vscode.Position, 0-based in both. Navigation (localBrowserSync's
 *  revealNode) constructs one from a spanmap's 1-based lines. */
export class Position {
  constructor(
    readonly line: number,
    readonly character: number,
  ) {}
}

/** vscode.Selection narrowed to the two-anchor form navigation constructs. Its
 *  `active` is structurally the `{ line, character }` a StubTextEditor's
 *  selection field already holds, so assigning one there just works. */
export class Selection {
  constructor(
    readonly anchor: Position,
    readonly active: Position,
  ) {}
}

/** vscode.ViewColumn values used by source navigation and candidate review. */
export enum ViewColumn {
  One = 1,
  Beside = -2,
}

/** vscode.TextEditorRevealType with vscode's own numbering, so a test that
 *  asserts the constant reads true. */
export enum TextEditorRevealType {
  InCenterIfOutsideViewport = 2,
}

export interface StubCommand {
  title: string;
  command: string;
  tooltip?: string;
  arguments?: unknown[];
}

export class CodeLens {
  constructor(readonly range: Range, readonly command?: StubCommand) {}
}

/** vscode.EventEmitter, minus the disposal bookkeeping nothing here exercises. */
export class EventEmitter<T> {
  private readonly listeners: Array<(value: T) => void> = [];

  readonly event = (listener: (value: T) => void): { dispose(): void } => {
    this.listeners.push(listener);
    return {
      dispose: () => {
        const index = this.listeners.indexOf(listener);
        if (index >= 0) this.listeners.splice(index, 1);
      },
    };
  };

  fire(value: T): void {
    this.listeners.slice().forEach((listener) => listener(value));
  }

  dispose(): void {
    this.listeners.length = 0;
  }
}

export const window = {
  tabGroups: { activeTabGroup: { activeTab: undefined as { input: unknown } | undefined } },
  registerWebviewViewProvider(id: string, provider: import('vscode').WebviewViewProvider, options?: unknown): { dispose(): void } {
    if (webviewState.providers.has(id)) throw new Error(`Duplicate webview provider: ${id}`);
    webviewState.providers.set(id, provider);
    webviewState.registrations.push({ id, options });
    return { dispose: () => { webviewState.providers.delete(id); } };
  },
  async showOpenDialog(): Promise<{ fsPath: string }[] | undefined> {
    return state.openDialogResult;
  },
  async showErrorMessage(message: string): Promise<undefined> {
    state.errorMessages.push(message);
    return undefined;
  },
  async showWarningMessage(message: string): Promise<string | undefined> {
    state.warningMessages.push(message);
    return state.warningAnswers.shift();
  },
  /** Records the options rather than only the answer, because for the API-key box the options
   *  ARE the behaviour under test (`password`, and the absence of a pre-filled `value`). */
  async showInputBox(options: StubInputBoxOptions = {}): Promise<string | undefined> {
    state.inputBoxOptions.push(options);
    return state.inputBoxAnswers.shift();
  },
  /** Answers with the item whose `label` the queue names, so a test says which provider the
   *  user clicked rather than which array index it happens to sit at. An unmatched label throws:
   *  silently answering `undefined` would turn "the picker stopped offering OpenRouter" into a
   *  passing "the user cancelled" test. */
  async showQuickPick<T extends { label: string }>(
    items: readonly T[],
    _options?: unknown,
  ): Promise<T | undefined> {
    const label = state.quickPickAnswers.shift();
    if (label === undefined) {
      return undefined;
    }
    const picked = items.find((item) => item.label === label);
    if (!picked) {
      throw new Error(`vscodeStub: no quick pick item labelled ${JSON.stringify(label)}`);
    }
    return picked;
  },
};

export const extensions = {
  getExtension(id: string): StubExtension | undefined {
    return state.extensions.get(id);
  },
};

/** vscode.DebugSession, narrowed to what a tracker factory reads to decide whether
 *  the session is one of ours. */
export interface StubDebugSession {
  type: string;
  name: string;
}

export interface StubTracker {
  onDidSendMessage?(message: unknown): void;
}

export interface StubTrackerFactory {
  createDebugAdapterTracker(session: StubDebugSession): StubTracker | undefined;
}

export const debug = {
  registerDebugAdapterTrackerFactory(
    debugType: string,
    factory: StubTrackerFactory,
  ): { dispose(): void } {
    const entry = { debugType, factory };
    state.debugTrackerFactories.push(entry);
    return {
      dispose: () => {
        const index = state.debugTrackerFactories.indexOf(entry);
        if (index >= 0) state.debugTrackerFactories.splice(index, 1);
      },
    };
  },
};

/**
 * Start a debug session the way the editor does, and return the trackers that
 * were actually given one.
 *
 * **The dispatch rule is the thing under test, so the stub has to model it.** VS
 * Code offers a session only to factories registered for `'*'` or for that
 * session's exact `type`. A test that reaches into a recorded factory and calls
 * `createDebugAdapterTracker` itself skips that step — and skipping it is what
 * made a tracker registered for the wrong type look correct. Drive sessions
 * through here instead.
 */
export function startDebugSession(session: StubDebugSession): StubTracker[] {
  return state.debugTrackerFactories
    .filter((entry) => entry.debugType === '*' || entry.debugType === session.type)
    .map((entry) => entry.factory.createDebugAdapterTracker(session))
    .filter((tracker): tracker is StubTracker => Boolean(tracker));
}

// ---------------------------------------------------------------------------
// Tree views (task.md F1)
// ---------------------------------------------------------------------------
// Appended rather than woven in: everything above is what the preflight/CodeLens seams need,
// and keeping the two groups separate makes a merge between milestones a merge of additions.

export enum TreeItemCollapsibleState {
  None = 0,
  Collapsed = 1,
  Expanded = 2,
}

export class ThemeIcon {
  constructor(readonly id: string) {}
}

/** vscode.MarkdownString. `isTrusted` is the flag that decides whether a `command:` link in a
 *  TreeView message is clickable, so a test can tell a decorative string from a live one. */
export class MarkdownString {
  constructor(readonly value: string, readonly isTrusted: boolean = false) {}
}

export class TreeItem {
  description?: string;
  tooltip?: string;
  iconPath?: unknown;
  contextValue?: string;
  command?: StubCommand;

  constructor(
    readonly label: string,
    readonly collapsibleState: TreeItemCollapsibleState = TreeItemCollapsibleState.None,
  ) {}
}

// ---------------------------------------------------------------------------
// Secrets and prompts (task.md P1, §6.5)
// ---------------------------------------------------------------------------
// Appended for the same reason the tree-view block above was: a milestone's additions stay a
// block, so two milestones merge as two additions rather than as one contested object literal.

/** vscode.InputBoxOptions, narrowed to the fields the provider dialog sets. `validateInput` is
 *  here so a test can run the real validator the command wired up, instead of a copy of it. */
export interface StubInputBoxOptions {
  title?: string;
  prompt?: string;
  placeHolder?: string;
  value?: string;
  password?: boolean;
  ignoreFocusOut?: boolean;
  validateInput?(value: string): unknown;
}

/** vscode.SecretStorage over `state.secrets`.
 *
 *  Deliberately not encrypted, not namespaced and not persisted: the point of the stub is that a
 *  test can look inside and assert WHICH keys were written, and that nothing else on the stub —
 *  `configuration`, `errorMessages`, `warningMessages` — ever contains the value. `onDidChange`
 *  is absent because nothing in the extension subscribes to it; add it when something does. */
export const secrets = {
  async get(key: string): Promise<string | undefined> {
    return state.secrets.get(key);
  },
  async store(key: string, value: string): Promise<void> {
    state.secrets.set(key, value);
  },
  async delete(key: string): Promise<void> {
    state.secrets.delete(key);
  },
};

// ---------------------------------------------------------------------------
// Documents, workspace edits and decorations (task.md P3)
// ---------------------------------------------------------------------------
// Appended as its own state object with its own reset, rather than as new fields on `state`
// above. P3 is built against this one file by several people at once, and an addition that
// touches no existing line merges; a new field on `state` touches three (the type, the literal
// and `reset`). Tests that need this half call `resetEditors()` as well as `reset()`.

/** vscode.TextDocument, narrowed to what the commit host and the decorations read. */
export interface StubTextDocument {
  uri: { fsPath: string; scheme: string; toString(): string };
  getText(): string;
  offsetAt(position: { line: number; character: number }): number;
  lineCount: number;
  /** Mirrors vscode's monotonically-increasing document version: bumped once per
   *  openStubEditor call, which is how tests model the user typing. There is no
   *  workspaceEdit path here, so no other bump exists. */
  readonly version: number;
  /** True while the buffer differs from disk; the validation host refuses to run then. */
  readonly isDirty: boolean;
}

/** vscode.TextEditor, narrowed the same way. Records what was drawn on it rather than
 *  drawing anything: "which ranges got the agent's mark" is the behaviour under test. */
export interface StubTextEditor {
  document: StubTextDocument;
  /** Enough of vscode.Selection for a cursor-sensitive caller: the 0-based active
   *  line. A test puts the caret somewhere by assigning a fresh object. */
  selection: { active: { line: number; character: number } };
  setDecorations(type: StubDecorationType, ranges: unknown[]): void;
  /** Camera moves are drawn nowhere; the observable truth is the selection. */
  revealRange(range: unknown, how: unknown): void;
}

/** The handle `createTextEditorDecorationType` hands back. Keeps its options so a test can
 *  assert the mark is a gutter icon and a background and not, say, an empty object. */
export class StubDecorationType {
  disposed = false;

  constructor(readonly options: Record<string, unknown>) {}

  dispose(): void {
    this.disposed = true;
  }
}

/** One `setDecorations` call: which editor, which type, which ranges. An empty `ranges`
 *  is how VS Code is told to CLEAR a decoration, so it is a real event, not a no-op. */
export interface StubDecorationCall {
  fsPath: string;
  type: StubDecorationType;
  ranges: unknown[];
}

/** vscode.WorkspaceEdit, keeping every replacement so a test can ask what the host asked the
 *  editor to do. The count of edits that reached `applyEdit` is the "one undo" assertion. */
export class WorkspaceEdit {
  readonly replacements: Array<{
    uri: { fsPath: string };
    range: Range;
    newText: string;
  }> = [];

  replace(uri: { fsPath: string }, range: Range, newText: string): void {
    this.replacements.push({ uri, range, newText });
  }

  get size(): number {
    return this.replacements.length;
  }
}

/** vscode.CodeActionKind. The providers in `codeActions.ts` name `QuickFix` in a static
 *  initialiser, so the module cannot even be imported without it. Nothing compares kinds,
 *  so the shape only has to be an object with the `value` the real one carries. */
export const CodeActionKind = {
  QuickFix: { value: 'quickfix' },
};

export const languages = {
  getDiagnostics(uri: { fsPath: string }): import('vscode').Diagnostic[] {
    return editorState.diagnostics.get(uri.fsPath) ?? [];
  },
};

export const editorState: {
  diagnostics: Map<string, import('vscode').Diagnostic[]>;
  /** Buffer text keyed by fsPath. `openTextDocument` throws for anything absent, because a
   *  path the test did not set up is a path the code under test computed wrongly. */
  documents: Map<string, string>;
  /** Document versions keyed by fsPath, paralleling `documents`; openStubEditor bumps. */
  versions: Map<string, number>;
  /** Every edit that reached `applyEdit`, in order. */
  appliedEdits: WorkspaceEdit[];
  /** Paths whose buffer is unsaved, so a test can model "the user has not saved yet" —
   *  which the validation host must refuse to run against. */
  dirtyDocuments: Set<string>;
  /** Editors the code under test can see. */
  visibleTextEditors: StubTextEditor[];
  /** Every decoration type created, in creation order. */
  decorationTypes: StubDecorationType[];
  /** Every `setDecorations` call, in order. */
  decorationCalls: StubDecorationCall[];
} = {
  dirtyDocuments: new Set<string>(),
  diagnostics: new Map(),
  documents: new Map(),
  versions: new Map(),
  appliedEdits: [],
  visibleTextEditors: [],
  decorationTypes: [],
  decorationCalls: [],
};

/** Informational notices (`showInformationMessage`), newest last. Kept in its own array for
 *  the merge reason above; cleared by `resetEditors`. */
export const infoMessages: string[] = [];
export const infoMessageDetails: string[] = [];
/** `undefined` accepts the dialog's default path. `null` cancels the dialog. */
export let saveDialogChoice: string | undefined | null = undefined;
export function setSaveDialogChoice(choice: string | undefined | null): void {
  saveDialogChoice = choice;
}

/** What `showInformationMessage` answers. The default — and the only safe product default —
 *  is undefined (dismissed): consent UIs in tests must never silently say yes, they must be
 *  told to. Cleared by `reset` and `resetEditors`, so a responder never leaks across tests. */
let infoResponder: ((message: string, items: string[]) => string | undefined) | undefined;

export function setInfoResponder(
  responder: ((message: string, items: string[]) => string | undefined) | undefined,
): void {
  infoResponder = responder;
}

/** Fires the way `vscode.workspace.onDidChangeTextDocument` does. A test types by rewriting
 *  `editorState.documents` and firing this, which is what the editor does in that order. */
export enum TextDocumentChangeReason { Undo = 1, Redo = 2 }
export const documentChanges = new EventEmitter<{ document: StubTextDocument; reason?: TextDocumentChangeReason }>();

/** Fires the way `vscode.workspace.onDidSaveTextDocument` does. The stub keeps no separate
 *  "saved" copy — the buffer IS the file for everything written through this harness — so
 *  a save is the event alone. */
export const saveChanges = new EventEmitter<StubTextDocument>();

/** Fires the way `vscode.workspace.onDidCloseTextDocument` does. */
export const configFileChanges = new EventEmitter<{ kind: 'create' | 'change' | 'delete'; uri: ReturnType<typeof Uri.file> }>();

export const documentCloses = new EventEmitter<StubTextDocument>();

/** Fires the way `vscode.window.onDidChangeVisibleTextEditors` does. */
export const visibleEditorChanges = new EventEmitter<StubTextEditor[]>();
export const activeEditorChanges = new EventEmitter<StubTextEditor | undefined>();

/** Fires the way `vscode.window.onDidChangeTextEditorSelection` does: the
 *  event's `selections` are what the cursor-sensitive listener reads, and the
 *  stub fires it only from tests (or explicitly, the way the editor would
 *  after a reveal moved the caret). */
export const selectionChanges = new EventEmitter<{
  textEditor: StubTextEditor;
  selections: Array<{ active: { line: number; character: number } }>;
}>();

export function resetEditors(): void {
  infoMessageDetails.length = 0;
  editorState.diagnostics.clear();
  // Emptied in place, not reassigned: `window.visibleTextEditors` below is a getter over this
  // exact array, and a fresh array would leave the module reading the old one.
  editorState.documents.clear();
  editorState.versions.clear();
  editorState.dirtyDocuments.clear();
  editorState.appliedEdits.length = 0;
  editorState.visibleTextEditors.length = 0;
  editorState.decorationTypes.length = 0;
  editorState.decorationCalls.length = 0;
  infoMessages.length = 0;
  saveDialogChoice = undefined;
}

/** Put a file in the workspace and open an editor on it. Returns the editor so a test can
 *  read what was drawn on it; call it again with new text to model the user typing. */
export function openStubEditor(fsPath: string, text: string): StubTextEditor {
  editorState.documents.set(fsPath, text);
  editorState.versions.set(fsPath, (editorState.versions.get(fsPath) ?? 0) + 1);
  const existing = editorState.visibleTextEditors.find(
    (editor) => editor.document.uri.fsPath === fsPath,
  );
  if (existing) {
    // The document the editor already holds reads through, so the new text and its
    // version are already visible on it; the caret is deliberately NOT reset —
    // an edit does not move the user's cursor.
    return existing;
  }
  const editor: StubTextEditor = {
    document: stubDocument(fsPath),
    selection: { active: { line: 0, character: 0 } },
    setDecorations(type: StubDecorationType, ranges: unknown[]): void {
      editorState.decorationCalls.push({ fsPath, type, ranges });
    },
    revealRange(): void {},
  };
  editorState.visibleTextEditors.push(editor);
  return editor;
}

/** `window.showTextDocument`: find-or-open an editor on an already-known
 *  document and bring it to the front (the last entry is the stub's active
 *  one). The text MUST already be registered — navigation only ever shows what
 *  the host already read, and anything else here is a code-under-test mistake. */
export function showStubEditor(document: StubTextDocument): StubTextEditor {
  const fsPath = document.uri.fsPath;
  if (!editorState.documents.has(fsPath)) {
    throw new Error(`vscodeStub: no document at ${fsPath}`);
  }
  const found = editorState.visibleTextEditors.find(
    (editor) => editor.document.uri.fsPath === fsPath,
  );
  const editor: StubTextEditor = found ?? {
    document: stubDocument(fsPath),
    selection: { active: { line: 0, character: 0 } },
    setDecorations(type: StubDecorationType, ranges: unknown[]): void {
      editorState.decorationCalls.push({ fsPath, type, ranges });
    },
    revealRange(): void {},
  };
  if (!found) {
    editorState.visibleTextEditors.push(editor);
  } else {
    const index = editorState.visibleTextEditors.indexOf(found);
    editorState.visibleTextEditors.splice(index, 1);
    editorState.visibleTextEditors.push(found);
  }
  return editor;
}

function stubDocument(fsPath: string): StubTextDocument {
  return {
    get isDirty(): boolean { return editorState.dirtyDocuments.has(fsPath); },
    // scheme: 'file' because every document this stub holds came from a path on
    // disk; a caller that refuses non-file URIs is reading exactly this field.
    uri: { fsPath, scheme: 'file', toString: () => fsPath },
    // Read through `editorState.documents` on every call rather than captured once: the
    // buffer the host digests must be the buffer as it is NOW, which is the whole point of
    // reading the document instead of the disk.
    getText: () => {
      const text = editorState.documents.get(fsPath);
      if (text === undefined) {
        throw new Error(`vscodeStub: no document at ${fsPath}`);
      }
      return text;
    },
    get lineCount(): number {
      return (editorState.documents.get(fsPath) ?? '').split('\n').length;
    },
    offsetAt(position: { line: number; character: number }): number {
      const text = editorState.documents.get(fsPath) ?? '';
      const lines = text.split('\n');
      return lines.slice(0, position.line).reduce((sum, line) => sum + line.length + 1, 0) + position.character;
    },
    get version(): number {
      return editorState.versions.get(fsPath) ?? 1;
    },
  };
}

/** A `SourceIO` over the stubbed workspace for the multi-file source-set
 *  collector (workspaceSources.ts): `editorState.documents` IS the filesystem,
 *  every registered document counts as open (so reads see the current buffer
 *  text and version), and enumeration is the `.py` subset under the folder. */
export function stubSourceIO(folder: string): SourceIO {
  const prefix = `${folder}/`;
  return {
    async list() {
      return [...editorState.documents.keys()]
        .filter((fsPath) => fsPath.startsWith(prefix) && fsPath.endsWith('.py'))
        .map((fsPath) => fsPath.slice(prefix.length));
    },
    async read(relativePath) {
      const text = editorState.documents.get(`${prefix}${relativePath}`);
      if (text === undefined) {
        throw new Error(`vscodeStub: no document at ${prefix}${relativePath}`);
      }
      return text;
    },
    openDocument(relativePath) {
      const fsPath = `${prefix}${relativePath}`;
      return editorState.documents.has(fsPath) ? stubDocument(fsPath) : undefined;
    },
  };
}

export enum OverviewRulerLane {
  Left = 1,
  Center = 2,
  Right = 4,
  Full = 7,
}

/** vscode.ThemeColor. Only the id matters: a decoration painted with a literal colour is the
 *  one that looks wrong in the other theme. */
export class ThemeColor {
  constructor(readonly id: string) {}
}

export enum ConfigurationTarget { Global = 1, Workspace = 2, WorkspaceFolder = 3 }
export enum DecorationRangeBehavior { OpenOpen = 0, ClosedClosed = 1, OpenClosed = 2, ClosedOpen = 3 }

// The three module objects above (`Uri`, `workspace`, `window`) are extended here rather than
// edited in place, for the merge reason at the top of this section.
Object.assign(Uri, {
  joinPath(base: { fsPath: string }, ...segments: string[]) {
    const fsPath = [base.fsPath.replace(/\/$/, ''), ...segments].join('/');
    return { fsPath, scheme: 'file', toString: () => fsPath };
  },
  parse(value: string) {
    // A value with no scheme separator is a file path, the way VS Code reads it;
    // `value.split(':')[0]` alone would pronounce every absolute path 'untitled'.
    return {
      fsPath: value,
      scheme: value.includes(':') ? value.split(':')[0] : 'file',
      toString: () => value,
    };
  },
});

Object.assign(workspace, {
  createFileSystemWatcher() {
    const subscribe = (kind: 'create' | 'change' | 'delete') => (listener: (uri: ReturnType<typeof Uri.file>) => void) =>
      configFileChanges.event(event => { if (event.kind === kind) listener(event.uri); });
    return { onDidCreate: subscribe('create'), onDidChange: subscribe('change'), onDidDelete: subscribe('delete'), dispose() {} };
  },
  async openTextDocument(uri: { fsPath: string }): Promise<StubTextDocument> {
    if (!editorState.documents.has(uri.fsPath)) {
      throw new Error(`vscodeStub: no document at ${uri.fsPath}`);
    }
    return stubDocument(uri.fsPath);
  },
  async applyEdit(edit: WorkspaceEdit): Promise<boolean> {
    editorState.appliedEdits.push(edit);
    return true;
  },
  onDidChangeTextDocument: documentChanges.event,
  onDidSaveTextDocument: saveChanges.event,
  onDidCloseTextDocument: documentCloses.event,
});

// `visibleTextEditors` is defined as an accessor and not through Object.assign, which would
// copy the getter's VALUE once and freeze the list as it was at load time.
Object.defineProperty(window, 'visibleTextEditors', {
  get: () => editorState.visibleTextEditors,
});

/** The active editor is the last one opened: close enough to "the editor in focus"
 *  for what the stub exists to prove — the few cursor-sensitive callers read
 *  `activeTextEditor.selection`, and a test moves the caret by assigning it.
 *  Accessor for the same freezing reason as `visibleTextEditors` above. */
Object.defineProperty(window, 'activeTextEditor', {
  get: () => editorState.visibleTextEditors[editorState.visibleTextEditors.length - 1],
});

Object.assign(window, {
  onDidChangeVisibleTextEditors: visibleEditorChanges.event,
  onDidChangeActiveTextEditor: activeEditorChanges.event,
  onDidChangeTextEditorSelection: selectionChanges.event,
  async showTextDocument(document: StubTextDocument): Promise<StubTextEditor> {
    return showStubEditor(document);
  },
  createTextEditorDecorationType(options: Record<string, unknown>): StubDecorationType {
    const type = new StubDecorationType(options);
    editorState.decorationTypes.push(type);
    return type;
  },
  async showSaveDialog(options?: { defaultUri?: { fsPath: string } }): Promise<{ fsPath: string } | undefined> {
    if (saveDialogChoice === null) return undefined;
    const fsPath = saveDialogChoice || options?.defaultUri?.fsPath;
    return fsPath ? { fsPath } : undefined;
  },
  async showInformationMessage(
    message: string,
    ...itemsAndOptions: unknown[]
  ): Promise<string | undefined> {
    infoMessages.push(message);
    const buttons = itemsAndOptions.filter((item): item is string => typeof item === 'string');
    const detail = itemsAndOptions.find(item => item && typeof item === 'object' && 'detail' in item) as { detail?: string } | undefined;
    if (detail?.detail) infoMessageDetails.push(detail.detail);
    return infoResponder?.(message, buttons);
  },
});

// Native webview views: hiding is distinct from disposal, and a hidden view may
// reject messages even with retained DOM. Tests exercise both legal host paths.
export class StubWebviewView {
  visible = true;
  title = '';
  disposed = false;
  deliverWhenHidden = false;
  htmlWrites = 0;
  readonly messages: unknown[] = [];
  private html = '';
  private readonly incoming = new EventEmitter<unknown>();
  private readonly disposedEvent = new EventEmitter<void>();
  private readonly visibility = new EventEmitter<void>();
  readonly onDidDispose = this.disposedEvent.event;
  readonly onDidChangeVisibility = this.visibility.event;
  readonly webview = {
    options: {} as import('vscode').WebviewOptions,
    get html(): string { return ownerForWebview.get(this)!.html; },
    set html(value: string) { const view = ownerForWebview.get(this)!; view.html = value; view.htmlWrites++; },
    onDidReceiveMessage: this.incoming.event,
    postMessage: async (message: unknown): Promise<boolean> => {
      if (this.disposed || (!this.visible && !this.deliverWhenHidden)) return false;
      this.messages.push(message);
      return true;
    },
  };
  constructor() { ownerForWebview.set(this.webview, this); }
  receive(message: unknown): void { if (!this.disposed) this.incoming.fire(message); }
  setVisible(visible: boolean): void { this.visible = visible; this.visibility.fire(); }
  show(): void { this.setVisible(true); }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.visible = false;
    this.disposedEvent.fire();
    this.incoming.dispose(); this.disposedEvent.dispose(); this.visibility.dispose();
  }
}
const ownerForWebview = new WeakMap<object, StubWebviewView>();
export const webviewState = {
  providers: new Map<string, import('vscode').WebviewViewProvider>(),
  registrations: [] as Array<{ id: string; options: unknown }>,
  views: [] as StubWebviewView[],
  contexts: new Map<string, unknown>(),
  commands: [] as Array<{ id: string; args: unknown[] }>,
  handlers: new Map<string, (...args: any[]) => unknown>(),
};
export function resetWebviews(): void {
  webviewState.views.forEach(view => view.dispose());
  webviewState.providers.clear(); webviewState.registrations = []; webviewState.views = [];
  webviewState.contexts.clear(); webviewState.commands = [];
  webviewState.handlers.clear();
}
export const commands = {
  registerCommand(id: string, handler: (...args: any[]) => unknown): { dispose(): void } {
    webviewState.handlers.set(id, handler);
    return { dispose: () => { webviewState.handlers.delete(id); } };
  },
  async executeCommand(id: string, ...args: unknown[]): Promise<unknown> {
    webviewState.commands.push({ id, args });
    if (webviewState.handlers.has(id)) return webviewState.handlers.get(id)!(...args);
    if (id === 'setContext') { webviewState.contexts.set(String(args[0]), args[1]); return; }
    if (id === 'tensorviz.sidebarCanvas.removeView') {
      webviewState.views.find(view => !view.disposed)?.setVisible(false);
      return;
    }
    const provider = webviewState.providers.get(id.replace(/\.focus$/, ''));
    if (!provider) throw new Error(`vscodeStub: unhandled command ${id}`);
    let view = webviewState.views.find(view => !view.disposed);
    if (!view) {
      view = new StubWebviewView(); webviewState.views.push(view);
      provider.resolveWebviewView(view as unknown as import('vscode').WebviewView,
        { state: undefined }, { isCancellationRequested: false, onCancellationRequested: () => ({ dispose() {} }) });
    } else view.show();
  },
};

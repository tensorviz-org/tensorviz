/**
 * The tensorViz activity-bar container: Models and Templates.
 *
 * Models is the workspace's `nn.Module` classes. It shows the index the CodeLens provider
 * already keeps rather than asking the backend again: one AST scan per root, two surfaces on
 * it, and no chance of the tree and the lenses disagreeing about what is in a file.
 *
 * Templates writes a starter model file. It deliberately does not call the backend: scaffolding
 * has to work in the first thirty seconds, before anything Python has started, and a template
 * that needs a running server is a template nobody reaches.
 */
import * as path from 'node:path';
import * as vscode from 'vscode';

import { WorkspaceClass } from './workspaceClassIndex';
import { OPEN_AS_GRAPH } from './commandIds';

export const MODELS_VIEW = 'tensorviz.models';
export const TEMPLATES_VIEW = 'tensorviz.templates';

/** Contributed in package.json; the Templates tree invokes it with a template id. */
export const NEW_MODEL_COMMAND = 'tensorviz.newModelFromTemplate';

/**
 * The proposal ide/patches/0003 grants our built-in in `product.json`
 * `extensionEnabledApiProposals`. It is NEVER declared in package.json: that manifest also
 * ships to stock VS Code, where declaring a proposal trips `vsce package` and disables the
 * extension on a released build.
 */
export const MARKDOWN_MESSAGE_PROPOSAL = 'treeViewMarkdownMessage';

/** How the views reach the class index. Structurally satisfied by `WorkspaceClassIndex`. */
export interface ClassIndex {
  classesFor(root: string): WorkspaceClass[] | undefined;
  refresh(uri: vscode.Uri): Promise<void>;
  readonly onDidChangeClasses: vscode.Event<void>;
}

// ---------------------------------------------------------------------------
// Pure decisions
// ---------------------------------------------------------------------------

/**
 * Whether this host granted us the markdown-message proposal.
 *
 * `ExtensionsProposedApi` copies `product.json`'s grants onto the extension description before
 * the extension host sees it, and `Extension.packageJSON` is that description — so the grant a
 * fork makes is readable here even though our manifest never mentions it. Everything else
 * (stock VS Code, Cursor, a fork build predating the grant) answers false and gets a plain
 * string, which is the degraded path.
 */
export function markdownMessageGranted(packageJSON: unknown): boolean {
  const declared = (packageJSON as { enabledApiProposals?: unknown } | undefined)
    ?.enabledApiProposals;
  return Array.isArray(declared) && declared.includes(MARKDOWN_MESSAGE_PROPOSAL);
}

export interface ModelsState {
  /** Whether `markdownMessageGranted` said yes. */
  markdown: boolean;
  /** Is a folder open at all? */
  hasWorkspace: boolean;
  /** Has the backend answered `/api/ide/workspace/classes` yet? */
  indexed: boolean;
  classCount: number;
  /** Basename of the open folder, so the empty state names the folder that was searched. */
  folderName?: string;
}

/**
 * What the Models view says when it has nothing to list.
 *
 * Returned rather than rendered so this is testable without a workbench: `markdown` tells the
 * caller which of `TreeView.message`'s two types to build, and it is the ONLY difference
 * between the granted and the degraded path — the sentence is the same either way, so no
 * channel is left with a worse explanation, only with a link the user has to find themselves.
 */
export function modelsMessage(state: ModelsState): { markdown: boolean; text: string } | undefined {
  if (state.classCount > 0) {
    return undefined;
  }
  if (!state.hasWorkspace) {
    return {
      markdown: state.markdown,
      text: state.markdown
        ? 'Open the folder your models live in to list them here. [Open Folder](command:workbench.action.files.openFolder) · [Get started](command:tensorviz.getStarted)'
        : 'Open the folder your models live in to list them here. Run "tensorViz: Get Started" for a tour.',
    };
  }
  // Indexing, not an instruction to start anything (task.md P0). There is nothing for the user
  // to start any more: the backend comes up on its own when a Python file that defines an
  // nn.Module is opened, so a row asking them to press play was asking them to do the one
  // thing the milestone exists to delete.
  if (!state.indexed) {
    const where = state.folderName ? ` ${state.folderName}` : ' this folder';
    return {
      markdown: state.markdown,
      text: state.markdown
        ? `Indexing${where} for \`nn.Module\` classes\u2026`
        : `Indexing${where} for nn.Module classes\u2026`,
    };
  }
  const searched = state.folderName ? ` in ${state.folderName}` : ' here';
  return {
    markdown: state.markdown,
    text: state.markdown
      ? `No \`nn.Module\` classes found${searched}. [New model from a template](command:tensorviz.newModelFromTemplate)`
      : `No nn.Module classes found${searched}. Use the Templates view to start one.`,
  };
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

export interface ModelTemplate {
  id: string;
  label: string;
  description: string;
  /** Default base name of the file, without `.py`. */
  fileStem: string;
  /** `{cls}` is replaced with the class name derived from the file name. */
  body: string;
}

/**
 * Two starters, not twelve. Each is a file the user is meant to edit immediately, so the value
 * is in it parsing into a graph on the first "Open as Graph" — not in covering architectures.
 * The CNN ends on a `mean` over the spatial dims on purpose: that is a plain-Python statement,
 * and seeing it appear as its own node is how the opaque-node behaviour explains itself.
 */
export const MODEL_TEMPLATES: readonly ModelTemplate[] = [
  {
    id: 'mlp',
    label: 'MLP classifier',
    description: 'Linear → ReLU → Dropout → Linear',
    fileStem: 'mlp',
    body: `import torch.nn as nn


# A flat 784-vector in batches of 2. Without this line the drawing tier cannot guess the
# input for a Linear-first model and falls back to the image default 1,3,224,224, which
# covers a correct model in shape errors on its very first open (task.md P0).
# tensorviz: input=2,784
class {cls}(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        return self.fc2(h)
`,
  },
  {
    id: 'cnn',
    label: 'Small CNN',
    description: 'Two conv blocks, global average pool, linear head',
    fileStem: 'cnn',
    body: `import torch.nn as nn


class {cls}(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        h = self.act1(self.conv1(x))
        h = self.pool(h)
        h = self.act2(self.conv2(h))
        h = h.mean(dim=(2, 3))
        return self.head(h)
`,
  },
];

/** `my_net` -> `MyNet`. A leading digit or an empty stem falls back to `Model`, because the
 *  result is pasted straight into a `class` statement and has to be an identifier. */
export function classNameFor(fileStem: string): string {
  const name = fileStem
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join('');
  return /^[A-Za-z_]/.test(name) ? name : 'Model';
}

export function renderTemplate(template: ModelTemplate, className: string): string {
  return template.body.split('{cls}').join(className);
}

// ---------------------------------------------------------------------------
// The trees
// ---------------------------------------------------------------------------

interface ModelsGroup {
  kind: 'workspace' | 'folder' | 'file';
  root: string;
  file: string;
  name: string;
  count: number;
  children: ModelsNode[];
}

/** Keep the owning root on each class: relative paths can repeat across workspaces. */
export type ModelsNode = ModelsGroup | {
  kind: 'model';
  root: string;
  name: string;
  entry: WorkspaceClass;
};

/** Group the existing source index without rescanning files or inferring model ownership. */
function modelHierarchy(root: string, name: string, classes: WorkspaceClass[]): ModelsGroup {
  const workspace: ModelsGroup = { kind: 'workspace', root, file: '', name, count: classes.length, children: [] };
  const groups = new Map<string, ModelsGroup>();
  for (const entry of classes) {
    let parent = workspace;
    const parts = entry.file.split('/');
    let file = '';
    for (let i = 0; i < parts.length; i++) {
      file = file ? `${file}/${parts[i]}` : parts[i];
      let group = groups.get(file);
      if (!group) {
        group = {
          kind: i === parts.length - 1 ? 'file' : 'folder',
          root, file, name: parts[i], count: 0, children: [],
        };
        groups.set(file, group);
        parent.children.push(group);
      }
      group.count++;
      parent = group;
    }
    parent.children.push({ kind: 'model', root, name: entry.name, entry });
  }
  for (const group of [workspace, ...groups.values()]) {
    group.children.sort((a, b) =>
      Number(b.kind === 'folder') - Number(a.kind === 'folder') ||
      a.name.localeCompare(b.name) ||
      (a.kind === 'model' && b.kind === 'model' ? a.entry.line - b.entry.line : 0));
  }
  return workspace;
}

export class ModelsTree implements vscode.TreeDataProvider<ModelsNode> {
  private readonly changed = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changed.event;
  /** Set by `attach`, so the empty-state message can be written where the user reads it. */
  private view: vscode.TreeView<ModelsNode> | undefined;

  constructor(private readonly index: ClassIndex) {}

  /** Roots the index is keyed by: `WorkspaceClassIndex.rootFor` uses the workspace folder. */
  private roots(): string[] {
    return (vscode.workspace.workspaceFolders ?? []).map((folder) => folder.uri.fsPath);
  }

  getTreeItem(node: ModelsNode): vscode.TreeItem {
    if (node.kind !== 'model') {
      const item = new vscode.TreeItem(node.name, node.kind === 'workspace'
        ? vscode.TreeItemCollapsibleState.Expanded : vscode.TreeItemCollapsibleState.Collapsed);
      item.id = JSON.stringify([node.kind, node.root, node.file]);
      item.description = `${node.count} ${node.count === 1 ? 'model' : 'models'}`;
      item.resourceUri = vscode.Uri.file(path.join(node.root, node.file));
      item.iconPath = new vscode.ThemeIcon(node.kind === 'file' ? 'file-code' : 'folder');
      item.tooltip = path.join(node.root, node.file);
      return item;
    }
    const entry = node.entry;
    const item = new vscode.TreeItem(entry.name, vscode.TreeItemCollapsibleState.None);
    item.id = JSON.stringify(['model', node.root, entry.file, entry.name, entry.line]);
    item.description = `line ${entry.line}`;
    item.iconPath = new vscode.ThemeIcon('circuit-board');
    item.tooltip = `${entry.name} (${entry.bases.join(', ')})\n${path.join(node.root, entry.file)}:${entry.line}`;
    item.contextValue = 'tensorviz.model';
    item.command = {
      title: 'Open as Graph',
      command: OPEN_AS_GRAPH,
      arguments: [
        { uri: vscode.Uri.file(path.join(node.root, entry.file)).toString(), cls: entry.name },
      ],
    };
    return item;
  }

  getChildren(parent?: ModelsNode): ModelsNode[] {
    if (parent) {
      return parent.kind === 'model' ? [] : parent.children;
    }
    const folders = vscode.workspace.workspaceFolders ?? [];
    const roots = folders.map((folder) => modelHierarchy(
      folder.uri.fsPath,
      folder.name ?? path.basename(folder.uri.fsPath),
      this.index.classesFor(folder.uri.fsPath) ?? [],
    ));
    this.showMessage(roots.reduce((total, root) => total + root.count, 0));
    // No start row: an unindexed folder shows the "Indexing…" message and then fills in.
    return folders.length === 1 ? roots[0].children : roots.filter((root) => root.count > 0);
  }

  private showMessage(classCount: number): void {
    if (!this.view) {
      return;
    }
    const roots = this.roots();
    const message = modelsMessage({
      markdown: markdownMessageGranted(
        vscode.extensions.getExtension('tensorviz.tensorviz')?.packageJSON,
      ),
      hasWorkspace: roots.length > 0,
      indexed: roots.some((root) => this.index.classesFor(root) !== undefined),
      classCount,
      folderName: roots[0] ? path.basename(roots[0]) : undefined,
    });
    if (!message) {
      this.view.message = undefined;
      return;
    }
    // `TreeView.message` is typed `string | undefined` in the stable API; a MarkdownString is
    // accepted only where the proposal is granted, which is exactly what `markdown` reports.
    this.view.message = message.markdown
      ? (new vscode.MarkdownString(message.text, true) as unknown as string)
      : message.text;
  }

  attach(view: vscode.TreeView<ModelsNode>): void {
    this.view = view;
  }

  refresh(): void {
    this.changed.fire();
  }

  dispose(): void {
    this.changed.dispose();
  }
}

export class TemplatesTree implements vscode.TreeDataProvider<ModelTemplate> {
  getTreeItem(template: ModelTemplate): vscode.TreeItem {
    const item = new vscode.TreeItem(template.label, vscode.TreeItemCollapsibleState.None);
    item.description = template.description;
    item.iconPath = new vscode.ThemeIcon('new-file');
    item.command = {
      title: 'New model from template',
      command: NEW_MODEL_COMMAND,
      arguments: [template.id],
    };
    return item;
  }

  getChildren(parent?: ModelTemplate): ModelTemplate[] {
    return parent ? [] : [...MODEL_TEMPLATES];
  }
}

// ---------------------------------------------------------------------------
// Scaffolding
// ---------------------------------------------------------------------------

export interface ScaffoldHost {
  /** Where `models/` should go. Undefined when no folder is open. */
  workspaceRoot(): string | undefined;
  ask(prompt: string, value: string): Thenable<string | undefined>;
  exists(fsPath: string): Promise<boolean>;
  write(fsPath: string, contents: string): Promise<void>;
  open(fsPath: string): Promise<void>;
  warn(message: string): Thenable<unknown>;
}

export type ScaffoldOutcome =
  | { ok: true; fsPath: string; className: string }
  | { ok: false; reason: 'no-workspace' | 'cancelled' | 'exists' };

/**
 * Write `models/<name>.py` and open it.
 *
 * Refuses to overwrite. The whole premise of the product is that the `.py` is the truth
 * (task.md §1), so a template that can silently replace one is the one bug this feature must
 * not have — and "pick another name" is a cheaper answer than an undo stack.
 */
export async function scaffoldModel(
  host: ScaffoldHost,
  template: ModelTemplate,
  /** A name the caller already asked for. Given one, this does NOT ask again.
   *
   *  The catalogue path has to settle the name first, because the backend writes the
   *  class name into the source it generates — so without this the user was asked for
   *  the same file name twice in a row, once by the caller and once here. */
  askedStem?: string,
): Promise<ScaffoldOutcome> {
  const root = host.workspaceRoot();
  if (!root) {
    await host.warn('Open a folder first — a new model file needs somewhere to go.');
    return { ok: false, reason: 'no-workspace' };
  }
  const stem = askedStem ?? (await host.ask('File name for the new model', template.fileStem));
  if (!stem) {
    return { ok: false, reason: 'cancelled' };
  }
  const cleaned = stem.replace(/\.py$/i, '');
  const fsPath = path.join(root, 'models', `${cleaned}.py`);
  if (await host.exists(fsPath)) {
    await host.warn(`${path.join('models', `${cleaned}.py`)} already exists.`);
    return { ok: false, reason: 'exists' };
  }
  const className = classNameFor(cleaned);
  await host.write(fsPath, renderTemplate(template, className));
  await host.open(fsPath);
  return { ok: true, fsPath, className };
}

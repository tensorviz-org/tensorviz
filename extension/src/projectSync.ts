/** Keeps the page's project view in step with the workspace.
 *
 * The class-level counterpart is `localBrowserSync.ts`. This one owns the other
 * half: which experiments exist, which one is focused, which frames are open, and
 * publishing each of those over the broker as ordered host state.
 *
 * The division of labour matters, because it is what keeps the page honest:
 *
 *  - the **page** names an experiment or a node and nothing else;
 *  - the **host** decides whether the index it named is still current, resolves
 *    what that experiment actually contains, and publishes the answer.
 *
 * So a page holding a stale index cannot make the canvas draw something that no
 * longer exists — the request is refused rather than served, exactly as a stale
 * `revealNode` is refused rather than aimed at an editor.
 *
 * Nothing here writes a workspace file.
 */

import {
  LocalBrowserProjectIdentity,
  isLocalBrowserProjectIdentity,
  sameProjectIdentity,
} from '@tensorviz/contracts/protocol';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import { ProjectIndexResult } from './projectIndex';
import { isNormalizedPythonPath } from './workspaceSources';

export interface ProjectDeclarationContext {
  project: LocalBrowserProjectIdentity;
  entrypoint: string;
  dependencies: ReadonlyMap<string, string>;
  labels: string[];
}

/** The index as the backend returns it; only the fields this controller reads. */
export interface IndexedProject {
  experiments: Array<{
    experiment_id: string;
    label: string;
    entrypoint: string;
    root: string;
    closure: string[];
    graph?: { nodes: Array<{ label: string }> };
  }>;
  unreached: Array<{ file: string; error?: string | null }>;
  roots: string[];
  truncated: boolean;
}

export interface ProjectSyncOptions {
  /** Index one workspace folder. Returns undefined when the backend is not up. */
  index(folder: string): Promise<ProjectIndexResult | undefined>;
  /** Build the document for one experiment with a given expansion set. */
  document(
    folder: string,
    experimentId: string,
    expanded: string[],
  ): Promise<{ document: unknown; spanmap: unknown; expanded: string[]; digest?: string } | undefined>;
  publish(message: Extract<LocalBrowserHostPublish, { type: 'projectIndex' | 'projectDocument' }>): void;
  log?(message: string): void;
  onDidChangeTarget?(): void;
}

/** Why a frame is open. `auto` is the editor's doing and the editor may undo it;
 *  `manual` is a click and survives a focus change; `pinned` survives everything
 *  short of the user closing it. */
export type ExpansionReason = 'auto' | 'manual' | 'pinned';

const RANK: Record<ExpansionReason, number> = { auto: 0, manual: 1, pinned: 2 };

export class ProjectSync {
  private folder = '';
  private index: IndexedProject | undefined;
  /** The index's own admission that its file set was incomplete. Held rather than
   *  passed through, so a replay cannot quietly report a partial project as whole. */
  private truncated = false;
  /** Files actually indexed, and files enumeration saw before any budget applied.
   *  Held so a replay reports the same partial project the first index admitted. */
  private indexed = 0;
  private discovered = 0;
  private digest = '';
  /** Bumped per published index, so two indexes of one workspace are ordered the
   *  way two projections of one class are. */
  private revision = 0;
  private focused = '';
  /** Per experiment, because expanding a frame in one and switching away should
   *  not lose it — the same reason positions survive a retarget. */
  /** experiment -> node -> WHY it is open (step 3.3).
   *
   * The reason is load-bearing, not bookkeeping. Closing a file may only undo what
   * opening it did: a node the EDITOR expanded collapses when focus leaves, and a
   * node the USER clicked open stays, because collapsing it would throw away a
   * decision they made by hand. Without the distinction, "recollapse on close"
   * silently destroys the reader's own expansions — the same class of defect as a
   * camera that moves while you are reading. */
  private readonly expanded = new Map<string, Map<string, ExpansionReason>>();
  /** Bumped on every retarget/re-index, so a late answer for a superseded index
   *  is dropped rather than published over a newer one. */
  private generation = 0;
  private documentGeneration = 0;
  private lastDocument?: Extract<LocalBrowserHostPublish, { type: 'projectDocument' }>;
  private declarationSources?: ReadonlyMap<string, string>;

  constructor(private readonly options: ProjectSyncOptions) {}

  identity(): LocalBrowserProjectIdentity | undefined {
    if (!this.folder || !this.digest) {
      return undefined;
    }
    return {
      workspaceFolder: this.folder,
      projectRoot: this.rootOf(this.focused),
      experimentId: this.focused,
      indexDigest: this.digest,
      indexRevision: this.revision,
    };
  }

  private rootOf(experimentId: string): string {
    const experiment = this.index?.experiments.find((e) => e.experiment_id === experimentId);
    return experiment?.root ?? '';
  }

  /** Index `folder` and publish the result. The focused experiment survives a
   *  re-index when it still exists, so typing does not move the user's view. */
  async open(folder: string): Promise<void> {
    this.invalidateDeclarations();
    if (this.folder !== folder) {
      this.index = undefined; this.digest = ''; this.focused = ''; this.expanded.clear(); this.lastDocument = undefined;
    }
    this.folder = folder;
    const generation = this.generation;

    let result: ProjectIndexResult | undefined;
    try { result = await this.options.index(folder); }
    catch (error) { if (generation !== this.generation) return; throw error; }
    if (generation !== this.generation) return;
    if (result === undefined) throw new Error('Project index unavailable');
    if (![result.indexed, result.discovered].every(count => Number.isSafeInteger(count) && count >= 0) ||
      typeof result.truncated !== 'boolean' || !isIndexedProject(result.index) || !isLocalBrowserProjectIdentity({ workspaceFolder: folder, projectRoot: '',
      experimentId: '', indexDigest: result.digest, indexRevision: this.revision + 1 })) throw new Error('Invalid project index');

    this.index = result.index;
    this.lastDocument = undefined;
    this.truncated = result.truncated;
    this.indexed = result.indexed;
    this.discovered = result.discovered;
    this.digest = result.digest;
    this.revision += 1;
    this.declarationSources = result.sources;

    const ids = this.index.experiments.map((e) => e.experiment_id);
    if (!ids.includes(this.focused)) {
      // Prefer something the code proved over something a filename suggested.
      this.focused = ids[0] ?? '';
    }

    const identity = this.identity();
    if (!identity) {
      return;
    }
    this.options.publish({
      type: 'projectIndex',
      project: identity,
      index: this.index,
      truncated: this.truncated,
      indexed: this.indexed,
      discovered: this.discovered,
    });

    if (this.focused) await this.publishDocument(generation);
  }

  /** Retarget/disposal revokes old declaration authority immediately, even if
   * an earlier index request is still in flight. */
  invalidateDeclarations(): void {
    this.options.onDidChangeTarget?.();
    this.declarationSources = undefined;
    this.generation += 1;
    this.documentGeneration += 1;
  }

  /** Say everything again, without indexing again.
   *
   *  A webview that reloads — the panel restored with the window, the user
   *  reloading it, VS Code re-creating it on a theme change — comes back with no
   *  memory and no way to ask. The host already replays the class projection when
   *  the app says `ready`, for exactly this reason: postMessage reports success for
   *  a message the page never saw, so a replay is the acknowledgement the transport
   *  does not provide. The project plane had no such replay, so a reloaded panel
   *  drew a document with no index behind it — no experiment rail, no status bar,
   *  and nothing saying why.
   *
   *  Cheap on purpose: the index is held, so this re-sends what is already known
   *  and replays the document from the same indexed sources. A reload never
   *  rebuilds against newer buffers under the old index identity. */
  async republish(): Promise<void> {
    const identity = this.identity();
    if (!identity || !this.index) {
      return;
    }
    this.options.publish({
      type: 'projectIndex',
      project: identity,
      index: this.index,
      truncated: this.truncated,
      indexed: this.indexed,
      discovered: this.discovered,
    });
    if (this.lastDocument && sameProjectIdentity(identity, this.lastDocument.project)) this.options.publish(this.lastDocument);
  }

  /** The page asked to focus another experiment. */
  async focus(asked: LocalBrowserProjectIdentity, experimentId: string): Promise<boolean> {
    if (!this.accepts(asked)) {
      return false;
    }
    if (experimentId && !this.index?.experiments.some((e) => e.experiment_id === experimentId)) {
      this.options.log?.(`project focus names an experiment this index does not have`);
      return false;
    }
    if (this.focused !== experimentId) this.options.onDidChangeTarget?.();
    this.focused = experimentId;
    await this.publishDocument(this.generation);
    return true;
  }

  /** The page opened or closed a frame. Static expansion only — this never runs
   *  the model, which is what the separate trace-deepening gesture is for. */
  async expand(
    asked: LocalBrowserProjectIdentity,
    nodeId: string,
    open: boolean,
    reason: ExpansionReason = 'manual',
  ): Promise<boolean> {
    if (!this.accepts(asked)) {
      return false;
    }
    this.setExpanded(nodeId, open, reason);
    await this.publishDocument(this.generation);
    return true;
  }

  /** Collapse everything the EDITOR opened, leaving what the user did alone.
   *  What "close the file and the graph goes back" is allowed to mean (3.3). */
  async collapseAuto(): Promise<void> {
    const open = this.expanded.get(this.focused);
    if (!open) {
      return;
    }
    let changed = false;
    for (const [nodeId, reason] of Array.from(open)) {
      if (reason === 'auto') {
        open.delete(nodeId);
        changed = true;
      }
    }
    if (changed) {
      await this.publishDocument(this.generation);
    }
  }

  /** Why a node is open, or undefined when it is not. */
  reasonFor(nodeId: string): ExpansionReason | undefined {
    return this.expanded.get(this.focused)?.get(nodeId);
  }

  private setExpanded(nodeId: string, open: boolean, reason: ExpansionReason): void {
    const current = this.expanded.get(this.focused) ?? new Map<string, ExpansionReason>();
    if (!open) {
      current.delete(nodeId);
    } else {
      // A hand-made decision outranks an automatic one, and is never quietly
      // downgraded: the editor re-entering a node the user pinned must not turn
      // that pin into something the next file-close can undo.
      const existing = current.get(nodeId);
      current.set(nodeId, RANK[reason] >= RANK[existing ?? 'auto'] ? reason : existing!);
    }
    this.expanded.set(this.focused, current);
  }

  /** Exact identity, like `sameTargetIdentity`. A page one index behind is refused
   *  rather than served: it is asking about something that no longer exists. */
  private accepts(asked: LocalBrowserProjectIdentity | undefined): boolean {
    const current = this.identity();
    if (!asked || !current) {
      return false;
    }
    const ok = sameProjectIdentity(asked, current);
    if (!ok) {
      this.options.log?.('project request refused: the page is holding a stale index');
    }
    return ok;
  }

  private async publishDocument(generation: number): Promise<void> {
    const identity = this.identity();
    if (!identity) return;
    const request = ++this.documentGeneration;
    if (!this.focused) {
      this.lastDocument = undefined;
      this.options.publish({ type: 'projectIndex', project: identity, index: this.index, truncated: this.truncated,
        indexed: this.indexed, discovered: this.discovered });
      this.lastDocument = { type: 'projectDocument', project: identity, document: {
        graph_id: '', experiment_id: '', name: '', nodes: [], edges: [], spanmap: {}, expanded: [],
      }, spanmap: {}, expanded: [] };
      this.options.publish(this.lastDocument);
      return;
    }
    const expanded = Array.from((this.expanded.get(this.focused) ?? new Map<string, ExpansionReason>()).keys()).sort();
    let built: Awaited<ReturnType<ProjectSyncOptions['document']>>;
    try { built = await this.options.document(this.folder, this.focused, expanded); }
    catch (error) { if (generation !== this.generation || request !== this.documentGeneration) return; throw error; }
    if (generation !== this.generation || request !== this.documentGeneration) return;
    if (built === undefined) throw new Error('Project document unavailable');
    if (built.digest !== undefined && built.digest !== identity.indexDigest) throw new Error('Project sources changed while drawing');
    const message: Extract<LocalBrowserHostPublish, { type: 'projectDocument' }> = {
      type: 'projectDocument',
      project: identity,
      document: built.document,
      spanmap: built.spanmap,
      expanded: built.expanded,
    };
    this.options.publish(message);
    this.lastDocument = message;
  }

  /** For tests, the snapshot path, and the declaration writer — which needs to know
   *  WHICH file a declaration belongs in, and which labels the graph has so a
   *  dangling edge is refused before it is written. */
  state(): {
    focused: string;
    revision: number;
    expanded: string[];
    entrypoint: string;
    labels: string[];
  } {
    const experiment = this.index?.experiments.find(
      (e) => e.experiment_id === this.focused,
    );
    return {
      focused: this.focused,
      revision: this.revision,
      expanded: Array.from((this.expanded.get(this.focused) ?? new Map<string, ExpansionReason>()).keys()).sort(),
      entrypoint: experiment?.entrypoint ?? '',
      labels: this.labels,
    };
  }

  /** Declaration authority is stricter than navigation: exact focus/root and
   * indexed source evidence are required, including while a re-index is pending. */
  declarationContext(asked: LocalBrowserProjectIdentity): ProjectDeclarationContext | undefined {
    const project = this.identity();
    const experiment = this.index?.experiments.find(e => e.experiment_id === this.focused);
    const sources = this.declarationSources;
    if (!project || !sameProjectIdentity(asked, project) || !experiment || !sources ||
      !isNormalizedPythonPath(experiment.entrypoint) || !sources.has(experiment.entrypoint)) return undefined;
    const dependencies = new Map<string, string>();
    for (const file of new Set([experiment.entrypoint, ...experiment.closure])) {
      const digest = sources.get(file);
      if (!isNormalizedPythonPath(file) || !digest) return undefined;
      dependencies.set(file, digest);
    }
    return { project, entrypoint: experiment.entrypoint, dependencies,
      labels: (experiment.graph?.nodes ?? []).map(node => node.label) };
  }

  /** Labels of the focused experiment's graph, kept from the last document. */
  private labels: string[] = [];

  rememberLabels(labels: string[]): void {
    this.labels = labels;
  }
}

/** Validate the fields consumed by the host before granting index authority. */
function isIndexedProject(value: unknown): value is IndexedProject {
  if (!value || typeof value !== 'object') return false;
  const index = value as IndexedProject;
  return Array.isArray(index.experiments) && index.experiments.every(e => e &&
    typeof e.experiment_id === 'string' && !!e.experiment_id && typeof e.label === 'string' &&
    typeof e.entrypoint === 'string' && typeof e.root === 'string' &&
    Array.isArray(e.closure) && e.closure.every(file => typeof file === 'string') &&
    (e.graph === undefined || (e.graph && Array.isArray(e.graph.nodes) &&
      e.graph.nodes.every(node => node && typeof node.label === 'string')))) &&
    Array.isArray(index.unreached) && Array.isArray(index.roots) && index.roots.every(root => typeof root === 'string') &&
    typeof index.truncated === 'boolean';
}

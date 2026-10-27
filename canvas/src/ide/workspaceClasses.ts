/**
 * The user's own `nn.Module` classes, for the palette's Workspace section
 * (task.md M6).
 *
 * These are NOT layerTypes.ts entries and NOT canonical registry ops: they are
 * classes found in the open folder by `/api/ide/workspace/classes`, which reads
 * source by AST and imports nothing (task.md §8.5). Inserting one produces a
 * `custom_code` node carrying `_class` and `_import_path`, which is how it stays
 * out of `CanonicalTemplateRegistry` — the template sweep enumerates that
 * registry and user code entering it would be a gate failure, not a feature.
 *
 * The scan needs an absolute root and the app has none: it is served over http
 * by the backend and framed by the webview shell, and nothing in that URL says
 * which folder is open. So the shell is asked, once, when the palette mounts.
 * Asking (rather than listening for an announcement) is what makes it reliable:
 * the shell answers `hostReady` within milliseconds of startup, long before any
 * palette exists, and a listener attached later would have missed it.
 *
 * When no shell answers — the hosted web product, a plain browser tab, an IDE
 * whose backend is not up — `workspaceRoot()` resolves to null and the palette
 * renders no Workspace section at all. Absent, not empty: a section that is
 * permanently "Loading…" in the web product would be a promise the product
 * cannot keep.
 */
import { API_BASE_URL, localApiHeaders } from '../config/apiBase';
import { bridgeActive, postToHost } from './bridge';

/** One class, exactly as `/api/ide/workspace/classes` reports it
 *  (backend/api/routers/ide.py, `workspace_classes`). */
export interface WorkspaceClass {
  name: string;
  /** Relative to the scanned root, forward slashes. */
  file: string;
  /** 1-based line of the `class` statement. */
  line: number;
  bases: string[];
  ctor_required: number;
}

interface WorkspaceClassesResponse {
  root: string;
  classes: WorkspaceClass[];
  file_count: number;
}

/** How long the shell gets to answer before the section stays absent. */
const ROOT_TIMEOUT_MS = 1500;

/** The open folder, once the host has named it. Cached: it cannot change without
 *  the window being reloaded, and the palette mounts on every ⇧A. */
let root: string | null = null;

/** Test seam. */
export function setWorkspaceRoot(value: string | null): void {
  root = value;
}

/**
 * Ask the shell which folder is open.
 *
 * Resolves to null rather than rejecting when there is no host, when the host
 * speaks a protocol without this message, or when it simply does not answer:
 * every one of those is "there is no workspace here", which the caller renders
 * the same way.
 */
export function workspaceRoot(): Promise<string | null> {
  if (root !== null) return Promise.resolve(root);
  if (!bridgeActive()) return Promise.resolve(null);
  return new Promise((resolve) => {
    let done = false;
    const finish = (value: string | null) => {
      if (done) return;
      done = true;
      window.removeEventListener('message', onMessage);
      window.clearTimeout(timer);
      root = value;
      resolve(value);
    };
    const onMessage = (event: MessageEvent) => {
      const data = event.data as { type?: string; root?: unknown } | undefined;
      if (!data || data.type !== 'workspaceRoot') return;
      finish(typeof data.root === 'string' && data.root ? data.root : null);
    };
    const timer = window.setTimeout(() => finish(null), ROOT_TIMEOUT_MS);
    window.addEventListener('message', onMessage);
    postToHost({ type: 'workspaceRoot' });
  });
}

/** The classes under `root`. Throws on a non-200 so the caller can decide that
 *  the section is absent rather than wrong. */
export async function fetchWorkspaceClasses(scanRoot: string): Promise<WorkspaceClass[]> {
  const response = await fetch(`${API_BASE_URL}/api/ide/workspace/classes`, {
    method: 'POST',
    // localApiHeaders is the IDE's per-launch secret and is empty everywhere else.
    headers: { 'Content-Type': 'application/json', ...localApiHeaders() },
    body: JSON.stringify({ root: scanRoot }),
  });
  if (!response.ok) {
    throw new Error(`/api/ide/workspace/classes returned ${response.status}`);
  }
  return ((await response.json()) as WorkspaceClassesResponse).classes || [];
}

/** At most this many rows, matching the template section's own cap: the palette
 *  is a search box, and a workspace with 300 blocks is searched, not scrolled. */
export const MAX_VISIBLE_WORKSPACE_ITEMS = 28;

/** The Workspace rows for a query, in the order they are shown. */
export function selectWorkspaceClasses(
  classes: readonly WorkspaceClass[],
  query: string,
): WorkspaceClass[] {
  const q = query.trim().toLowerCase();
  const matched = q
    ? classes.filter(
        (entry) => entry.name.toLowerCase().includes(q) || entry.file.toLowerCase().includes(q),
      )
    : classes;
  return [...matched]
    .sort((a, b) => a.name.localeCompare(b.name))
    .slice(0, MAX_VISIBLE_WORKSPACE_ITEMS);
}

/**
 * The module path `custom_code` codegen imports the class from — derived the way
 * `repo_scan.instantiate` does, from the file's path relative to the scanned
 * root. `models/blocks.py` is `models.blocks`.
 */
export function importPathOf(file: string): string {
  return file
    .replace(/\\/g, '/')
    .replace(/\.py$/, '')
    .replace(/\/__init__$/, '')
    .split('/')
    .filter(Boolean)
    .join('.');
}

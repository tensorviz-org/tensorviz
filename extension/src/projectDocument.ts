import {
  ProjectSourceIO,
  bodiesFor,
  collectProjectSources,
  vscodeProjectSourceIO,
} from './projectSources';

/** What building a project document needs, named rather than reached for.
 *
 * Extracted from `LocalBrowserGraph` when the project canvas moved into the IDE
 * panel (step 2.6): two surfaces draw the same experiments from the same backend,
 * and a second copy of this would be two implementations of one contract, drifting
 * the moment either is fixed. Nothing here knows which surface asked. */
export interface ProjectDocumentDeps {
  origin(): string | undefined;
  token(): string | undefined;
  sourceIO(folder: string): ProjectSourceIO;
  log(line: string): void;
  fetchImpl?: typeof fetch;
}

/** The default IO: the workspace's own files, read through VS Code. */
export function defaultProjectDocumentIO(folder: string): ProjectSourceIO {
  return vscodeProjectSourceIO(folder);
}

/** Draw one experiment. Separate from the index because the two change at
 *  different rates — an index moves whenever any file does. */
export async function buildProjectDocument(
  deps: ProjectDocumentDeps,
  folder: string,
  experimentId: string,
  expanded: string[],
): Promise<{ document: unknown; spanmap: unknown; expanded: string[]; digest?: string } | undefined> {

  const origin = deps.origin();
  const token = deps.token();
  if (!origin || !token) {
    return undefined;
  }
  const io = deps.sourceIO(folder);
  const set = await collectProjectSources(io);
  const response = await (deps.fetchImpl ?? fetch)(`${origin}/api/ide/project/document`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-tensorviz-token': token },
    body: JSON.stringify({
      protocol_version: 1,
      root: folder,
      manifest: set.manifest,
      // The index filled the backend's cache, so drawing normally costs no
      // upload at all; bodies are sent only if that cache has been evicted.
      bodies: [],
      truncated: set.truncated,
      experiment_id: experimentId,
      expanded,
    }),
  });
  if (!response.ok) {
    deps.log(`Project: document refused (${response.status}).`);
    return undefined;
  }
  const body = (await response.json()) as {
    status: string;
    missing?: string[];
    document?: unknown;
    spanmap?: unknown;
    expanded?: string[];
    digest?: string;
  };
  if (body.status === 'need_bodies') {
    const retry = await (deps.fetchImpl ?? fetch)(`${origin}/api/ide/project/document`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-tensorviz-token': token },
      body: JSON.stringify({
        protocol_version: 1,
        root: folder,
        manifest: set.manifest,
        bodies: bodiesFor(set, body.missing ?? []).bodies,
        truncated: set.truncated,
        experiment_id: experimentId,
        expanded,
      }),
    });
    if (!retry.ok) {
      return undefined;
    }
    const filled = (await retry.json()) as {
      document?: unknown;
      spanmap?: unknown;
      expanded?: string[];
      digest?: string;
    };
    return {
      document: filled.document,
      spanmap: filled.spanmap,
      expanded: filled.expanded ?? expanded,
      digest: filled.digest,
    };
  }
  return {
    document: body.document,
    spanmap: body.spanmap,
    expanded: body.expanded ?? expanded,
    digest: body.digest,
  };
}

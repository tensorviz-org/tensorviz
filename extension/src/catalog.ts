/**
 * The backend's template catalogue, as files the user can open.
 *
 * `MODEL_TEMPLATES` in views.ts is two hand-written starters, and deliberately so —
 * each is a file meant to be edited immediately, and each carries the
 * `# tensorviz: input=` hint that makes it draw correctly on the first open. They stay.
 *
 * What they were not is a catalogue. The backend serves 56 gallery starters and 17
 * popular architectures, every one of which instantiates, and the IDE offered two of
 * them. The web gallery used to be reachable inside the editor and is now gone (it
 * offered a canvas mutation the IDE cannot perform, task.md M5), so the remaining path
 * has to carry the catalogue itself.
 *
 * What a template means in an IDE is a NEW `.py` FILE, not a canvas edit: the `.py` is
 * the authority, so a starter architecture is source that lands beside the user's code
 * and is theirs to change. That is what the scaffolder already does, and this hands it a
 * template whose body came from the backend rather than from a literal in this file.
 */
import type { ModelTemplate } from './views';

/** One entry as the picker shows it. */
export interface CatalogEntry {
  id: string;
  label: string;
  description: string;
  /** Which endpoint instantiates it — the groups answer on different routes. */
  group: 'template' | 'popular' | 'canonical';
  /** The gallery category, shown so the list is navigable at 73 items. */
  category?: string;
}

/** What a caller must give us to reach the backend. Kept as data so the pure half tests. */
export interface CatalogHost {
  origin(): string | undefined;
  headers(): Record<string, string>;
  /** POST returning parsed JSON, or throwing with the server's message. */
  post(url: string, headers: Record<string, string>, body: unknown): Promise<unknown>;
  get(url: string, headers: Record<string, string>): Promise<unknown>;
}

/** A file stem the user will recognise, from an id nobody typed. */
export function stemFor(id: string): string {
  return id.replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '').toLowerCase() || 'model';
}

/**
 * The catalogue as flat picker entries.
 *
 * Shaped from the same two blocks the gallery reads — `templates` and `popular_models`.
 * `module_templates` is deliberately not here: those are canvas blocks with an entry and
 * exit node, meant to be inserted into a graph rather than to stand as a file, and 15 of
 * the ids that block advertises do not exist in the registry at all.
 */
export function entriesFrom(categories: unknown, perCategory: Map<string, unknown>): CatalogEntry[] {
  const out: CatalogEntry[] = [];
  const seen = new Set<string>();
  const list = (categories as { categories?: Array<Record<string, unknown>> })?.categories ?? [];
  for (const category of list) {
    const cid = String(category.id ?? category.name ?? '');
    const body = perCategory.get(cid) as Record<string, unknown> | undefined;
    if (!body) continue;
    for (const [block, group] of [
      ['templates', 'template'],
      ['popular_models', 'popular'],
    ] as const) {
      const entries = body[block];
      if (!entries || typeof entries !== 'object') continue;
      for (const [id, meta] of Object.entries(entries as Record<string, unknown>)) {
        if (seen.has(`${group}:${id}`)) continue;
        seen.add(`${group}:${id}`);
        const m = (meta ?? {}) as Record<string, unknown>;
        out.push({
          id,
          group,
          category: cid,
          label: String(m.name ?? id),
          description: String(m.description ?? ''),
        });
      }
    }
  }
  return out;
}

/**
 * A canonical block, in the shape the code generator accepts.
 *
 * `/api/canonical-templates/{id}/instantiate` answers with a block SPEC — ids, ops and
 * params, plus the entry and exit the canvas wires it in by — not a graph. The canvas
 * turns that into nodes itself (ModelBuilder.instantiateTemplateOnCanvas): op mapped to a
 * layerType id, then inputNode/outputNode/layerNode, at a position. Doing the same here
 * is what makes this a file the generator can write rather than a request it refuses with
 * `nodes.0.type: missing`.
 */
export function placeBlock(spec: unknown): { nodes: unknown[]; edges: unknown[] } {
  const s = (spec ?? {}) as { nodes?: Array<Record<string, unknown>>; edges?: Array<Record<string, unknown>> };
  const opId = (op: unknown) =>
    String(op ?? '').replace(/^(torch\.nn\.|nn\.|torch\.|functional\.|F\.)/, '').toLowerCase();
  const nodes = (s.nodes ?? []).map((n) => {
    const id = opId(n.op);
    const type = id === 'input' ? 'inputNode' : id === 'output' ? 'outputNode' : 'layerNode';
    return {
      id: n.id,
      type,
      position: { x: 0, y: 0 },
      data: {
        layerType: { id, name: String(n.op ?? id), parameters: [] },
        parameters: { ...((n.params as Record<string, unknown>) ?? {}) },
      },
    };
  });
  const edges = (s.edges ?? []).map((e) => ({ id: e.id, source: e.source, target: e.target, data: {} }));
  return { nodes, edges };
}

/** Turn a catalogue entry into a template the scaffolder can write. */
export async function templateFor(
  host: CatalogHost,
  entry: CatalogEntry,
  className: string,
): Promise<ModelTemplate> {
  const origin = host.origin();
  if (!origin) {
    throw new Error('The tensorViz backend is not running yet.');
  }
  const headers = host.headers();
  const route =
    entry.group === 'popular'
      ? `${origin}/api/popular-models/${encodeURIComponent(entry.id)}/instantiate`
      : entry.group === 'canonical'
        ? `${origin}/api/canonical-templates/${encodeURIComponent(entry.id)}/instantiate`
        : `${origin}/api/templates/${encodeURIComponent(entry.id)}/instantiate`;
  const instantiated = (await host.post(route, headers, {
    config: {},
    ...(entry.group === 'canonical' ? {} : { model_name: entry.label || entry.id }),
  })) as { graph?: unknown; input_shape?: number[] };
  const graph =
    entry.group === 'canonical' ? placeBlock(instantiated) : (instantiated?.graph ?? instantiated);
  if (!graph) {
    throw new Error(`${entry.label} produced no graph.`);
  }
  // The class name has to be decided before the code is generated: the generator writes
  // it into the source, and renaming afterwards would leave the file disagreeing with
  // itself. The scaffolder derives it from the file name, so it is passed in.
  const generated = (await host.post(`${origin}/api/models/generate-code`, headers, {
    graph,
    class_name: className,
    // A canonical block is a FRAGMENT with no Input node, so the generator cannot size
    // its example tensor from the graph and falls back to an ImageNet default. It then
    // WRITES that default into the file as `# tensorviz: input=...`, and the trace
    // worker obeys the declaration — so a token-sequence block was traced with
    // [1, 3, 224, 224] and died on its first LayerNorm. The registry already computes
    // the right seed; this carries it across. Only the canonical route returns the
    // field, so the spread is empty for the other groups, whose graphs have a real
    // Input node the generator sizes from.
    ...(instantiated?.input_shape ? { input_shape: instantiated.input_shape } : {}),
  })) as { code?: string; pytorch_code?: string };
  const body = generated?.code ?? generated?.pytorch_code ?? '';
  if (!body.trim()) {
    throw new Error(`${entry.label} generated no code.`);
  }
  return {
    id: entry.id,
    label: entry.label,
    description: entry.description,
    fileStem: stemFor(entry.id),
    // Already carries the class name, so `{cls}` must not be substituted again — and a
    // generated body could legitimately contain braces.
    body,
  };
}

/** Fetch the catalogue. Returns [] when the backend is not up, so the picker still opens. */
export async function fetchCatalog(host: CatalogHost): Promise<CatalogEntry[]> {
  const origin = host.origin();
  if (!origin) return [];
  const headers = host.headers();
  const categories = await host.get(`${origin}/api/templates/categories`, headers);
  const list = (categories as { categories?: Array<Record<string, unknown>> })?.categories ?? [];
  const perCategory = new Map<string, unknown>();
  for (const category of list) {
    const cid = String(category.id ?? category.name ?? '');
    if (!cid) continue;
    try {
      perCategory.set(
        cid,
        await host.get(`${origin}/api/templates/categories/${encodeURIComponent(cid)}`, headers),
      );
    } catch {
      // One unreadable category must not cost the user the other four.
    }
  }
  const entries = entriesFrom(categories, perCategory);

  // The 306 canonical blocks. Listed separately because they are not in the category
  // catalogue at all — the block that claims to list them names 15 ids the registry has
  // never had — and because the registry is the only thing that knows what exists.
  try {
    const listed = (await host.get(`${origin}/api/canonical-templates`, headers)) as {
      templates?: Array<{ id?: string; name?: string; family?: string }>;
    };
    for (const t of listed?.templates ?? []) {
      if (!t?.id) continue;
      entries.push({
        id: t.id,
        group: 'canonical',
        category: t.family || 'blocks',
        label: t.name || t.id,
        description: 'Block',
      });
    }
  } catch {
    // An older backend has no listing. The 73 architectures above still stand.
  }
  return entries;
}

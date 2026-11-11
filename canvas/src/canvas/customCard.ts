/**
 * What a custom-code card says, decided once (task.md M6).
 *
 * `custom_code` and `python` are the two placeholder ops: code that is real and
 * that the op registry deliberately does not model. Their cards are therefore
 * drawn from the provenance keys the projection carries (task.md §6.1) rather
 * than from an OpSpec's parameter list — and two files have to agree on what
 * those keys produce: LayerShapeUtil, which renders the card, and cardMetrics,
 * which computes the height tldraw's selection box, resize handles and hit area
 * are built from. A row rendered in one and not counted in the other is a card
 * whose lower half cannot be clicked, which is why this lives in its own module
 * and both import it instead of each deriving the same thing twice.
 *
 * Deliberately free of UI dependencies, for the same reason cardMetrics is: it is pulled
 * into Jest suites that cannot transform tldraw's ESM-only dependency.
 */
import { sourceOperationDisplayName } from './operationDisplayName';

/** The two placeholder ops (backend/core/op_registry.py, "code we do not model"). */
export type CustomOp = 'custom_code' | 'python';

export interface CustomCardModel {
  op: CustomOp;
  /** Class name or result operation; source text when parser evidence is absent. */
  title: string;
  /** Exact callable under a readable operation title, never a generated node ID. */
  callable?: string;
  /** One line of code under the title, or '' when it would only repeat the title. */
  snippet: string;
  /** `file:line` where the class is defined, or '' when the projection carried none. */
  location: string;
  /** The research kind — dataset, trainer, evaluator — when this card came from a
   *  project graph rather than a model's forward pass. '' on the per-class tier. */
  kind?: string;
  /** Why the graph decided that, in the words the canvas shows on hover:
   *  "subclasses nn.Module", "calls .backward() and an optimizer .step()".
   *
   *  A guess the user cannot interrogate is a guess they cannot correct, and on a
   *  read-only surface correcting it — by writing an SDK declaration — is the only
   *  input channel they have. */
  reason?: string;
  /** `static` was read from the code; `inferred` is a rule guessing; `declared` is
   *  the user's own word and outranks both. */
  provenance?: string;
  /** Whether this node can be validated, decided by the index before anything runs
   *  (step 5.9).
   *
   *  On the card rather than behind the button, because a refusal that arrives
   *  after the press has already wasted it. `unsupported` draws nothing at all: a
   *  dataset has no forward pass, and a row saying so on every such card would be
   *  noise that teaches the reader to stop reading the row. */
  validatable?: { state: 'ready' | 'needs' | 'unsupported'; detail: string; remedy: string };
  /** Opening a project frame reads source without executing the model. */
  projectExpandable?: boolean;
}

const SNIPPET_MAX = 52;

/** Names are identities. Layout wraps them; it must never discard their suffix. */
const identity = (value: unknown): string => typeof value === 'string' ? value.trim() : '';

/** A card is one line tall per row: a statement spanning lines is shown as its
 *  first line, and an over-long one is cut where it stops being readable. */
function oneLine(value: unknown, max: number): string {
  if (typeof value !== 'string') return '';
  const first = value.split('\n')[0].trim();
  if (first.length <= max) return first;
  return `${first.slice(0, max - 1)}…`;
}

/** `models/blocks.py:5`. The leading directories are dropped past the last two
 *  segments: a card is ~200px wide and an absolute path pushes the line number —
 *  the half a person actually navigates by — off the end of the row. */
function locationOf(parameters: Record<string, unknown>): string {
  const file = parameters._source_file ?? parameters._decl_file;
  const line = parameters._source_line ?? parameters._decl_line ?? parameters._lineno;
  if (typeof file !== 'string' || !file) return '';
  const segments = file.replace(/\\/g, '/').split('/').filter(Boolean);
  const shown = segments.slice(-2).join('/');
  return typeof line === 'number' && line > 0 ? `${shown}:${line}` : shown;
}

/** What a container of modules is actually holding.
 *
 * `nn.ModuleList([CrossAttnBlock(dim, heads) for _ in range(2)])` is titled `ModuleList`
 * by its own class, which tells a reader nothing they could not see — the interesting
 * fact is that it holds two CrossAttnBlocks. The projection already records them in
 * `_modules`, so when every entry is the same class the card says so. Mixed contents
 * fall through to the container's own class name rather than naming one arbitrary member.
 */
function containerTitle(parameters: Record<string, unknown>): string {
  const modules = parameters._modules;
  if (!Array.isArray(modules) || modules.length === 0) return '';
  const names = modules.map(entry =>
    entry && typeof entry === 'object' && typeof (entry as { op?: unknown }).op === 'string'
      ? (entry as { op: string }).op : '');
  if (names.some(name => !name) || new Set(names).size !== 1) return '';
  return names.length > 1 ? `${names[0]} ×${names.length}` : names[0];
}

/** The index's verdict on this node, when it carried one. Read defensively: the
 *  card is drawn from whatever the projection happened to send, and a malformed
 *  verdict must leave the card intact rather than blank it. */
function validatableOf(parameters: Record<string, unknown>): CustomCardModel['validatable'] {
  const raw = parameters._validatable as
    | { state?: unknown; detail?: unknown; remedy?: unknown }
    | undefined;
  const state = raw?.state;
  if (state !== 'ready' && state !== 'needs' && state !== 'unsupported') {
    return undefined;
  }
  return {
    state,
    detail: typeof raw?.detail === 'string' ? raw.detail : '',
    remedy: typeof raw?.remedy === 'string' ? raw.remedy : '',
  };
}

/**
 * The card for one node, or null when the node is an ordinary modelled layer.
 *
 */
export function customCardModel(
  layerTypeId: string | undefined,
  parameters: Record<string, unknown> | undefined,
): CustomCardModel | null {
  if (layerTypeId !== 'custom_code' && layerTypeId !== 'python') return null;
  const params = parameters || {};
  const rawCall = oneLine(params._raw_call, SNIPPET_MAX);

  if (layerTypeId === 'python') {
    const result = params._source_result_call as { name?: unknown; target?: unknown } | undefined;
    const name = identity(result?.name);
    const readable = sourceOperationDisplayName(result?.target);
    return {
      op: 'python',
      title: readable || name || identity(params._raw_call) || 'Python',
      callable: readable ? oneLine(result?.target, SNIPPET_MAX) : undefined,
      // Keep nested argument preparation in the source detail. A flat list of
      // calls is neither the purpose of this statement nor its dataflow.
      snippet: name ? rawCall : '',
      location: locationOf(params),
    };
  }

  // A project-graph card. It carries its own label and the rule that classified
  // it; without this branch every research node rendered as an unlabelled
  // "Custom Code" box, because `_class` — the per-class tier's title — is absent.
  //
  // `_kind` is overloaded, which is why the fallbacks below are ordered rather than
  // a single `|| kind`. A research node's `_kind` is its role — dataset, trainer —
  // and reads well as a title. A per-class node's `_kind` is the literal string
  // "class", and EVERY per-class custom card carries it with no `_label`, so the
  // generic word won over the real name: a model whose blocks are
  // `nn.ModuleList([CrossAttnBlock(...) ...])`, `RotaryEmbedding(...)` and
  // `nn.ModuleList([DiTBlock(...) ...])` drew three identical cards all saying
  // "class". The name was in `_class` the whole time.
  const kind = typeof params._kind === 'string' ? params._kind : '';
  if (kind) {
    const label = typeof params._label === 'string' ? params._label : '';
    return {
      op: 'custom_code',
      title: identity(label) || containerTitle(params) || identity(params._class) || kind,
      // The reason is the snippet row: it is the useful thing to read under the
      // name, and a research card has no single statement to quote anyway.
      snippet: oneLine(params._reason, SNIPPET_MAX),
      location: locationOf(params),
      kind,
      reason: typeof params._reason === 'string' ? params._reason : '',
      provenance: typeof params._provenance === 'string' ? params._provenance : 'static',
      validatable: validatableOf(params),
      projectExpandable: params._expandable === true,
    };
  }

  const cls = typeof params._class === 'string' ? params._class : '';
  return {
    op: 'custom_code',
    title: cls || 'Custom Code',
    snippet: rawCall,
    location: locationOf(params),
  };
}

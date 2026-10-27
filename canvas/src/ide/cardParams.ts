/**
 * What a node card should actually show in its parameter list.
 *
 * task.md P0 hygiene, and §3.8's first three carried-in defects. The projection puts three
 * kinds of thing into `data.parameters`, and only one of them is a parameter:
 *
 *   1. real constructor arguments — `in_features: 784`, `p: 0.1`. These are the point.
 *   2. provenance the ingest tier attaches — `_shape_provenance`, `_out_shape`. These are
 *      how the graph knows what it knows, and they were being drawn as if the user had
 *      passed them to `nn.Linear`.
 *   3. `_repeat_count`, which is structural: it says this card stands for N identical
 *      blocks. Drawn as a parameter it read as `_repeat_count: 2`, which looks like a
 *      constructor argument nobody wrote.
 *
 * The rule is the leading underscore, matching Python's own convention for "internal", and
 * it is applied here rather than at the source because the same keys are what the canvas,
 * the validator and the AI context all read. Hiding them from the *drawing* is the fix; the
 * data stays intact.
 */

/** Keys the projection attaches that are not constructor arguments. */
export const PROVENANCE_PREFIX = '_';

/** The repeat marker, promoted out of the parameter list into a badge. */
export const REPEAT_KEY = '_repeat_count';

/** A container's member list, summarised into `contents` rather than drawn as a row. */
export const MEMBERS_KEY = '_modules';

export interface CardParams {
  /** `[key, value]` pairs to render as the parameter list, in the order given. */
  readonly entries: ReadonlyArray<readonly [string, unknown]>;
  /** `"4 × Linear"` for a container whose members are all the same class, else undefined.
   *
   * A `nn.ModuleList` is read by the parser as `_modules: [{op, params}, …]`, and the card
   * rendered that list with `String(value)` — so the first thing a user saw on opening a
   * model with a `ModuleList` was `modules  [object Object],[object Object],…`. The list
   * is not a constructor argument the user typed and it is not drawable as one, but what
   * it *says* — how many of what — is exactly what the card should carry until the trace
   * arrives and folds the block properly. */
  readonly contents?: string;
  /** `"×2"` when this card stands for more than one identical block, else undefined.
   *
   * Undefined rather than `"×1"`: a card that stands for exactly one block is the normal
   * case, and a badge on every card would say nothing while costing the space that makes
   * the badge legible where it matters. */
  readonly repeat?: string;
}

function repeatCountOf(parameters: Record<string, unknown>): number | undefined {
  const raw = parameters[REPEAT_KEY];
  const count = typeof raw === 'string' ? Number(raw) : raw;
  if (typeof count !== 'number' || !Number.isFinite(count) || count <= 1) {
    return undefined;
  }
  return count;
}

/**
 * Split a projection's parameters into what to draw and what to badge.
 *
 * Pure, and deliberately so: it takes the parameter bag and returns what the card needs,
 * with no React and no knowledge of the renderer, which is what lets it be tested against
 * a table of real projections rather than through a rendered card.
 */
export function cardParams(parameters: Record<string, unknown> | undefined): CardParams {
  if (!parameters) {
    return { entries: [] };
  }
  const count = repeatCountOf(parameters);
  const entries = Object.entries(sourceParameterValues(parameters)).filter(
    ([key, value]) => !key.startsWith(PROVENANCE_PREFIX) && isDrawableValue(value),
  );
  const contents = describeMembers(parameters);
  const result: { entries: typeof entries; repeat?: string; contents?: string } = { entries };
  if (count !== undefined) result.repeat = `×${count}`;
  if (contents !== undefined) result.contents = contents;
  return result;
}

/**
 * Can this value be written on one line of a card?
 *
 * A parameter row is `key  value`, and `String(value)` of anything structured is either
 * `[object Object]` or a wall of JSON that pushes the card past the height the layout
 * reserved for it. Scalars and flat scalar tuples draw; objects are summarised or dropped.
 * `null` draws as "None", which is a real and readable Python value.
 */
function isDrawableValue(value: unknown): boolean {
  if (Array.isArray(value)) return value.every(item => item === null || ['string', 'boolean', 'number'].includes(typeof item));
  return value === null || (typeof value !== 'object' && typeof value !== 'function');
}

/** Source expressions are displayed and compared, never handed to a tensor constructor. */
export function sourceParameterValues(parameters: Record<string, unknown>): Record<string, unknown> {
  const expressions = parameters._source_arguments;
  return { ...parameters, ...(expressions && typeof expressions === 'object' && !Array.isArray(expressions)
    ? Object.fromEntries(Object.entries(expressions).filter(([key, value]) => !key.startsWith('_') && typeof value === 'string')) : {}) };
}

/** A member of a container, as the parser reads it out of the constructor call. */
interface ParsedMember {
  op?: unknown;
}

/**
 * `"4 × Linear"` when every member of a container is the same class, `"4 layers"` when
 * they differ, undefined when there is no member list to describe.
 *
 * Deliberately not a `repeat` badge: the badge means "this card stands for N identical
 * blocks", and on a container card the header still says `ModuleList`, so `×4` there would
 * read as four ModuleLists. Saying what the members are is the part that is true of the
 * container itself.
 *
 * The key is `_modules`, under the same prefix as the rest of the provenance above. It was
 * `modules` — `nn.ModuleList`'s own first constructor argument — which put a list of
 * `{op, params}` dicts in the one bag the backend forwards to a torch constructor, and a
 * correct model opened on the canvas wearing `ModuleList — build error: dict is not a
 * Module subclass`. The prefix is what says "display data" to every filter on both sides.
 */
function describeMembers(parameters: Record<string, unknown>): string | undefined {
  const members = parameters[MEMBERS_KEY];
  if (!Array.isArray(members) || members.length === 0) {
    return undefined;
  }
  const ops = members.map((member) => {
    const op = (member as ParsedMember | null)?.op;
    return typeof op === 'string' && op ? op.split('.').pop() ?? '' : '';
  });
  const first = ops[0];
  const uniform = first !== '' && ops.every((op) => op === first);
  return uniform ? `${members.length} × ${first}` : `${members.length} layers`;
}

/**
 * Whether a node's shapes were measured, inferred or never reached.
 *
 * Reading `_shape_provenance` here rather than in the card keeps the one place that knows
 * the key's spelling next to the one place that hides it — the pair drifted apart once
 * already, which is how the key ended up rendered as a parameter in the first place.
 *
 * Canonical source correspondence distinguishes missing calls, capture limits and
 * unresolved mappings. `unreached` is retained only for legacy graph readers.
 */
export type ShapeProvenance = 'static' | 'observed' | 'declared' | 'unreached' | 'failed'
  | 'not_observed' | 'capture_limited' | 'static_only' | 'unresolved' | 'state';

export function shapeProvenance(
  parameters: Record<string, unknown> | undefined,
): ShapeProvenance | undefined {
  const raw = parameters?.['_shape_provenance'];
  return ['static', 'observed', 'declared', 'unreached', 'failed', 'not_observed', 'capture_limited', 'static_only', 'unresolved', 'state'].includes(raw as string)
    ? raw as ShapeProvenance
    : undefined;
}

/**
 * Whether anybody chose the tensor the run measured.
 *
 * Every shape on the canvas is a measurement of one input. When the file says
 * `# tensorviz: input=2,8` that input is the model the user meant; when it says
 * nothing, the backend seeds a tensor off the first layer's width or brute-forces
 * candidates until one survives a forward pass. Both come back `tier: "real"` with
 * the same node count and the same candidate count, so the two were indistinguishable
 * — and the cards presented a measurement of an input nobody had picked as fact.
 *
 * Mirrors `IngestDiagnostics.input_provenance` (`services/ingest/spec.py`), read here
 * for the same reason `shapeProvenance` is: one place knows the key's spelling, and it
 * is the place that hides the key from the drawn parameter list.
 */
export type InputProvenance =
  | 'declared'
  | 'explicit'
  | 'seed-from-first-layer'
  | 'search-candidate'
  | 'failed-candidate'
  | 'assumed-default';

const INPUT_PROVENANCES: readonly InputProvenance[] = [
  'declared', 'explicit', 'seed-from-first-layer', 'search-candidate', 'failed-candidate', 'assumed-default',
];

export function inputProvenance(
  parameters: Record<string, unknown> | undefined,
): InputProvenance | undefined {
  const raw = parameters?.['_input_provenance'];
  return INPUT_PROVENANCES.includes(raw as InputProvenance) ? (raw as InputProvenance) : undefined;
}

/** What to say about a guessed input, at the two lengths the product has room for. */
export interface InputProvenanceNote {
  /** One word, for the Input card's value slot. */
  readonly label: string;
  /** One sentence, for the surfaces with room — the Validate verdict. */
  readonly detail: string;
}

const DECLARE = 'Declare one with `# tensorviz: input=` to measure the model you mean.';

/**
 * What the Input card and the Validate verdict should say about where the traced
 * tensor came from, or undefined when there is nothing worth saying.
 *
 * A stated input is unremarkable and gets no note: a note on every card says
 * nothing while costing the space that makes a note legible where it matters —
 * the same reasoning that keeps `×1` off the repeat badge. A guessed one is
 * marked, plainly and without apology, because the numbers beside it are about to
 * be believed.
 */
export function inputProvenanceNote(
  parameters: Record<string, unknown> | undefined,
): InputProvenanceNote | undefined {
  switch (inputProvenance(parameters)) {
    case 'seed-from-first-layer':
      return {
        label: 'guessed',
        detail: `The input was read off the first layer's width, not from your source. ${DECLARE}`,
      };
    case 'search-candidate':
      return {
        label: 'guessed',
        detail: `The input is the first shape that survived a forward pass, not one you chose. ${DECLARE}`,
      };
    case 'failed-candidate':
      return {
        label: 'failed guess',
        detail: 'This automatic input failed during the forward pass. The failure may be caused by the guessed input. Save explicit inputs in a tensorviz.toml model profile and retry.',
      };
    case 'assumed-default':
      return {
        label: 'default',
        detail: `The input is this architecture's built-in default, not one you chose. ${DECLARE}`,
      };
    default:
      // `declared` and `explicit` were both chosen by a person; so is an input the
      // backend never reported on, which is every graph read from the source alone.
      return undefined;
  }
}

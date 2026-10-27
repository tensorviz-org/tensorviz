/**
 * The Properties panel edits a node's parameters as strings. This is the boundary
 * where real values become those strings and come back, and it is load-bearing in
 * a way it was not before: the commit is now automatic (debounced, no Apply), so
 * anything that does not survive format -> parse is rewritten silently.
 *
 * Extracted from BuilderPageV2.tsx so it can be unit-tested. That file pulls in
 * tldraw and cannot be imported under the CRA jest transform, which is why this
 * conversion had no tests at all — and why a value-inverting bug lived in it. The
 * call sites keep their exact text (`formatParamValue(key, value)`), because
 * source-contract tests read them.
 *
 * ── the bug this exists to close ─────────────────────────────────────────────
 * `formatValue` spells booleans the Python way, 'True' / 'False' — right for a
 * PyTorch tool, and the same spelling the card and the code snippet use. But
 * `parseParameterInput` only recognised the lowercase form, so a real boolean came
 * back as the STRING 'True'. Two consequences, both silent:
 *
 *   - the panel's dropdown compared against 'true' and so displayed every enabled
 *     flag as False — and could not be corrected by hand, because a <select> fires
 *     onChange only when the DOM selection actually moves;
 *   - `bias="False"` is a non-empty string, and a non-empty string is truthy in
 *     Python. Measured: nn.Linear(128, 64, bias=False) has no bias and 8192
 *     parameters; bias='False' HAS a bias and 8256. Exported code and the built
 *     model both turn the bias back on, while shape inference reports the graph
 *     healthy because _shape_linear never looks at `bias`.
 *
 * A sweep of 476 real (key, value) pairs — every catalogue default plus 37 live
 * template instantiations — found nine more kinds that did not survive: `null`
 * (`stride=None` is the torch default and arrives on every code import), arrays
 * holding null (`AdaptiveAvgPool2d((None, 7))` formatted to the unparseable
 * '[, 7]'), floats at or below 1e-7 (String() switches to exponent form and the
 * numeric regex had no exponent branch), leading-dot decimals ('.5', which a
 * number input happily produces), numeric-looking STRINGS that are documented as
 * comma-separated ('512' for normalized_shape), and strings that merely look like
 * JSON ('[3, 4, 5]' from a template).
 *
 * ── the shape of the fix ─────────────────────────────────────────────────────
 * Two halves, and the first is the one that matters:
 *
 *  1. `commitParameters` writes back only the keys the user actually MOVED, and
 *     takes every other key verbatim from the node. Guessing a type from a string
 *     is only ever needed for a field someone typed in; for the rest there is a
 *     real value already, so there is nothing to guess and nothing to lose. That
 *     closes the whole class at once rather than patching kinds one at a time.
 *  2. `parseParameterInput` accepts what `formatParamValue` writes, so a field
 *     that IS edited round-trips, and a graph already holding the string 'True'
 *     heals the next time that key is touched.
 */

import { formatInputShape } from '../../utils/inputShape';

/** Python spelling, because the card, the dropdown labels and the code snippet all use it. */
export function formatValue(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (Array.isArray(value)) return `[${value.join(', ')}]`;
  return String(value);
}

/**
 * An Input node's `shape` must show its canonical comma form, not the bracketed array
 * form above. Bracketing it was ambiguous to read AND broke editing: pressing Apply wrote
 * the literal string "[1, 64, 48]" back into the param, which then parses to nothing and
 * silently falls back to the default shape.
 */
export function formatParamValue(key: string, value: unknown): string {
  if (key === 'shape') return formatInputShape(value);
  // Element-wise JSON, not formatValue's join(): join renders null as an EMPTY
  // string, so an imported `nn.AdaptiveAvgPool2d((None, 7))` showed '[, 7]' — which
  // JSON.parse then rejects, so the commit wrote that text over the tuple. Numeric
  // arrays are unaffected ([7, 7] reads identically).
  if (Array.isArray(value)) return `[${value.map((item) => JSON.stringify(item)).join(', ')}]`;
  return formatValue(value);
}

export function parseParameterInput(raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === '') return '';
  // Case-insensitive: format writes 'True'/'False', the <select> writes
  // 'true'/'false', and a user may type either. Accepting only one of the two was
  // the value-inverting bug described at the top of this file.
  if (/^true$/i.test(trimmed)) return true;
  if (/^false$/i.test(trimmed)) return false;
  // `stride=None` is the documented torch default and every code import carries it.
  if (trimmed === 'null') return null;
  // Leading dot ('.5', '-.5') because a number input produces it, and an exponent
  // group because String(1e-8) is '1e-8'. Without these, `nn.Dropout(p='.5')` raises
  // a TypeError inside shape inference while Export still succeeds — the worst split.
  // Non-finite results are rejected rather than stored: '1e999' would otherwise commit
  // as Infinity and re-format to the string 'Infinity' on the next seed.
  if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return numeric;
  }
  if ((trimmed.startsWith('[') && trimmed.endsWith(']')) || (trimmed.startsWith('{') && trimmed.endsWith('}'))) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return raw;
    }
  }
  return raw;
}

/**
 * The parameter map to commit: the node's own values, with only the keys whose
 * string form actually changed replaced by the parsed draft.
 *
 * Three defects fall out of this rather than being patched individually.
 *
 *  - A key the user never touched keeps its real value, so no amount of typing in
 *    one field can change the type of another. That is what makes the nine lossy
 *    kinds above unreachable in the common case.
 *  - `_`-prefixed keys survive. The panel deliberately hides them (they are
 *    internal render props), but hiding them in the editor is not a licence to
 *    delete them from the graph — and the commit replaces the whole map, so the
 *    first keystroke on a template node used to drop `_shape`, `_subtitle` and
 *    `_accent`, which the backend authors and its own tests assert are there.
 *  - The draft is re-seeded only when the SELECTION changes, so a graph edit from
 *    somewhere else — the toolbar's Undo, an applied AI operation — leaves the
 *    panel showing values that are no longer in the graph. Writing back only what
 *    moved means the next keystroke can no longer resurrect them.
 *
 * @param current  the node's parameters right now
 * @param baseline the draft as it was seeded or last committed
 * @param draft    the draft as it is now
 */
export function commitParameters(
  current: Record<string, unknown>,
  baseline: Record<string, string>,
  draft: Record<string, string>,
): Record<string, unknown> {
  const next: Record<string, unknown> = { ...current };
  Object.entries(draft).forEach(([key, raw]) => {
    // A key present in the draft but not in the baseline is new — a field the panel
    // rendered from the catalogue for a parameter the node did not carry — so it
    // counts as moved.
    if (Object.prototype.hasOwnProperty.call(baseline, key) && baseline[key] === raw) return;
    next[key] = parseParameterInput(raw);
  });
  return next;
}

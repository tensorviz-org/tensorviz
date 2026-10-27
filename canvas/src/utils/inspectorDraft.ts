/**
 * What the Properties panel is allowed to put in its editable draft.
 *
 * Lives here rather than inside BuilderPageV2 so it can be tested directly:
 * BuilderPageV2 pulls in the canvas and is not mountable or importable under the
 * CRA jest transform, which is why this rule had only ever been pinned as source
 * text — twice, once for the seeding effect and once, through a 900-character
 * backward window off the first literal "Reset" in the file, for the Reset chip.
 * Two call sites and one rule, so the rule is now one function that both call.
 *
 * The rule: underscore-prefixed keys are internal render props (_accent is a CSS
 * variable, _subtitle the card's caption, _shape the inferred shape) and are not
 * parameters. The Input node exposed all three as editable rows, and editing
 * _shape reported success while changing nothing, because the real field is
 * `shape` and _shape is derived from inference.
 *
 * It filters on the prefix ALONE. An allow-list would drop the layer parameters
 * the frontend schema does not know about, which is most of them.
 */
import { formatParamValue } from '../components/ui2/parameterValues';

/** The editable draft for a node's parameters: real parameters only, as strings. */
export function inspectorDraftParameters(
  parameters: Record<string, unknown> | undefined | null,
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(parameters || {})
      .filter(([key]) => !key.startsWith('_'))
      .map(([key, value]) => [key, formatParamValue(key, value)])
  );
}

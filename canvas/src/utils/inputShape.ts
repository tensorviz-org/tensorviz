/**
 * One parser and one formatter for an Input node's `shape` param.
 *
 * The canonical stored form is a COMMA STRING ("1,64,48") — what ingest writes, what the
 * templates carry, and what `operation_resolver._shape_param` now emits. But three
 * surfaces used to read the value raw and each broke differently when a list arrived
 * instead: the inspector field showed "[1, 64, 48]", the canvas card showed "16448"
 * (React concatenates an array's children), and Export PyTorch Code died with
 * "inputNode.data.parameters.shape.split is not a function".
 *
 * Parsing stays tolerant of both forms on purpose: graphs saved before the backend fix
 * still hold arrays, and a stored document is not something a reader gets to reject.
 */

export const DEFAULT_INPUT_SHAPE: number[] = [1, 3, 224, 224];

/** The dims a value actually carries, or null when it carries none.
 *
 *  "" is the case that matters: an empty string is not undefined, so a caller that
 *  only guards `=== undefined` falls straight through to the default and states it
 *  as fact. Everything that has to decide whether a shape is REAL reads this. */
export const readInputShape = (rawShape: unknown): number[] | null => {
  const dims = Array.isArray(rawShape)
    ? rawShape.map((value) => Number(value))
    : String(rawShape ?? '')
        .split(',')
        .map((part) => parseInt(part.trim(), 10));

  const usable = dims
    .filter((value) => Number.isFinite(value) && value > 0)
    .map((value) => Math.floor(value));

  return usable.length > 0 ? usable : null;
};

/** Accepts "1,64,48", [1, 64, 48], ["1", 64] or nonsense; always returns usable dims.
 *
 *  For the callers that need A tensor to write code with — Export PyTorch Code, the
 *  shape-inference payload — where falling back to a default beats refusing. Anything
 *  that SHOWS the value to a reader wants `readInputShape`/`displayInputShape`. */
export const parseInputShape = (rawShape: unknown): number[] =>
  readInputShape(rawShape) ?? [...DEFAULT_INPUT_SHAPE];

/** The canonical stored form. Use this whenever writing the param back. */
export const formatInputShape = (rawShape: unknown): string =>
  parseInputShape(rawShape).join(',');

/** Human-facing form for a card, matching the existing canvas style: "(1, 64, 48)",
 *  or null when the value carries no shape at all.
 *
 *  Null rather than the default, and deliberately NOT parseInputShape's behaviour: a
 *  code generator needs *a* tensor and [1,3,224,224] is a defensible one to pick, but
 *  a CARD is a claim about THIS model. Printing (1, 3, 224, 224) for a character LM,
 *  in the same slot and typography as the measured numbers, while the card beside it
 *  honestly says "Not recorded" and the header says "static outline", is the one
 *  number a reader is most likely to trust and act on — and nobody supplied it. */
export const displayInputShape = (rawShape: unknown): string | null => {
  const dims = readInputShape(rawShape);
  return dims ? `(${dims.join(', ')})` : null;
};

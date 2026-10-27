/**
 * One derived line describing what a node actually does, for the card to show
 * above its parameter rows.
 *
 * A Slice card renders `dim 1`, `start 0`, `length 16` — three numbers that never
 * say "slice". The reader has to reassemble the operation in their head, on every
 * card, at every zoom level. So the card earns a single line in the notation the
 * user was thinking in: `x[:, 0:16]`.
 *
 * A pure module rather than a function on LayerType: `layerType` is serialised
 * into `node.data` and persisted, and functions do not survive the round trip
 * through JSON.
 *
 * Returns null for the many ops that read perfectly well as plain rows — the
 * caller renders nothing in that case.
 */

/** Params arrive from number inputs, so a value may be a numeric string. */
const num = (value: unknown): number | null => {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

/** A comma string or a real list, as the Input node's shape param does it. */
const intList = (value: unknown): number[] | null => {
  if (Array.isArray(value)) {
    const out = value.map(num);
    return out.every((v): v is number => v !== null) ? out : null;
  }
  if (typeof value === 'string') {
    const parts = value.replace(/[()[\]]/g, '').split(',').map((p) => p.trim()).filter(Boolean);
    if (parts.length === 0) return null;
    const out = parts.map(num);
    return out.every((v): v is number => v !== null) ? out : null;
  }
  return null;
};

function narrowSummary(params: Record<string, unknown>): string | null {
  const dim = num(params.dim);
  const start = num(params.start);
  const length = num(params.length);
  if (dim === null || start === null || length === null) return null;
  if (!Number.isInteger(dim) || !Number.isInteger(start) || !Number.isInteger(length)) return null;
  if (length < 1 || start < 0) return null;

  // A negative dim counts from the right, so how many axes precede it is unknown
  // here — ellipsis is the only honest rendering. Spelling ':' the wrong number
  // of times would be exactly the kind of confident guess this feature removes.
  const lead = dim < 0 ? '..., ' : Array(dim).fill(':').join(', ') + (dim > 0 ? ', ' : '');
  return `x[${lead}${start}:${start + length}]`;
}

function interpolateSummary(params: Record<string, unknown>): string | null {
  const mode = typeof params.mode === 'string' && params.mode ? params.mode : null;
  const size = intList(params.size);
  const scale = num(params.scale_factor);
  // size wins when both are set, matching how the Inspector clears the other one.
  if (size) return `→ (${size.join(', ')})${mode ? ` ${mode}` : ''}`;
  if (scale !== null) return `${scale}×${mode ? ` ${mode}` : ''}`;
  return null;
}

function reshapeSummary(params: Record<string, unknown>): string | null {
  const shape = intList(params.shape);
  return shape ? `→ (${shape.join(', ')})` : null;
}

export function opSummary(opId: string, params: Record<string, unknown> | undefined): string | null {
  if (!params) return null;
  switch (opId) {
    case 'narrow':
      return narrowSummary(params);
    case 'interpolate':
    case 'upsample':
      return interpolateSummary(params);
    case 'reshape':
      return reshapeSummary(params);
    default:
      return null;
  }
}

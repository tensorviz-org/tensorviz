/**
 * What a Slice means before the user has said anything.
 *
 * Two decisions, and both exist to make the default APPLY on its own rather than
 * sit there waiting to be configured:
 *
 *  - the LAST axis. `dim: 1` (the old catalogue default) is the channel axis of a
 *    4D feature map and the feature axis of a 2D activation — the same number
 *    meaning two different things. `-1` is the axis people mean when they say "the
 *    features", at every rank, and it is what torch.narrow itself accepts, so
 *    nothing downstream has to resolve it.
 *  - the FULL extent of that axis. A default of `length: 1` is a real slice, which
 *    is the problem: applying it automatically would silently reshape the tensor
 *    and break the next layer, so the user's first sight of the feature would be
 *    an in_features mismatch they did not ask for. Selecting the whole axis
 *    changes nothing, stays valid, and opens the range control fully selected —
 *    the drag handles are then the obvious next move.
 *
 * The extent needs the upstream shape, which is known only after inference has
 * run. When it has not, fall back to 1 rather than guessing a width.
 */
export interface SliceDefaults {
  dim: number;
  start: number;
  length: number;
}

/** `inputShape` is the shape arriving at the Slice — the upstream node's output. */
export function sliceDefaults(inputShape?: number[]): SliceDefaults {
  const extent = Array.isArray(inputShape) && inputShape.length > 1
    ? inputShape[inputShape.length - 1]
    : undefined;
  return {
    dim: -1,
    start: 0,
    length: Number.isInteger(extent) && (extent as number) > 0 ? (extent as number) : 1,
  };
}

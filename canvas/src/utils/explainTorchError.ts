/**
 * Plain language for the torch errors a beginner actually hits.
 *
 * A validation failure surfaces the raw message from PyTorch — accurate, and
 * useless to someone who has not seen it before:
 *
 *   "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of
 *    size: [1, 32]"
 *   "mat1 and mat2 shapes cannot be multiplied (1x788544 and 512x256)"
 *
 * Both say the same thing in the same breath — "a Conv2d needs image-shaped data
 * and you gave it a flat vector", "the Linear layer expects 512 numbers and 788,544
 * arrived" — and neither says it in words. The raw message is KEPT, because it is
 * what a user searches for and what an experienced user reads first; the sentence
 * is added ahead of it.
 *
 * Deliberately small. Guessing at an error we do not recognise would be worse than
 * silence, so an unmatched message passes through untouched.
 */

type Explainer = { match: RegExp; explain: (m: RegExpMatchArray) => string };

const EXPLAINERS: Explainer[] = [
  {
    // mat1 and mat2 shapes cannot be multiplied (1x788544 and 512x256)
    match: /mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)/i,
    explain: (m) =>
      `This Linear layer expects ${Number(m[3]).toLocaleString('en-US')} input features, but ` +
      `${Number(m[2]).toLocaleString('en-US')} arrived. Set its in_features to ` +
      `${Number(m[2]).toLocaleString('en-US')}, or change the layers before it so they produce ` +
      `${Number(m[3]).toLocaleString('en-US')}.`,
  },
  {
    // Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 32]
    match: /Expected .*input to (conv\dd), but got input of size: \[([^\]]*)\]/i,
    explain: (m) =>
      `A ${m[1].replace('conv', 'Conv')} needs image-shaped data — batch, channels, height, ` +
      `width — but the shape reaching it is [${m[2]}]. Something upstream has flattened it; a ` +
      `Flatten or a Linear layer before a convolution is the usual cause.`,
  },
  {
    // Given groups=1, weight of size [64, 3, 3, 3], expected input[1, 16, 32, 32] to have 3 channels, but got 16 channels instead
    match: /expected input\[[^\]]*\] to have (\d+) channels, but got (\d+) channels instead/i,
    explain: (m) =>
      `This layer is configured for ${m[1]} input channels but is receiving ${m[2]}. Set its ` +
      `in_channels to ${m[2]} to match the layer feeding it.`,
  },
  {
    match: /Output size is too small|Calculated padded input size per channel/i,
    explain: () =>
      'The image has been shrunk to nothing by this point — repeated pooling or striding has ' +
      'reduced it below the kernel size. Use fewer pooling layers, or start from a larger input.',
  },
  {
    match: /Trying to create tensor with negative dimension (-?\d+)/i,
    explain: (m) =>
      `A layer is configured with ${m[1]}, which is not a size anything can have. Every ` +
      'dimension — in_features, out_features, channels, kernel size — has to be a positive whole number.',
  },
  {
    match: /shape '\[([^\]]*)\]' is invalid for input of size (\d+)/i,
    explain: (m) =>
      `A reshape to [${m[1]}] does not fit ${Number(m[2]).toLocaleString('en-US')} values. The target ` +
      'shape has to multiply out to exactly the number of values arriving.',
  },
];

/** The raw message with a plain-language sentence in front of it, when we have one. */
export function explainTorchError(message: string): string {
  if (!message) return message;
  for (const { match, explain } of EXPLAINERS) {
    const found = message.match(match);
    if (found) return `${explain(found)} (${message})`;
  }
  return message;
}

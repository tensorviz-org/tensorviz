/**
 * The torch class for a layer, or null when we cannot vouch for one.
 *
 * Lives here rather than inside BuilderPageV2 so it can be tested directly:
 * BuilderPageV2 pulls in the canvas and is not mountable or importable under the
 * CRA jest transform, which is why its behaviour had only ever been pinned as
 * source text.
 *
 * The rule is "say nothing rather than guess". The Layer Code preview used to fall
 * through to `nn.${name.replace(/[^a-zA-Z0-9]/g, '')}` for anything it did not
 * recognise, which printed `nn.Input(shape=…)` for a graph endpoint that has no
 * torch class at all, `nn.Output`, and `nn.torchmatmul` for `torch.matmul` (the
 * strip eats the dot). A student who copied that into their own script got an
 * immediate AttributeError and learned something false about PyTorch from a
 * teaching tool.
 *
 * Extending the table is not the fix: the backend op registry holds well over a
 * hundred ops and the frontend cannot see it. Codegen resolves the real class
 * through that registry and refuses unregistered ops, so Export Code is the
 * authority — this is a preview, and it now admits when it does not know.
 */

/** Graph endpoints and module wrappers. None of these is a torch layer. */
const NON_TORCH_LAYER_TYPES = new Set(['input', 'output', 'module']);

/** The ops this preview can name without guessing. */
const KNOWN: Record<string, string> = {
  batchnorm1d: 'nn.BatchNorm1d',
  batchnorm2d: 'nn.BatchNorm2d',
  conv1d: 'nn.Conv1d',
  conv2d: 'nn.Conv2d',
  conv3d: 'nn.Conv3d',
  dropout: 'nn.Dropout',
  flatten: 'nn.Flatten',
  layernorm: 'nn.LayerNorm',
  linear: 'nn.Linear',
  maxpool1d: 'nn.MaxPool1d',
  maxpool2d: 'nn.MaxPool2d',
  relu: 'nn.ReLU',
  sigmoid: 'nn.Sigmoid',
  softmax: 'nn.Softmax',
  tanh: 'nn.Tanh',
};

export function torchClassForLayer(layerType: string, layerName: string): string | null {
  if (NON_TORCH_LAYER_TYPES.has((layerType || '').toLowerCase())) return null;
  // An already-qualified name is used as it stands: `torch.matmul` is not an nn
  // member, and mangling it into `nn.torchmatmul` names nothing.
  if ((layerName || '').includes('.')) return layerName;
  return KNOWN[(layerType || '').toLowerCase()] ?? null;
}

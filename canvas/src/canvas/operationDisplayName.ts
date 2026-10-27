import type { ModelNodeData } from '../types';

const names: Record<string, string> = {
  _native_multi_head_attention: 'Multi-head attention',
  scaled_dot_product_attention: 'Scaled attention',
  _softmax: 'Softmax', _log_softmax: 'Log softmax', _unsafe_view: 'Reshape',
  _to_copy: 'Convert dtype / device', to: 'Convert dtype / device', type_as: 'Match dtype / device',
  add: 'Add', addmm: 'Matrix multiply + add', baddbmm: 'Batch multiply + add',
  addmv: 'Matrix-vector multiply + add', bmm: 'Batch matrix multiply', mv: 'Matrix-vector multiply',
  dot: 'Dot product', einsum: 'Einstein sum', outer: 'Outer product', linear: 'Linear',
  cat: 'Concatenate', stack: 'Stack', clone: 'Clone', copy: 'Copy', detach: 'Detach',
  contiguous: 'Contiguous copy', div: 'Divide', floor_divide: 'Floor divide',
  true_divide: 'Divide', expand: 'Broadcast', expand_as: 'Broadcast to match',
  flatten: 'Flatten', gelu: 'GELU', relu6: 'ReLU6', leaky_relu: 'Leaky ReLU',
  silu: 'SiLU', elu: 'ELU', selu: 'SELU', celu: 'CELU', prelu: 'PReLU',
  _prelu_kernel: 'PReLU', rrelu: 'Randomized ReLU', hardtanh: 'Hard tanh',
  hardswish: 'Hard swish', hardsigmoid: 'Hard sigmoid', log_sigmoid_forward: 'Log sigmoid',
  softplus: 'Softplus', softsign: 'Softsign', mish: 'Mish', glu: 'Gated linear unit',
  layer_norm: 'Layer norm', native_layer_norm: 'Layer norm', rms_norm: 'RMS norm',
  group_norm: 'Group norm', native_group_norm: 'Group norm', instance_norm: 'Instance norm',
  batch_norm: 'Batch norm', native_batch_norm: 'Batch norm',
  _native_batch_norm_legit: 'Batch norm', _native_batch_norm_legit_no_training: 'Batch norm',
  _native_batch_norm_legit_functional: 'Batch norm', cudnn_batch_norm: 'Batch norm',
  miopen_batch_norm: 'Batch norm', _batch_norm_with_update: 'Batch norm',
  _batch_norm_no_update: 'Batch norm', _batch_norm_with_update_functional: 'Batch norm',
  dropout: 'Dropout', native_dropout: 'Dropout', _fused_dropout: 'Dropout',
  feature_dropout: 'Channel dropout', alpha_dropout: 'Alpha dropout',
  matmul: 'Matrix multiply', mean: 'Mean', mm: 'Matrix multiply', mul: 'Multiply',
  permute: 'Permute axes', relu: 'ReLU', reshape: 'Reshape', interpolate: 'Interpolate', rsqrt: 'Reciprocal sqrt',
  sigmoid: 'Sigmoid', slice: 'Slice', select: 'Select index', narrow: 'Narrow slice',
  index: 'Index tensor', index_select: 'Select indices', index_put: 'Write at indices',
  _index_put_impl: 'Write at indices', gather: 'Gather', scatter: 'Scatter',
  scatter_add: 'Scatter + add', scatter_reduce: 'Scatter + reduce',
  masked_fill: 'Fill masked values', masked_select: 'Select masked values',
  masked_scatter: 'Write masked values', where: 'Select by condition',
  embedding: 'Embedding lookup', embedding_bag: 'Pooled embedding lookup',
  _embedding_bag: 'Pooled embedding lookup', _embedding_bag_forward_only: 'Pooled embedding lookup',
  softmax: 'Softmax', log_softmax: 'Log softmax', normalize: 'Normalize', split: 'Split', chunk: 'Chunk',
  split_with_sizes: 'Split', squeeze: 'Squeeze', sub: 'Subtract', sum: 'Sum',
  t: 'Transpose', transpose: 'Transpose', unbind: 'Unbind', unsqueeze: 'Unsqueeze',
  view: 'Reshape', view_as: 'Reshape to match', repeat: 'Repeat', repeat_interleave: 'Repeat elements',
  as_strided: 'Strided view', unfold: 'Sliding windows', im2col: 'Image to columns', col2im: 'Columns to image',
  amax: 'Maximum', amin: 'Minimum', argmax: 'Argmax', argmin: 'Argmin', prod: 'Product',
  cumsum: 'Cumulative sum', cumprod: 'Cumulative product', var: 'Variance', std: 'Standard deviation',
  var_mean: 'Variance + mean', std_mean: 'Standard deviation + mean',
  linalg_vector_norm: 'Vector norm', linalg_matrix_norm: 'Matrix norm',
  pow: 'Power', sqrt: 'Square root', neg: 'Negate', abs: 'Absolute value',
  reciprocal: 'Reciprocal', clamp: 'Clamp', clamp_min: 'Clamp minimum', clamp_max: 'Clamp maximum',
  eq: 'Equal', ne: 'Not equal', lt: 'Less than', le: 'Less than or equal',
  gt: 'Greater than', ge: 'Greater than or equal',
  empty: 'Allocate tensor', empty_like: 'Allocate matching tensor', empty_strided: 'Allocate strided tensor',
  new_empty: 'Allocate tensor', new_empty_strided: 'Allocate strided tensor',
  zeros: 'Zeros', zeros_like: 'Matching zeros', new_zeros: 'Zeros',
  ones: 'Ones', ones_like: 'Matching ones', new_ones: 'Ones',
  full: 'Fill tensor', full_like: 'Matching filled tensor', new_full: 'Fill tensor',
  arange: 'Range', linspace: 'Evenly spaced values', rand: 'Uniform random values',
  rand_like: 'Matching uniform random values', randn: 'Normal random values',
  randn_like: 'Matching normal random values', randint: 'Random integers',
  constant_pad_nd: 'Constant padding', pad: 'Pad',
  cross_entropy_loss: 'Cross-entropy loss', nll_loss: 'Negative log-likelihood loss',
  nll_loss_forward: 'Negative log-likelihood loss', nll_loss2d_forward: 'Negative log-likelihood loss 2D',
  nll_loss2d: 'Negative log-likelihood loss 2D', nll_loss_backward: 'Negative log-likelihood loss backward',
  nll_loss2d_backward: 'Negative log-likelihood loss 2D backward',
  mse_loss: 'Mean squared error loss', l1_loss: 'Absolute error loss',
  smooth_l1_loss: 'Smooth L1 loss', binary_cross_entropy: 'Binary cross-entropy',
  binary_cross_entropy_with_logits: 'Binary cross-entropy with logits',
  _softmax_backward_data: 'Softmax backward', _log_softmax_backward_data: 'Log softmax backward',
  convolution_backward: 'Convolution backward', native_layer_norm_backward: 'Layer norm backward',
  native_batch_norm_backward: 'Batch norm backward', native_group_norm_backward: 'Group norm backward',
};

function familyName(base: string): string | undefined {
  if (/^_scaled_dot_product_(?:flash_attention(?:_for_cpu)?|efficient_attention|attention_math(?:_for_mps)?|cudnn_attention)(?:_backward)?$/.test(base)) {
    return `Scaled attention${base.endsWith('_backward') ? ' backward' : ''}`;
  }
  const pool = /^_?(adaptive_)?(avg|max)_pool([123])d(?:_with_indices)?(_backward)?$/.exec(base);
  if (pool) {
    const reduction = pool[2] === 'avg' ? 'average' : 'max';
    return `${pool[1] ? `Adaptive ${reduction}` : reduction[0].toUpperCase() + reduction.slice(1)} pool ${pool[3]}D${pool[4] ? ' backward' : ''}`;
  }
  const conv = /^(?:_?convolution|conv([123])d|conv_transpose([123])d)$/.exec(base);
  if (conv) return conv[2] ? `Transposed convolution ${conv[2]}D` : `Convolution${conv[1] ? ` ${conv[1]}D` : ''}`;
  const resize = /^_?upsample_(nearest(?:_exact)?|linear|bilinear|bicubic|trilinear)([123])d(_backward)?$/.exec(base);
  if (resize) return `${resize[1] === 'nearest_exact' ? 'Exact nearest' : resize[1][0].toUpperCase() + resize[1].slice(1)} resize ${resize[2]}D${resize[3] ? ' backward' : ''}`;
  const padding = /^(reflection|replication)_pad([123])d(_backward)?$/.exec(base);
  if (padding) return `${padding[1] === 'reflection' ? 'Reflection' : 'Replication'} padding ${padding[2]}D${padding[3] ? ' backward' : ''}`;
  return undefined;
}

/** The parser supplies canonical targets only after proving their Torch import. */
export function sourceOperationDisplayName(target: unknown): string | undefined {
  if (typeof target !== 'string') return undefined;
  const match = /^(?:torch|torch\.nn\.functional)\.([a-zA-Z0-9_]+)$/.exec(target);
  return match && Object.prototype.hasOwnProperty.call(names, match[1]) ? names[match[1]] : undefined;
}

/** Display names only: source/captured targets, identities and connections stay exact. */
export function operationDisplayName(data: Pick<ModelNodeData, 'captureRef' | 'captureLabel' | 'layerType' | 'operationGroup'>): string | null {
  if (data.operationGroup?.kind === 'slice') return 'Slice';
  if ((data.captureRef && data.captureRef.kind !== 'operation') || data.operationGroup || data.layerType.category === 'module') return null;
  // Source projection emits canonical Torch targets, not ATen capture targets.
  // Share their vocabulary without interpreting bare names of user classes.
  const source = /^(?:torch\.(reshape|permute|cat|stack|add|sub|mul|div|matmul|bmm)|(?:torch\.nn\.functional\.)?(interpolate))$/.exec(data.layerType.name ?? '');
  if (source) return names[source[1] ?? source[2]];
  // Legacy/FX/imported graphs may have a target but no semantic capture reference.
  const match = [data.layerType.name, data.captureLabel, data.layerType.id, data.layerType.displayName]
    .map(target => target?.match(/^(?:torch\.ops\.)?aten(?:\.|::)([a-zA-Z0-9_]+)(?:\.[a-zA-Z0-9_]+)?$/))
    .find(Boolean);
  if (!match) return null;
  const inplace = match[1].endsWith('_') && !match[1].endsWith('__');
  const base = inplace ? match[1].slice(0, -1) : match[1];
  const label = (Object.prototype.hasOwnProperty.call(names, base) ? names[base] : undefined) ?? familyName(base) ?? base.replace(/^_+/, '').split('_').filter(Boolean)
    .map((word, index) => index === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word)
    .join(' ');
  return label ? `${label}${inplace ? ' (in place)' : ''}` : null;
}

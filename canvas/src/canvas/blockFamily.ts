import type { ModelEdge, ModelNode, ModelNodeData } from '../types';

/** What an operation does. Separate from selection (sage) and execution status (icons). */
export type BlockFamily = 'convolution' | 'linear' | 'normalization' | 'activation' | 'pooling' | 'dropout'
  | 'recurrent' | 'attention' | 'embedding' | 'padding' | 'resampling' | 'shape' | 'distance' | 'loss' | 'arithmetic';
/** Structural or unclassified blocks: a neutral marker, told apart by glyph and name. */
export type NeutralBlock = 'input' | 'output' | 'container' | 'composite' | 'unknown';
export type BlockKind = BlockFamily | NeutralBlock;

export interface BlockClassification {
  kind: BlockKind;
  /** Family name for the legend and inspector. */
  label: string;
  /** Lazy, Quantized or QAT: the operation keeps its family and gains this label. */
  modifier?: string;
}

export const BLOCK_LABELS: Record<BlockKind, string> = {
  convolution: 'Convolution', linear: 'Linear', normalization: 'Normalization', activation: 'Activation',
  pooling: 'Pooling', dropout: 'Dropout', recurrent: 'Recurrent', attention: 'Attention / Transformer',
  embedding: 'Embedding / sparse', padding: 'Padding', resampling: 'Resampling', shape: 'Shape / rearrangement',
  distance: 'Distance / similarity', loss: 'Loss', arithmetic: 'Arithmetic / merge',
  input: 'Input', output: 'Output', container: 'Container', composite: 'Composite module', unknown: 'Unclassified',
};

/** Legend and swatch order: the families first, neutral blocks last. */
export const BLOCK_ORDER: BlockKind[] = ['convolution', 'linear', 'normalization', 'activation', 'pooling', 'dropout',
  'recurrent', 'attention', 'embedding', 'padding', 'resampling', 'shape', 'distance', 'loss', 'arithmetic',
  'input', 'output', 'container', 'composite', 'unknown'];

const NEUTRAL = new Set<BlockKind>(['input', 'output', 'container', 'composite', 'unknown']);
export const isBlockFamily = (kind: BlockKind): kind is BlockFamily => !NEUTRAL.has(kind);

const classes = (family: BlockFamily, names: string): Array<[string, BlockFamily]> =>
  names.split(' ').map(name => [name, family]);
const dims = (stem: string) => [1, 2, 3].map(n => `${stem}${n}d`).join(' ');

/** torch.nn module classes, grouped as the torch.nn reference groups them, with the
 *  UI's adjustments: attention joins Transformer layers, shuffles join Shape. */
const MODULES = new Map<string, BlockFamily | 'container'>([
  ...classes('convolution', `${dims('Conv')} ${dims('ConvTranspose')} Unfold Fold`),
  ...classes('pooling', `${dims('MaxPool')} ${dims('MaxUnpool')} ${dims('AvgPool')} FractionalMaxPool2d FractionalMaxPool3d ${dims('LPPool')} ${dims('AdaptiveMaxPool')} ${dims('AdaptiveAvgPool')}`),
  ...classes('padding', `${dims('ReflectionPad')} ${dims('ReplicationPad')} ${dims('ZeroPad')} ${dims('ConstantPad')} ${dims('CircularPad')}`),
  ...classes('activation', 'ELU Hardshrink Hardsigmoid Hardtanh Hardswish LeakyReLU LogSigmoid PReLU ReLU ReLU6 RReLU SELU CELU GELU Sigmoid SiLU Mish Softplus Softshrink Softsign Tanh Tanhshrink Threshold GLU Softmin Softmax Softmax2d LogSoftmax AdaptiveLogSoftmaxWithLoss'),
  ...classes('normalization', `${dims('BatchNorm')} GroupNorm SyncBatchNorm ${dims('InstanceNorm')} LayerNorm LocalResponseNorm CrossMapLRN2d RMSNorm`),
  ...classes('recurrent', 'RNNBase RNN LSTM GRU RNNCellBase RNNCell LSTMCell GRUCell'),
  ...classes('attention', 'Transformer TransformerEncoder TransformerDecoder TransformerEncoderLayer TransformerDecoderLayer MultiheadAttention'),
  ...classes('linear', 'Linear Bilinear NonDynamicallyQuantizableLinear'),
  ...classes('dropout', 'Dropout Dropout1d Dropout2d Dropout3d AlphaDropout FeatureAlphaDropout'),
  ...classes('embedding', 'Embedding EmbeddingBag'),
  ...classes('distance', 'CosineSimilarity PairwiseDistance'),
  ...classes('resampling', 'Upsample UpsamplingNearest2d UpsamplingBilinear2d'),
  ...classes('shape', 'Flatten Unflatten PixelShuffle PixelUnshuffle ChannelShuffle Identity'),
  ...['Sequential', 'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict', 'Container'].map(name => [name, 'container'] as [string, 'container']),
]);
const MODULES_LOWER = new Map(Array.from(MODULES, ([name, family]) => [name.toLowerCase(), family]));
const PARALLEL = new Set(['DataParallel', 'DistributedDataParallel']);

/** Function/operator names shared by ATen targets, torch.* and torch.nn.functional. */
const OPERATIONS = new Map<string, BlockFamily>([
  ...classes('convolution', 'convolution conv1d conv2d conv3d conv_transpose1d conv_transpose2d conv_transpose3d conv_tbc convolution_overrideable mkldnn_convolution cudnn_convolution cudnn_convolution_transpose _conv_depthwise2d im2col col2im fold'),
  ...classes('linear', 'linear bilinear _trilinear addmm mm bmm matmul baddbmm addbmm addmv mv dot vdot outer ger einsum tensordot'),
  ...classes('normalization', 'batch_norm native_batch_norm cudnn_batch_norm miopen_batch_norm layer_norm native_layer_norm group_norm native_group_norm instance_norm rms_norm _fused_rms_norm local_response_norm normalize'),
  ...classes('activation', 'relu relu6 leaky_relu gelu silu elu selu celu prelu _prelu_kernel rrelu rrelu_with_noise hardtanh hardswish hardsigmoid log_sigmoid log_sigmoid_forward logsigmoid sigmoid tanh softplus softsign softshrink hardshrink tanhshrink threshold mish glu softmax log_softmax softmin'),
  ...classes('dropout', 'dropout native_dropout _fused_dropout feature_dropout alpha_dropout feature_alpha_dropout dropout1d dropout2d dropout3d'),
  ...classes('recurrent', 'rnn lstm gru rnn_tanh rnn_relu lstm_cell gru_cell rnn_tanh_cell rnn_relu_cell _thnn_fused_lstm_cell _thnn_fused_gru_cell _cudnn_rnn mkldnn_rnn_layer'),
  ...classes('attention', 'scaled_dot_product_attention _native_multi_head_attention multi_head_attention_forward'),
  ...classes('embedding', 'embedding embedding_bag embedding_renorm one_hot'),
  ...classes('padding', 'pad constant_pad_nd _pad_circular _pad_enum'),
  ...classes('resampling', 'interpolate upsample grid_sample grid_sampler grid_sampler_2d grid_sampler_3d affine_grid affine_grid_generator'),
  ...classes('shape', 'view _unsafe_view reshape view_as reshape_as permute transpose t flatten unflatten squeeze unsqueeze expand expand_as contiguous pixel_shuffle pixel_unshuffle channel_shuffle native_channel_shuffle split split_with_sizes chunk unbind slice select narrow as_strided unfold movedim moveaxis swapaxes swapdims repeat repeat_interleave tile flip roll alias index index_select gather scatter scatter_add scatter_reduce masked_fill masked_select masked_scatter index_put _index_put_impl where take'),
  ...classes('distance', 'cosine_similarity pairwise_distance cdist _cdist_forward _euclidean_dist pdist _pdist_forward dist'),
  ...classes('arithmetic', 'add sub rsub mul div true_divide floor_divide remainder fmod neg pow sqrt rsqrt exp log abs reciprocal square sign clamp clamp_min clamp_max clip maximum minimum addcmul addcdiv lerp cat concat concatenate stack hstack vstack sum mean prod amax amin max min argmax argmin var std var_mean std_mean cumsum cumprod logsumexp norm linalg_vector_norm round floor ceil sin cos tan asin acos atan sinh cosh eq ne lt le gt ge'),
]);

function operationFamily(name: string): BlockFamily | undefined {
  const base = name.endsWith('_') && !name.endsWith('__') ? name.slice(0, -1) : name;
  const known = OPERATIONS.get(base) ?? OPERATIONS.get(base.replace(/_backward(?:_data)?$/, ''));
  if (known) return known;
  const bare = base.replace(/^_+/, '').replace(/_backward(?:_data)?$/, '');
  if (OPERATIONS.has(bare)) return OPERATIONS.get(bare);
  if (/^(?:slow_conv|thnn_conv|_slow_conv2d|cudnn_convolution|_convolution)/.test(bare)) return 'convolution';
  if (/^(?:adaptive_)?(?:avg|max|lp)_pool[123]d(?:_with_indices)?$|^(?:max_unpool|fractional_max_pool)[123]d$/.test(bare)) return 'pooling';
  if (/^(?:native_)?batch_norm|^batch_norm_|^native_batch_norm_legit/.test(bare)) return 'normalization';
  if (/^scaled_dot_product_/.test(bare)) return 'attention';
  if (/^embedding_bag/.test(bare)) return 'embedding';
  if (/^(?:reflection|replication)_pad[123]d$/.test(bare)) return 'padding';
  if (/^upsample_(?:nearest(?:_exact)?|linear|bilinear|bicubic|trilinear)[123]d(?:_aa)?$/.test(bare)) return 'resampling';
  if (/(?:^|_)loss(?:_forward)?$|^binary_cross_entropy(?:_with_logits)?$|^cross_entropy$|^kl_div$|^nll_loss/.test(bare)) return 'loss';
  return undefined;
}

/** `torch.nn.modules.conv.Conv2d`, `torch.ao.nn.quantized.modules.linear.Linear`. */
function torchModule(identity: string): BlockClassification | undefined {
  const match = /^torch\.(?:ao\.)?nn\.((?:[a-z0-9_]+\.)*)([A-Z][A-Za-z0-9_]*)$/.exec(identity);
  if (!match) return undefined;
  const [, path, name] = match;
  if (PARALLEL.has(name)) return { kind: 'container', label: 'Parallel wrapper' };
  // Fused intrinsic modules (ConvReLU2d, ConvBnReLU2d) are several operations.
  if (/(?:^|\.)intrinsic\./.test(path)) return { kind: 'composite', label: BLOCK_LABELS.composite };
  const modifier = /(?:^|\.)qat\./.test(path) ? 'QAT' : /(?:^|\.)(?:quantized|quantizable)\./.test(path) ? 'Quantized' : undefined;
  return moduleClass(name, modifier) ?? (name.endsWith('Loss') ? withModifier('loss', modifier) : undefined);
}

function moduleClass(name: string, modifier?: string, lookup: Map<string, BlockFamily | 'container'> = MODULES): BlockClassification | undefined {
  const known = lookup.get(name);
  if (known) return withModifier(known, modifier);
  const lazy = /^lazy/i.test(name) && lookup.get(lookup === MODULES ? name.slice(4) : name.slice(4).toLowerCase());
  return lazy ? withModifier(lazy, modifier ? `Lazy · ${modifier}` : 'Lazy') : undefined;
}

const withModifier = (kind: BlockKind, modifier?: string): BlockClassification =>
  ({ kind, label: BLOCK_LABELS[kind], ...(modifier ? { modifier } : {}) });

type Classifiable = Pick<ModelNodeData, 'layerType' | 'captureRef' | 'captureLabel' | 'operationGroup'>;

/**
 * Resolve a node's block family from its canonical identity: the captured module
 * class, the captured ATen operator, or the source catalog/torch target. Display
 * names are never read, so `nn.Bilinear` (Linear) and a bilinear resize
 * (Resampling) stay apart, and a user subclass named `Linear` stays a composite.
 */
export function classifyBlock(data: Classifiable | undefined, nodeType?: string): BlockClassification {
  const layerType = data?.layerType;
  const id = String(layerType?.id ?? '');
  const name = String(layerType?.name ?? '');
  const captured = data?.captureRef?.kind;
  if (nodeType === 'inputNode' || id === 'input' || captured === 'input') return withModifier('input');
  if (nodeType === 'outputNode' || id === 'output' || captured === 'output') return withModifier('output');
  if (data?.operationGroup?.kind === 'slice') return withModifier('shape');
  if (captured === 'invocation' || (id === 'module' && /\./.test(name))) {
    return torchModule(name) ?? withModifier('composite');
  }
  const aten = [name, data?.captureLabel, id].map(target => /^(?:torch\.ops\.)?aten(?:\.|::)([a-zA-Z0-9_]+)/.exec(target ?? ''))
    .find(Boolean);
  if (aten) {
    const family = operationFamily(aten[1]);
    return family ? withModifier(family) : withModifier('unknown');
  }
  if (captured === 'operation' || captured === 'value') return withModifier('unknown');
  // A source call is not evidence of a module or of the operation's purpose.
  if (id === 'python') return { kind: 'unknown', label: 'Python call' };
  if (id === 'module' || id === 'custom_code') return withModifier('composite');
  const torchTarget = /^torch\.(?:nn\.functional\.|nn\.)?([A-Za-z0-9_]+)$/.exec(name);
  if (torchTarget) {
    const module = moduleClass(torchTarget[1]);
    if (module) return module;
    const family = operationFamily(torchTarget[1]);
    if (family) return withModifier(family);
  }
  // Source catalog ids: lower-cased torch.nn class names, then function names.
  const catalog = moduleClass(id, undefined, MODULES_LOWER) ?? (id.endsWith('loss') ? withModifier('loss') : undefined);
  if (catalog) return catalog;
  const family = operationFamily(id);
  return family ? withModifier(family) : withModifier('unknown');
}

export const classifyNode = (node: Pick<ModelNode, 'data' | 'type'>): BlockClassification => classifyBlock(node.data, node.type);

/**
 * The families a composite module contains, in data-flow order, from the child
 * nodes this graph actually holds. Consecutive repeats collapse; neutral children
 * are omitted. Empty when no child is classified, so nothing is inferred.
 */
export function moduleComposition(moduleId: string, nodes: ModelNode[], edges: ModelEdge[], limit = 5): BlockFamily[] {
  const children = nodes.filter(node => node.data.parentId === moduleId || node.parentNode === moduleId);
  const ids = new Set(children.map(node => node.id));
  const incoming = new Map(children.map(node => [node.id, 0]));
  const next = new Map<string, string[]>();
  for (const edge of edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target) || edge.source === edge.target) continue;
    incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1);
    next.set(edge.source, [...(next.get(edge.source) ?? []), edge.target]);
  }
  const order: ModelNode[] = [];
  const ready = children.filter(node => !incoming.get(node.id));
  const byId = new Map(children.map(node => [node.id, node]));
  while (ready.length) {
    const node = ready.shift()!;
    order.push(node);
    for (const target of next.get(node.id) ?? []) {
      incoming.set(target, incoming.get(target)! - 1);
      if (!incoming.get(target)) ready.push(byId.get(target)!);
    }
  }
  // A cycle cannot be ordered; keep document order for whatever remains.
  children.forEach(node => { if (!order.includes(node)) order.push(node); });
  const families: BlockFamily[] = [];
  for (const node of order) {
    const { kind } = classifyNode(node);
    if (isBlockFamily(kind) && families[families.length - 1] !== kind) families.push(kind);
  }
  return families.slice(0, limit);
}

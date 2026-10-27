import { classifyBlock, classifyNode, moduleComposition } from '../../canvas/blockFamily';
import { getLayerByType } from '../../data/layerTypes';
import type { ModelEdge, ModelNode, ModelNodeData } from '../../types';

const captured = (kind: string, name: string, captureLabel?: string) => ({
  captureRef: { kind }, captureLabel, layerType: { id: kind === 'invocation' ? 'module' : 'captured_operation', name },
} as unknown as ModelNodeData);
const source = (id: string, name = id) => ({ layerType: { id, name } } as ModelNodeData);
const family = (data: ModelNodeData, type?: string) => classifyBlock(data, type).kind;

describe('block family classification', () => {
  it.each([
    ['torch.nn.modules.conv.Conv2d', 'convolution'], ['torch.nn.modules.conv.ConvTranspose2d', 'convolution'],
    ['torch.nn.modules.linear.Linear', 'linear'], ['torch.nn.modules.linear.Bilinear', 'linear'],
    ['torch.nn.modules.batchnorm.BatchNorm2d', 'normalization'], ['torch.nn.modules.normalization.LayerNorm', 'normalization'],
    ['torch.nn.modules.normalization.RMSNorm', 'normalization'], ['torch.nn.modules.activation.SiLU', 'activation'],
    ['torch.nn.modules.activation.Softmax', 'activation'], ['torch.nn.modules.pooling.AdaptiveAvgPool2d', 'pooling'],
    ['torch.nn.modules.dropout.Dropout2d', 'dropout'], ['torch.nn.modules.rnn.LSTM', 'recurrent'],
    ['torch.nn.modules.activation.MultiheadAttention', 'attention'], ['torch.nn.modules.transformer.TransformerEncoderLayer', 'attention'],
    ['torch.nn.modules.sparse.EmbeddingBag', 'embedding'], ['torch.nn.modules.padding.ReflectionPad2d', 'padding'],
    ['torch.nn.modules.upsampling.Upsample', 'resampling'], ['torch.nn.modules.flatten.Flatten', 'shape'],
    ['torch.nn.modules.pixelshuffle.PixelShuffle', 'shape'], ['torch.nn.modules.distance.CosineSimilarity', 'distance'],
    ['torch.nn.modules.loss.CrossEntropyLoss', 'loss'], ['torch.nn.modules.loss.TripletMarginWithDistanceLoss', 'loss'],
    ['torch.nn.modules.container.Sequential', 'container'], ['torch.nn.modules.container.ModuleList', 'container'],
  ])('captured module %s is %s', (cls, expected) => {
    expect(family(captured('invocation', cls))).toBe(expected);
  });

  it.each([
    ['aten.convolution.default', 'convolution'], ['aten._convolution.default', 'convolution'],
    ['aten.addmm.default', 'linear'], ['aten.bmm.default', 'linear'],
    ['aten.native_layer_norm.default', 'normalization'], ['aten._native_batch_norm_legit_no_training.default', 'normalization'],
    ['aten.relu_.default', 'activation'], ['aten._softmax.default', 'activation'],
    ['aten.max_pool2d_with_indices.default', 'pooling'], ['aten._adaptive_avg_pool2d.default', 'pooling'],
    ['aten.native_dropout.default', 'dropout'], ['aten._scaled_dot_product_flash_attention_for_cpu.default', 'attention'],
    ['aten.embedding.default', 'embedding'], ['aten.reflection_pad2d.default', 'padding'],
    ['aten.upsample_bilinear2d.vec', 'resampling'], ['aten.view.default', 'shape'], ['aten.t.default', 'shape'],
    ['aten.cosine_similarity.default', 'distance'], ['aten.nll_loss_forward.default', 'loss'],
    ['aten.add.Tensor', 'arithmetic'], ['aten.cat.default', 'arithmetic'], ['aten.mul.Tensor', 'arithmetic'],
  ])('captured operation %s is %s', (target, expected) => {
    expect(family(captured('operation', target, target))).toBe(expected);
  });

  it('keeps nn.Bilinear and a bilinear resize in different families', () => {
    expect(family(captured('invocation', 'torch.nn.modules.linear.Bilinear'))).toBe('linear');
    expect(family(captured('operation', 'aten.upsample_bilinear2d.vec'))).toBe('resampling');
    expect(family(source('interpolate', 'torch.nn.functional.interpolate'))).toBe('resampling');
    expect(family(source('bilinear'))).toBe('linear');
  });

  it('gives equivalent module, functional, source and ATen forms one family', () => {
    const forms = [captured('invocation', 'torch.nn.modules.activation.GELU'), source('gelu', 'GELU'),
      source('gelu', 'torch.nn.functional.gelu'), captured('operation', 'aten.gelu.default')];
    expect(forms.map(form => family(form))).toEqual(['activation', 'activation', 'activation', 'activation']);
    expect(['conv2d', 'batchnorm2d', 'adaptiveavgpool2d', 'flatten', 'add', 'cat', 'mseloss']
      .map(id => family(source(id)))).toEqual(['convolution', 'normalization', 'pooling', 'shape', 'arithmetic', 'arithmetic', 'loss']);
  });

  it('classifies every source catalog entry from its id, not its display name', () => {
    const unknown = ['conv2d', 'relu', 'maxpool2d', 'layernorm', 'transformerencoderlayer', 'lstm', 'embedding', 'zeropad2d',
      'upsample', 'pixelshuffle', 'crossentropyloss', 'matmul', 'identity']
      .filter(id => { const layer = getLayerByType(id); return !layer || family({ layerType: layer } as ModelNodeData) === 'unknown'; });
    expect(unknown).toEqual([]);
  });

  it('keeps user classes, custom code and fused modules neutral even when their names look standard', () => {
    expect(family(captured('invocation', 'whisper.model.Linear'))).toBe('composite');
    expect(family(captured('invocation', 'models.blocks.ConvNormAct'))).toBe('composite');
    expect(family(captured('invocation', 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU2d'))).toBe('composite');
    expect(family(source('custom_code', 'Custom Code'))).toBe('composite');
    expect(family(source('module', 'Module'))).toBe('composite');
    expect(family(source('mystery_op', 'MyConv2d'))).toBe('unknown');
    expect(family(captured('value', 'State'))).toBe('unknown');
  });

  it('labels lazy, quantized and parallel variants without losing their family', () => {
    expect(classifyBlock(captured('invocation', 'torch.nn.modules.conv.LazyConv2d'))).toEqual({ kind: 'convolution', label: 'Convolution', modifier: 'Lazy' });
    expect(classifyBlock(source('lazylinear', 'LazyLinear'))).toMatchObject({ kind: 'linear', modifier: 'Lazy' });
    expect(classifyBlock(captured('invocation', 'torch.ao.nn.quantized.modules.linear.Linear'))).toMatchObject({ kind: 'linear', modifier: 'Quantized' });
    expect(classifyBlock(captured('invocation', 'torch.nn.parallel.data_parallel.DataParallel'))).toEqual({ kind: 'container', label: 'Parallel wrapper' });
  });

  it('marks inputs and outputs by kind', () => {
    expect(family(source('input', 'Input'), 'inputNode')).toBe('input');
    expect(family({ layerType: { id: 'output' } } as ModelNodeData)).toBe('output');
    expect(classifyNode({ type: 'inputNode', data: captured('input', 'Input') } as ModelNode).label).toBe('Input');
  });
});

describe('module composition', () => {
  const node = (id: string, parentId: string | undefined, cls: string) =>
    ({ id, position: { x: 0, y: 0 }, data: { ...captured('invocation', cls), parentId } } as unknown as ModelNode);
  const edge = (source: string, target: string) => ({ id: `${source}-${target}`, source, target } as ModelEdge);

  it('reads verified children in data-flow order and collapses repeats', () => {
    const nodes = [node('block', undefined, 'models.blocks.ConvNormAct'), node('block.act', 'block', 'torch.nn.modules.activation.SiLU'),
      node('block.bn', 'block', 'torch.nn.modules.batchnorm.BatchNorm2d'), node('block.conv', 'block', 'torch.nn.modules.conv.Conv2d'),
      node('block.conv2', 'block', 'torch.nn.modules.conv.Conv2d'), node('other', undefined, 'torch.nn.modules.linear.Linear')];
    const edges = [edge('block.conv', 'block.conv2'), edge('block.conv2', 'block.bn'), edge('block.bn', 'block.act'), edge('block.act', 'other')];
    expect(moduleComposition('block', nodes, edges)).toEqual(['convolution', 'normalization', 'activation']);
  });

  it('infers nothing for a module without classified children', () => {
    expect(moduleComposition('block', [node('block', undefined, 'models.Block'), node('block.inner', 'block', 'models.Inner')], [])).toEqual([]);
  });
});

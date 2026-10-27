import { operationDisplayName } from '../../canvas/operationDisplayName';
import type { ModelNodeData } from '../../types';

const operation = (target: string): ModelNodeData => ({
  captureRef: { kind: 'operation' }, captureLabel: target,
  layerType: { name: target }, parameters: {},
} as ModelNodeData);

it.each([
  ['aten.mean.dim', 'Mean'], ['aten.view.default', 'Reshape'],
  ['aten._unsafe_view.default', 'Reshape'], ['aten.bmm.default', 'Batch matrix multiply'],
  ['aten.addmm.default', 'Matrix multiply + add'], ['aten.transpose.int', 'Transpose'],
  ['aten._softmax.default', 'Softmax'], ['aten.native_layer_norm.default', 'Layer norm'],
  ['aten._scaled_dot_product_flash_attention_for_cpu.default', 'Scaled attention'],
  ['aten._native_multi_head_attention.default', 'Multi-head attention'],
  ['aten._scaled_dot_product_attention_math.default', 'Scaled attention'],
  ['aten._scaled_dot_product_cudnn_attention.default', 'Scaled attention'],
  ['aten._scaled_dot_product_flash_attention_backward.default', 'Scaled attention backward'],
  ['aten._native_batch_norm_legit_no_training.default', 'Batch norm'],
  ['aten._native_batch_norm_legit_functional.default', 'Batch norm'],
  ['aten.native_group_norm.default', 'Group norm'], ['aten.rms_norm.default', 'RMS norm'],
  ['aten.max_pool2d_with_indices.default', 'Max pool 2D'],
  ['aten.max_pool3d_with_indices_backward.default', 'Max pool 3D backward'],
  ['aten._adaptive_avg_pool2d.default', 'Adaptive average pool 2D'],
  ['aten.adaptive_max_pool1d.default', 'Adaptive max pool 1D'],
  ['aten.avg_pool3d.default', 'Average pool 3D'],
  ['aten.conv1d.default', 'Convolution 1D'], ['aten._convolution.default', 'Convolution'],
  ['aten.conv_transpose2d.input', 'Transposed convolution 2D'],
  ['aten.upsample_bilinear2d.vec', 'Bilinear resize 2D'],
  ['aten._upsample_nearest_exact2d.default', 'Exact nearest resize 2D'],
  ['aten.upsample_nearest2d_backward.default', 'Nearest resize 2D backward'],
  ['aten.reflection_pad2d.default', 'Reflection padding 2D'],
  ['aten.replication_pad1d.default', 'Replication padding 1D'],
  ['aten.hardtanh.default', 'Hard tanh'], ['aten.silu.default', 'SiLU'],
  ['aten._log_softmax.default', 'Log softmax'], ['aten.native_dropout.default', 'Dropout'],
  ['aten.index.Tensor', 'Index tensor'], ['aten.index_put_.default', 'Write at indices (in place)'],
  ['aten.masked_fill.Scalar', 'Fill masked values'], ['aten.gather.default', 'Gather'],
  ['aten.embedding.default', 'Embedding lookup'], ['aten.where.self', 'Select by condition'],
  ['aten.var_mean.correction', 'Variance + mean'], ['aten.rsqrt.default', 'Reciprocal sqrt'],
  ['aten.empty.memory_format', 'Allocate tensor'], ['aten.new_zeros.default', 'Zeros'],
  ['aten.nll_loss_forward.default', 'Negative log-likelihood loss'],
  ['aten.mse_loss.default', 'Mean squared error loss'],
  ['aten.native_layer_norm_backward.default', 'Layer norm backward'],
  ['aten.nll_loss_backward.default', 'Negative log-likelihood loss backward'],
  ['aten::flatten.using_ints', 'Flatten'], ['aten::conv3d', 'Convolution 3D'],
  ['torch.ops.aten.permute.default', 'Permute axes'], ['aten.add_.Tensor', 'Add (in place)'],
  ['aten.__and__.Tensor', 'And'],
  ['aten.some_future_kernel.default', 'Some future kernel'],
  ['aten.constructor.default', 'Constructor'],
])('renames %s without changing captured data', (target, label) => {
  const data = operation(target);
  const before = JSON.stringify(data);
  expect(operationDisplayName(data)).toBe(label);
  expect(JSON.stringify(data)).toBe(before);
});

it('preserves module, custom operation and grouped layout names', () => {
  expect(operationDisplayName({ ...operation('aten.mean.dim'), captureRef: { kind: 'invocation' } as ModelNodeData['captureRef'] })).toBeNull();
  expect(operationDisplayName(operation('custom.kernel'))).toBeNull();
  expect(operationDisplayName({ ...operation('aten.view.default'), operationGroup: { members: ['a', 'b'], labels: ['permute', 'view'] } })).toBeNull();
});

it('uses the exact layer target when an older capture has no label', () => {
  const data = operation('aten.mean.dim');
  delete data.captureLabel;
  expect(operationDisplayName(data)).toBe('Mean');
});

it.each(['name', 'id', 'displayName'] as const)('renames imported operations whose target is stored as %s, without requiring capture metadata', field => {
  const data = { layerType: { [field]: 'aten.max_pool2d_with_indices.default' }, parameters: {} } as ModelNodeData;
  expect(operationDisplayName(data)).toBe('Max pool 2D');
});

it('prefers an exact operation target over a custom capture label', () => {
  expect(operationDisplayName({ ...operation('aten.gelu.default'), captureLabel: 'activation_1' })).toBe('GELU');
});

it.each(['Linear', 'view', 'custom::view', 'my_aten.view.default', 'aten.view.default.extra'])('does not reinterpret the non-ATen name %s', target => {
  expect(operationDisplayName({ ...operation(target), captureRef: undefined })).toBeNull();
});

it('does not rename a module using an ATen-like class name', () => {
  const data = operation('aten.mean.dim');
  data.captureRef = undefined;
  data.layerType.category = 'module';
  expect(operationDisplayName(data)).toBeNull();
});

it.each([
  ['torch.reshape', 'Reshape'], ['torch.permute', 'Permute axes'],
  ['torch.cat', 'Concatenate'], ['torch.nn.functional.interpolate', 'Interpolate'],
])('uses the shared vocabulary for source operation %s', (target, label) => {
  expect(operationDisplayName({ ...operation(target), captureRef: undefined, captureLabel: 'permute@Net#0' })).toBe(label);
});

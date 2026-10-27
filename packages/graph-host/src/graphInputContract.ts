/** The graph-setup input grammar, published to MCP clients.
 *
 *  These values mirror the backend's Pydantic contracts
 *  (`services/validation/contracts.py`). The compiler keeps them honest:
 *  `TENSOR_GENERATOR_FIELDS` and `TENSOR_DTYPE_ORDER` are keyed by the exact
 *  union types in the generated contracts, so adding a generator kind or dtype
 *  in the backend fails the build until it is described here. No client should
 *  have to read tensorViz's source to fill a setup draft.
 */

import type { components } from '@tensorviz/contracts/validationContracts.generated';

export type TensorValue = components['schemas']['TensorValue'];
export type TensorGenerator = TensorValue['generator'];
export type TensorGeneratorKind = TensorGenerator['kind'];
export type TensorDtype = TensorValue['dtype'];

/** Every generator kind the backend accepts, with the fields it requires. */
export const TENSOR_GENERATOR_FIELDS: Record<TensorGeneratorKind, { fields: string[]; rules: string }> = {
  normal: { fields: ['mean', 'std', 'seed'], rules: 'Floating-point dtype only; std > 0.' },
  uniform: { fields: ['low', 'high', 'seed'], rules: 'Floating-point dtype only; low < high.' },
  randint: { fields: ['low', 'high', 'seed'], rules: 'Integer dtype only; low < high and both fall within the dtype range.' },
  full: { fields: ['value'], rules: 'A boolean for a bool tensor, otherwise a number matching the dtype.' },
  values: { fields: ['values'], rules: 'Exactly product(shape) literals; a bool tensor accepts only booleans.' },
};

/** Every dtype the backend accepts, in a stable order. */
export const TENSOR_DTYPE_ORDER: Record<TensorDtype, number> = {
  bool: 0, uint8: 1, int8: 2, int16: 3, int32: 4, int64: 5,
  float16: 6, bfloat16: 7, float32: 8, float64: 9,
};

export const TENSOR_GENERATOR_KINDS = Object.keys(TENSOR_GENERATOR_FIELDS) as TensorGeneratorKind[];
export const TENSOR_DTYPES = Object.keys(TENSOR_DTYPE_ORDER) as TensorDtype[];

/** The published contract a client reads before filling a draft. */
export const INPUT_CONTRACT = {
  argumentKinds: ['float', 'none', 'list', 'tuple', 'dict', 'tensor'],
  tensor: {
    kind: 'tensor',
    fields: ['kind', 'shape', 'dtype', 'requires_grad', 'generator'],
    shape: 'Up to 32 non-negative dimensions; at most 4,194,304 elements in total.',
    requiresGrad: 'Floating-point dtype only.',
    generator: Object.fromEntries(TENSOR_GENERATOR_KINDS.map(kind => [kind, TENSOR_GENERATOR_FIELDS[kind]])),
  },
  dtypes: TENSOR_DTYPES,
  binding: { fields: ['args', 'kwargs'], order: 'Positional args first, then keyword kwargs.' },
  inputFactory: {
    shape: 'inputs: { factory: { file: "model.py", callable: "graph_inputs", kwargs: {} } }',
    returns: 'A dictionary of named forward inputs, or { args: [...], kwargs: {...} }.',
    note: 'Use an explicit workspace function when example inputs already exist in Python. Runs once in the capture worker with source freshness checks. Mutually exclusive with args/kwargs bindings.',
  },
  draft: {
    shape: '{ profile: { kind: "model", model: "<module>:<Class>", mode, device, seed, factory?, init?, inputs: { args, kwargs } | { factory: { file, callable, kwargs } } } }',
    note: 'Leave a field out only when source already resolved it. Read the target model\'s forward to choose each tensor shape, dtype and name; the grammar above is all the tool needs.',
  },
  limits: { elementsPerTensor: 4_194_304, tensorsPerBinding: 8, dimensions: 32 },
} as const;

import * as assert from 'node:assert/strict';
import { shapeHintsByFile } from '../../projection';

const REL = 'models/net.py';
const BLOCKS_REL = 'models/blocks.py';

suite('shapeHintsByFile captured tensor shapes', () => {
  const declaration = { file: REL, start: [7, 8], end: [7, 40], kind: 'decl' };
  const call = { file: REL, start: [11, 20], end: [11, 30], kind: 'call' };
  const node = (nodeId: string, shape: unknown, metadata = {}) => ({
    node_id: nodeId, params: { _out_shape: shape, _shape_provenance: 'observed' },
    metadata: { captureRef: { kind: 'invocation' }, declarationSpan: declaration, ...metadata },
  });

  test('anchors measured shapes to the executed call instead of the declaration', () => {
    for (const [shape, label] of [[[2, 16], '2x16'], [[0, 8], '0x8'], [[], 'scalar'], ['2x16', '2x16']] as const) {
      const hints = shapeHintsByFile({ traced: true,
        document: { semantic: { nodes: [node('call:up', shape)] } },
        spanmap: { 'call:up': call },
      });
      assert.deepEqual(hints.get(REL), [{ nodeId: 'call:up', line: 11, column: 30, shape: label, provenance: 'observed' }]);
    }
  });

  test('keeps nested, repeated and cross-file calls at their own expression ends', () => {
    const hints = shapeHintsByFile({ traced: true,
      document: { semantic: { nodes: [node('norm', [2, 16]), node('head', [2, 8]), node('head:2', [1, 8])] } },
      spanmap: {
        norm: call,
        head: { ...call, start: [11, 8], end: [11, 31] },
        'head:2': { ...call, file: BLOCKS_REL, start: [15, 12], end: [17, 9] },
      },
    });
    assert.deepEqual(hints.get(REL), [
      { nodeId: 'norm', line: 11, column: 30, shape: '2x16', provenance: 'observed' },
      { nodeId: 'head', line: 11, column: 31, shape: '2x8', provenance: 'observed' },
    ]);
    assert.deepEqual(hints.get(BLOCKS_REL), [
      { nodeId: 'head:2', line: 17, column: 9, shape: '1x8', provenance: 'observed' },
    ]);
  });

  test('older timing evidence adds nothing to source hints', () => {
    const timing = { cpuTotalUs: [10, 20, 30], cpuSelfUs: [4, 5, 6], deviceTotalUs: null, deviceMethod: null };
    const hints = shapeHintsByFile({ traced: true,
      document: { semantic: { nodes: [node('call:up', [2, 16], { moduleLatency: timing }), node('call:pair', undefined, { moduleLatency: timing })],
        metadata: { latencyProfile: { status: 'measured', device: 'cpu' } } } },
      spanmap: { 'call:up': call, 'call:pair': { ...call, start: [12, 8], end: [12, 20] } },
    });
    assert.deepEqual(hints.get(REL), [{ nodeId: 'call:up', line: 11, column: 30, shape: '2x16', provenance: 'observed' }],
      'a timed call shows its shape only, and a call without a tensor shape shows nothing');
  });

  test('shows the outer module output, not internal operations mapped to the same call', () => {
    const hints = shapeHintsByFile({ traced: true,
      document: { semantic: { nodes: [
        node('encoder', [1, 10, 32]),
        node('encoder.linear', [1, 10, 128], { parentId: 'encoder' }),
        operation('encoder.op', [32, 128], ['input'], ['internal'], 'encoder.linear'),
      ] } },
      spanmap: { encoder: call, 'encoder.linear': call, 'encoder.op': call },
    });
    assert.deepEqual(hints.get(REL), [
      { nodeId: 'encoder', line: 11, column: 30, shape: '1x10x32', provenance: 'observed' },
    ]);
  });

  test('never substitutes an internal shape for an unavailable module output', () => {
    const hints = shapeHintsByFile({ traced: true,
      document: { semantic: { nodes: [node('encoder', undefined), node('encoder.linear', [1, 10, 128], { parentId: 'encoder' }),
        operation('encoder.op', [1, 10, 32], ['input'], ['internal'], 'encoder')] } },
      spanmap: { encoder: call, 'encoder.linear': call, 'encoder.op': call },
    });
    assert.equal(hints.size, 0);
  });

  const operation = (id: string, shape: unknown, inputs: string[], outputs: Array<string | null>, parentId = 'root') =>
    node(id, shape, { captureRef: { kind: 'operation' }, parentId,
      capturePorts: [
        ...inputs.map(valueId => ({ direction: 'in', valueId })),
        ...outputs.map(valueId => ({ direction: 'out', valueId })),
      ],
    });

  test('shows the final sliced tensor, not the intermediate dimension, independently of record order', () => {
    const slice = { ...call, start: [11, 35], end: [11, 66] };
    const nodes = [
      node('embed', [1, 8, 32]),
      operation('slice:0', [1, 12, 32], ['pos'], ['partial']),
      operation('slice:1', [1, 8, 32], ['partial'], ['sliced']),
      operation('add', [1, 8, 32], ['embedded', 'sliced'], ['sum']),
    ];
    for (const ordered of [nodes, [...nodes].reverse()]) {
      const hints = shapeHintsByFile({ traced: true,
        document: { semantic: { nodes: ordered } },
        spanmap: { embed: call, 'slice:0': slice, 'slice:1': slice,
          add: { ...slice, start: [11, 12] } },
      });
      assert.deepEqual(hints.get(REL), [
        { nodeId: 'embed', line: 11, column: 30, shape: '1x8x32', provenance: 'observed' },
        { nodeId: 'slice:1', line: 11, column: 66, shape: '1x8x32', provenance: 'observed' },
      ], 'the slice and addition share one matching hint at their common expression end');
    }
  });

  test('retains different nested outputs, tensor methods and helper-file expressions', () => {
    const hints = shapeHintsByFile({ traced: true,
      document: { semantic: { nodes: [
        operation('sum', [2, 8], ['sliced', 'broadcast'], ['sum']),
        operation('slice', [1, 8], ['x'], ['sliced']),
        operation('transpose', [8, 2], ['sum'], ['transposed']),
      ] } },
      spanmap: { sum: call, slice: { ...call, start: [11, 25] },
        transpose: { ...call, file: BLOCKS_REL, start: [15, 12], end: [17, 9] } },
    });
    assert.deepEqual(hints.get(REL)?.map(hint => [hint.nodeId, hint.shape]), [['slice', '1x8'], ['sum', '2x8']]);
    assert.deepEqual(hints.get(BLOCKS_REL), [
      { nodeId: 'transpose', line: 17, column: 9, shape: '8x2', provenance: 'observed' },
    ]);
  });

  test('never guesses an expression result from ambiguous, failed or unavailable terminal operations', () => {
    for (const last of [
      operation('last', [2, 8], ['independent'], ['other']),
      operation('last', undefined, ['partial'], ['a', 'b']),
      operation('last', undefined, ['partial'], [null]),
      { ...operation('last', [2, 8], ['partial'], ['result']), params: { _out_shape: [2, 8], _shape_provenance: 'failed' } },
    ]) {
      assert.equal(shapeHintsByFile({ traced: true,
        document: { semantic: { nodes: [operation('first', [2, 12], ['x'], ['partial']), last] } },
        spanmap: { first: call, last: call },
      }).size, 0);
    }
  });

  test('omits static, invalid, missing-call and declaration-only evidence', () => {
    for (const [shape, provenance, span, traced] of [
      [[2, 16], 'static', call, true],
      [[2, null], 'observed', call, true],
      [[2, 16], 'observed', undefined, true],
      [[2, 16], 'observed', declaration, true],
      [[2, 16], 'observed', call, false],
    ] as const) {
      assert.equal(shapeHintsByFile({ traced,
        document: { semantic: { nodes: [{ ...node('call:up', shape),
          params: { _out_shape: shape, _shape_provenance: provenance } }] } },
        spanmap: { 'call:up': span },
      }).size, 0);
    }
  });
});

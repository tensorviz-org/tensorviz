import { estimateNodeParams, graphMetrics, projectMetrics, formatParams, formatSize } from '../../services/graphMetrics';
import { ModelNode, SavedGraph } from '../../types';
import { GraphDocumentV2 } from '../../types/graphDocument';

/**
 * The projects table and the home cards show a parameter count and a model size.
 * Neither is persisted and no endpoint serves them for a LIST of projects, so
 * they are derived here from the layer configs. Two things must hold, or the
 * numbers are worse than absent:
 *
 *  1. The arithmetic matches torch's own parameter shapes. A Conv2d is
 *     (in/groups)*out*kh*kw + out, not a guess.
 *  2. An estimate is never reported as a measurement. `exact` is the flag the UI
 *     uses to append "est.", and a recorded verify must win over the estimate.
 */

function node(name: string, parameters: Record<string, unknown>, extra: Record<string, unknown> = {}): ModelNode {
  return {
    id: `${name}-${Math.random()}`,
    position: { x: 0, y: 0 },
    data: { layerType: { name } as any, parameters, ...extra },
  } as unknown as ModelNode;
}

describe('estimateNodeParams', () => {
  it('counts a Linear layer as in*out + bias', () => {
    // 784*128 = 100352 weights, +128 bias.
    expect(estimateNodeParams(node('Linear', { in_features: 784, out_features: 128 }))).toBe(100480);
  });

  it('drops the bias term when bias is disabled', () => {
    expect(estimateNodeParams(node('Linear', { in_features: 784, out_features: 128, bias: false }))).toBe(100352);
  });

  it('counts a Conv2d over its kernel area, not per pixel', () => {
    // 3*64*7*7 = 9408, +64 bias. This is resnet18's stem, so the number is checkable.
    expect(estimateNodeParams(node('Conv2d', { in_channels: 3, out_channels: 64, kernel_size: 7 }))).toBe(9472);
  });

  it('accepts a kernel given as a pair', () => {
    expect(estimateNodeParams(node('Conv2d', { in_channels: 3, out_channels: 64, kernel_size: [7, 7] }))).toBe(9472);
  });

  it('divides a grouped convolution by its groups', () => {
    // Depthwise 7x7 over 96 channels: (96/96)*96*49 + 96.
    expect(estimateNodeParams(node('Conv2d', { in_channels: 96, out_channels: 96, kernel_size: 7, groups: 96 }))).toBe(4800);
  });

  it('counts BatchNorm as one weight and one bias per channel', () => {
    expect(estimateNodeParams(node('BatchNorm2d', { num_features: 64 }))).toBe(128);
  });

  it('counts an unaffine norm as weightless', () => {
    expect(estimateNodeParams(node('BatchNorm2d', { num_features: 64, affine: false }))).toBe(0);
  });

  it('counts MultiheadAttention as its four projections', () => {
    // 4*d^2 + 4*d for d=192.
    expect(estimateNodeParams(node('MultiheadAttention', { embed_dim: 192 }))).toBe(148224);
  });

  it('returns zero for a layer that genuinely holds no weights', () => {
    expect(estimateNodeParams(node('ReLU', {}))).toBe(0);
    expect(estimateNodeParams(node('MaxPool2d', { kernel_size: 2 }))).toBe(0);
  });

  it('returns zero rather than a guess for a layer it does not know', () => {
    // Inventing a number for an unrecognised layer would make the total silently
    // wrong, which is worse than an under-count the UI marks as an estimate.
    expect(estimateNodeParams(node('SomeFutureLayer', { in_features: 512, out_features: 512 }))).toBe(0);
  });
});

describe('graphMetrics', () => {
  it('excludes Input and Output markers from the layer count', () => {
    const graph = {
      nodes: [
        node('Input', { shape: [1, 3, 224, 224] }),
        node('Conv2d', { in_channels: 3, out_channels: 8, kernel_size: 3 }),
        node('Output', {}),
      ],
    };
    expect(graphMetrics(graph).layers).toBe(1);
  });

  it('excludes a module grouping frame from the layer count', () => {
    // An edge-free kind="module" node is canvas grouping, not an executable
    // layer. Counting frames as layers is the overcount this replaces.
    const graph = {
      nodes: [
        node('Conv2d', { in_channels: 3, out_channels: 8, kernel_size: 3 }),
        node('ResidualBlock', {}, { isGroup: true }),
      ],
    };
    expect(graphMetrics(graph).layers).toBe(1);
  });

  it('prefers the document node kinds over the derived projection', () => {
    // The derived ModelGraph adds display frames, so the document is authority.
    const document = {
      schemaVersion: 2,
      semantic: {
        nodes: [
          { node_id: 'a', kind: 'input' },
          { node_id: 'b', kind: 'layer' },
          { node_id: 'c', kind: 'layer' },
          { node_id: 'd', kind: 'module' },
          { node_id: 'e', kind: 'output' },
        ],
      },
    } as unknown as GraphDocumentV2;
    const graph = { nodes: [node('Conv2d', { in_channels: 3, out_channels: 8, kernel_size: 3 })] };
    expect(graphMetrics(graph, document).layers).toBe(2);
  });

  it('multiplies a repeated block by its repeat count and ignores hidden copies', () => {
    // A x3 stack renders one representative child; the hidden siblings exist for
    // codegen. Counting both the multiplier and the copies would triple-count.
    const graph = {
      nodes: [
        node('Linear', { in_features: 10, out_features: 10 }, { repeatCount: 3 }),
        node('Linear', { in_features: 10, out_features: 10 }, { repeatHidden: true }),
      ],
    };
    // (10*10 + 10) * 3 = 330, and the hidden node contributes nothing.
    expect(graphMetrics(graph).params).toBe(330);
    expect(graphMetrics(graph).layers).toBe(1);
  });

  it('sizes the model as four bytes per parameter and flags itself as an estimate', () => {
    const graph = { nodes: [node('Linear', { in_features: 100, out_features: 10 })] };
    const m = graphMetrics(graph);
    expect(m.params).toBe(1010);
    expect(m.sizeBytes).toBe(4040);
    expect(m.exact).toBe(false);
  });
});

describe('projectMetrics', () => {
  const graph = { nodes: [node('Linear', { in_features: 100, out_features: 10 })], edges: [] };

  it('falls back to the estimate when nothing has verified the project', () => {
    const project = { graph, document: undefined } as unknown as SavedGraph;
    const m = projectMetrics(project);
    expect(m.params).toBe(1010);
    expect(m.exact).toBe(false);
  });

  it("prefers torch's recorded numbers over its own estimate", () => {
    // The estimate cannot see anything the graph does not declare, so a recorded
    // verify must win — and must be reported as exact so the UI drops "est.".
    const document = {
      schemaVersion: 2,
      semantic: {
        nodes: [],
        metadata: {
          verification: {
            ok: true,
            revision: 4,
            errorCount: 0,
            summary: { layerCount: 68, params: 11_689_512, sizeMb: 44.6 },
          },
        },
      },
    } as unknown as GraphDocumentV2;
    const m = projectMetrics({ graph, document } as unknown as SavedGraph);
    expect(m.layers).toBe(68);
    expect(m.params).toBe(11_689_512);
    expect(m.sizeBytes).toBe(Math.round(44.6 * 1024 * 1024));
    expect(m.exact).toBe(true);
  });

  it('does not throw when the project has no document at all', () => {
    // Existing fixtures across the suite build SavedGraph-shaped objects with no
    // document field, and a list render must survive them.
    expect(() => projectMetrics({ graph } as unknown as SavedGraph)).not.toThrow();
  });
});

describe('formatting', () => {
  it('renders parameter counts at the design’s magnitudes', () => {
    expect(formatParams(11_689_512)).toBe('11.7M');
    expect(formatParams(52_600)).toBe('52.6K');
    expect(formatParams(124_000_000)).toBe('124M');
    expect(formatParams(512)).toBe('512');
  });

  it('renders an em dash rather than "0" when there is nothing to report', () => {
    // "0M" would assert the model has no parameters; the dash says unknown.
    expect(formatParams(0)).toBe('—');
    expect(formatSize(0)).toBe('—');
  });

  it('renders sizes at the design’s magnitudes', () => {
    expect(formatSize(44.6 * 1024 * 1024)).toBe('44.6 MB');
    expect(formatSize(507 * 1024 * 1024)).toBe('507 MB');
    expect(formatSize(2 * 1024 * 1024 * 1024)).toBe('2.0 GB');
  });

  it('drops to KB rather than showing a small model as "0.0 MB"', () => {
    // A 676-parameter template is 2.6 KB. "0.0 MB" is the one thing it is not:
    // sizeless. This is the real mlp_tiny case.
    expect(formatSize(676 * 4)).toBe('3 KB');
    expect(formatSize(52_600 * 4)).toBe('0.2 MB');
  });
});

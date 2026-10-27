import { GraphDocumentV2, ModelGraph, ModelNode, SavedGraph } from '../types';
import { readVerification } from './verificationRecord';

/**
 * Derived per-project metrics: layer count, parameter count, model size.
 *
 * The redesign's project cards and the projects table carry params and size
 * beside the layer count. Neither is persisted on the project record and no
 * endpoint serves them for a LIST of projects — asking the backend per row would
 * be one request per project on every page load. Every weighted layer's shape is
 * already in the saved graph, so the count is derivable on the client from data
 * we have.
 *
 * This mirrors `_calculate_model_summary` in backend/server_v2.py, which is
 * the same estimation the builder already shows, extended to the other weighted
 * layers. It is an ESTIMATE — that is why the UI labels it as one and why
 * unknown layers contribute 0 rather than a guess.
 */

const BYTES_PER_FLOAT32 = 4;

function num(value: unknown, fallback = 0): number {
  const n = typeof value === 'string' ? Number(value) : value;
  return typeof n === 'number' && Number.isFinite(n) ? n : fallback;
}

/** kernel_size arrives as 3 or [3, 3] or "3" depending on how the node was made. */
function kernelVolume(value: unknown, rank: number): number {
  if (Array.isArray(value)) {
    return value.reduce<number>((acc, v) => acc * Math.max(1, num(v, 1)), 1);
  }
  const k = Math.max(1, num(value, 1));
  return k ** rank;
}

function layerTypeName(node: ModelNode): string {
  return (
    node.data?.layerType?.name ||
    node.data?.layerType?.displayName ||
    ''
  );
}

/**
 * Trainable parameters for one layer. Returns 0 for layers that genuinely hold
 * no weights (activations, pooling, reshape) AND for layers we cannot compute —
 * the two are indistinguishable here on purpose, because inventing a number for
 * an unrecognised layer would make the total silently wrong.
 */
export function estimateNodeParams(node: ModelNode): number {
  const type = layerTypeName(node);
  const p = node.data?.parameters ?? {};

  switch (type) {
    case 'Linear': {
      const inF = num(p.in_features);
      const outF = num(p.out_features);
      return inF * outF + (p.bias === false ? 0 : outF);
    }
    case 'Conv1d':
    case 'Conv2d':
    case 'Conv3d':
    case 'ConvTranspose1d':
    case 'ConvTranspose2d':
    case 'ConvTranspose3d': {
      const rank = Number(type.slice(-2, -1)) || 2;
      const inC = num(p.in_channels);
      const outC = num(p.out_channels);
      const groups = Math.max(1, num(p.groups, 1));
      const weights = (inC / groups) * outC * kernelVolume(p.kernel_size, rank);
      return weights + (p.bias === false ? 0 : outC);
    }
    case 'Embedding':
      return num(p.num_embeddings) * num(p.embedding_dim);
    case 'BatchNorm1d':
    case 'BatchNorm2d':
    case 'BatchNorm3d':
    case 'InstanceNorm1d':
    case 'InstanceNorm2d':
      // weight + bias, one pair per channel.
      return p.affine === false ? 0 : num(p.num_features) * 2;
    case 'GroupNorm':
      return p.affine === false ? 0 : num(p.num_channels) * 2;
    case 'LayerNorm': {
      const shape = p.normalized_shape;
      const size = Array.isArray(shape)
        ? shape.reduce<number>((acc, v) => acc * Math.max(1, num(v, 1)), 1)
        : num(shape);
      return p.elementwise_affine === false ? 0 : size * 2;
    }
    case 'MultiheadAttention': {
      // in_proj (3 × d²) + out_proj (d²), biases included by default.
      const d = num(p.embed_dim);
      return d ? 4 * d * d + 4 * d : 0;
    }
    case 'LSTM':
    case 'GRU': {
      const inSize = num(p.input_size);
      const hidden = num(p.hidden_size);
      if (!inSize || !hidden) return 0;
      const gates = type === 'LSTM' ? 4 : 3;
      const layers = Math.max(1, num(p.num_layers, 1));
      const dirs = p.bidirectional ? 2 : 1;
      let total = 0;
      for (let layer = 0; layer < layers; layer += 1) {
        const layerIn = layer === 0 ? inSize : hidden * dirs;
        const perDir =
          gates * hidden * layerIn + gates * hidden * hidden + (p.bias === false ? 0 : 2 * gates * hidden);
        total += perDir * dirs;
      }
      return total;
    }
    default:
      return 0;
  }
}

export type GraphMetrics = {
  /**
   * Executable layers. Excludes the Input and Output markers and every
   * kind='module' grouping frame — a frame is canvas grouping, not a layer, and
   * counting it inflated every number the app showed.
   */
  layers: number;
  params: number;
  /** float32 bytes. */
  sizeBytes: number;
  /**
   * True when params came from torch via a recorded verify, false when this
   * module estimated them from the layer configs. The UI must not present an
   * estimate as a measurement.
   */
  exact: boolean;
};

const NON_LAYER_TYPES = new Set(['Input', 'Output']);

/**
 * The authoritative layer count is the document's own node kinds. The derived
 * ModelGraph is a projection that adds display frames, so counting it overcounts
 * — the trap this repo has already been burned by.
 */
function documentLayerCount(document?: GraphDocumentV2 | null): number | null {
  const nodes = document?.semantic?.nodes;
  if (!Array.isArray(nodes)) return null;
  return nodes.filter((node) => node?.kind === 'layer').length;
}

export function graphMetrics(
  graph?: Pick<ModelGraph, 'nodes'> | null,
  document?: GraphDocumentV2 | null,
): GraphMetrics {
  const nodes = graph?.nodes ?? [];
  let params = 0;
  let derivedLayers = 0;
  for (const node of nodes) {
    // A repeated stack keeps one representative child rendered and hides the
    // rest; counting the hidden copies would double-count a ×12 encoder.
    if (node.data?.repeatHidden) continue;
    const type = layerTypeName(node);
    const isFrame = node.data?.isGroup || node.data?.type === 'module' || node.data?.type === 'group';
    if (!NON_LAYER_TYPES.has(type) && !isFrame) derivedLayers += 1;
    const repeat = Math.max(1, num(node.data?.repeatCount, 1));
    params += estimateNodeParams(node) * repeat;
  }
  const layers = documentLayerCount(document) ?? derivedLayers;
  return { layers, params, sizeBytes: params * BYTES_PER_FLOAT32, exact: false };
}

/**
 * Per-project metrics, preferring a recorded verify over this module's estimate.
 *
 * `verification` is written by the builder's verify path into the graph
 * document's semantic metadata (see recordVerification), which round-trips
 * through graphDocumentAdapter untouched — so a project that has been verified
 * carries torch's own layer count, parameter total and size, and one that has
 * not falls back to the estimate above.
 */
export function projectMetrics(project: Pick<SavedGraph, 'graph' | 'document'>): GraphMetrics {
  const estimate = graphMetrics(project.graph, project.document);
  const verified = readVerification(project.document);
  if (!verified?.summary) return estimate;
  const { layerCount, params, sizeMb } = verified.summary;
  return {
    layers: typeof layerCount === 'number' ? layerCount : estimate.layers,
    params: typeof params === 'number' ? params : estimate.params,
    sizeBytes:
      typeof sizeMb === 'number' ? Math.round(sizeMb * 1024 * 1024) : estimate.sizeBytes,
    exact: true,
  };
}

/** 11.7M, 52.6K, 124M — the design's compact params notation. */
export function formatParams(params: number): string {
  if (params <= 0) return '—';
  if (params >= 1e9) return `${(params / 1e9).toFixed(params / 1e9 >= 10 ? 0 : 1)}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(params / 1e6 >= 100 ? 0 : 1)}M`;
  if (params >= 1e3) return `${(params / 1e3).toFixed(params / 1e3 >= 100 ? 0 : 1)}K`;
  return String(params);
}

/** 44.6 MB, 507 MB, 2.6 KB — the design's size notation. */
export function formatSize(bytes: number): string {
  if (bytes <= 0) return '—';
  const mb = bytes / (1024 * 1024);
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  if (mb >= 100) return `${Math.round(mb)} MB`;
  // Below a tenth of a megabyte, "0.0 MB" reads as "no size" — which is the one
  // thing it does not mean. A small template really is a few kilobytes.
  if (mb < 0.1) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${mb.toFixed(1)} MB`;
}

import { createWriteStream } from 'node:fs';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';

export interface HubLink { repo: string; revision: string; file?: string }
export interface WeightTensor { name: string; shape: number[]; dtype: string; elements: number; bytes: number }
export interface HubWeights extends HubLink {
  file: string; commit: string; url: string; fileBytes: number; headerBytes: number;
  tensors: WeightTensor[]; config?: Record<string, unknown>; configNote?: string;
}
export interface ExampleInput { name: string; shape: number[]; dtype: 'float32' | 'int64' | 'bool'; fill: 'zeros' | 'ones' }
export type HubFetch = typeof fetch;
const MAX_HEADER = 8 * 1024 * 1024;
export const MAX_DOWNLOAD = 2 * 1024 * 1024 * 1024;

function segments(value: string): string[] {
  const parts = value.split('/');
  if (parts.some(part => !part || part === '.' || part === '..' || /[\\\x00-\x1f\x7f]/.test(part))) {
    throw new Error('Use a Hugging Face model URL with a valid repository, revision and file path.');
  }
  return parts;
}

export function parseHubLink(value: string): HubLink {
  let url: URL;
  try { url = new URL(value.trim()); } catch { throw new Error('Paste a https://huggingface.co model or safetensors file URL.'); }
  if (url.protocol !== 'https:' || url.hostname !== 'huggingface.co' || url.port || url.username || url.password) {
    throw new Error('Use a https://huggingface.co model URL without credentials.');
  }
  const parts = url.pathname.replace(/^\/|\/$/g, '').split('/').map(decodeURIComponent);
  if (parts.length < 2 || parts.slice(0, 2).some(part => !/^[\w.-]+$/.test(part) || ['.', '..'].includes(part))
      || ['datasets', 'spaces', 'api'].includes(parts[0])) throw new Error('Use a Hugging Face model repository URL.');
  const repo = parts.slice(0, 2).join('/');
  let revision = 'main', file: string | undefined;
  if (parts.length > 2) {
    if (!['tree', 'blob', 'resolve'].includes(parts[2]) || !parts[3]) throw new Error('Use a repository, tree, blob or resolve URL.');
    revision = parts[3];
    segments(revision);
    file = parts[2] === 'tree' ? url.searchParams.get('show_file_info') ?? undefined : parts.slice(4).join('/');
    if (parts[2] === 'tree' && !file && parts.length > 4) throw new Error('Select a safetensors file in this folder and paste its file URL.');
  }
  if (file) {
    segments(file);
    if (!file.endsWith('.safetensors')) throw new Error('This importer supports .safetensors. A .pt/.bin checkpoint needs its model code; use Open as Graph with a local checkpoint recipe.');
  } else if (parts[2] === 'blob' || parts[2] === 'resolve') throw new Error('The file URL is missing a safetensors filename.');
  return { repo, revision, ...(file ? { file } : {}) };
}

const fileUrl = (link: HubLink, commit: string, file: string) =>
  `https://huggingface.co/${link.repo}/resolve/${encodeURIComponent(commit)}/${file.split('/').map(encodeURIComponent).join('/')}`;

export class HubAccessError extends Error {
  constructor() { super('Hugging Face requires access to these files. Accept the model’s access conditions on Hugging Face, then provide a read token.'); }
}

async function request(url: string, token: string | undefined, signal: AbortSignal, fetcher: HubFetch,
  headers: Record<string, string> = {}): Promise<Response> {
  const response = await fetcher(url, { signal, headers: { ...headers, ...(token ? { Authorization: `Bearer ${token}` } : {}) } });
  if (!response.ok) {
    await response.body?.cancel();
    if (response.status === 401 || response.status === 403) throw new HubAccessError();
    throw new Error(response.status === 404 ? 'The Hugging Face repository, revision or file was not found.' : `Hugging Face request failed (${response.status}). Try again.`);
  }
  return response;
}

async function boundedBody(response: Response, limit: number): Promise<Buffer> {
  if (!response.body) throw new Error('Hugging Face returned an empty response.');
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.length;
      if (bytes > limit) throw new Error('The remote metadata exceeds the supported size.');
      chunks.push(value);
    }
  } finally { await reader.cancel(); }
  return Buffer.concat(chunks);
}

async function readJson(url: string, token: string | undefined, signal: AbortSignal, fetcher: HubFetch): Promise<Record<string, unknown>> {
  const data = JSON.parse((await boundedBody(await request(url, token, signal, fetcher), MAX_HEADER)).toString('utf8'));
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('The remote metadata is not a JSON object.');
  return data;
}

async function range(url: string, start: number, end: number, token: string | undefined, signal: AbortSignal, fetcher: HubFetch) {
  const response = await request(url, token, signal, fetcher, { Range: `bytes=${start}-${end}`, 'Accept-Encoding': 'identity' });
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('content-range') ?? '');
  if (response.status !== 206 || !match || Number(match[1]) !== start || Number(match[2]) !== end
      || !Number.isSafeInteger(Number(match[3])) || Number(match[3]) <= end) {
    await response.body?.cancel();
    throw new Error('The server did not return the requested header range. No full weight download was attempted.');
  }
  const body = await boundedBody(response, end - start + 1);
  if (body.length !== end - start + 1) throw new Error('The safetensors header is incomplete.');
  return { body, total: Number(match[3]) };
}

export function parseWeightHeader(body: Buffer, dataBytes: number): WeightTensor[] {
  const header = JSON.parse(body.toString('utf8'));
  if (!header || typeof header !== 'object' || Array.isArray(header)) throw new Error('Invalid safetensors header.');
  const tensors: (WeightTensor & { start: number; end: number })[] = [];
  for (const [name, entry] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const item = entry as { dtype?: unknown; shape?: unknown; data_offsets?: unknown } | null;
    if (!name || name.length > 2048 || name.split('.').length > 64 || !item || typeof item.dtype !== 'string'
        || !/^[A-Z0-9_]{1,24}$/.test(item.dtype) || !Array.isArray(item.shape) || item.shape.length > 32
        || item.shape.some(d => !Number.isSafeInteger(d) || d < 0) || !Array.isArray(item.data_offsets)
        || item.data_offsets.length !== 2 || item.data_offsets.some(n => !Number.isSafeInteger(n) || n < 0)) {
      throw new Error('Invalid tensor shape, dtype or offsets in the safetensors header.');
    }
    const [start, end] = item.data_offsets;
    const elements = item.shape.reduce((a, b) => a * b, 1);
    if (end < start || end > dataBytes || !Number.isSafeInteger(elements)) throw new Error('Invalid tensor size in the safetensors header.');
    const width: Record<string, number> = { BOOL: 1, U8: 1, I8: 1, F8_E4M3: 1, F8_E5M2: 1, I16: 2, U16: 2, F16: 2, BF16: 2, I32: 4, U32: 4, F32: 4, I64: 8, U64: 8, F64: 8 };
    if (width[item.dtype] && elements * width[item.dtype] !== end - start) throw new Error('The tensor shape and dtype disagree with its stored bytes.');
    tensors.push({ name, shape: item.shape, dtype: item.dtype, elements, bytes: end - start, start, end });
    if (tensors.length > 100000) throw new Error('The weight explorer supports up to 100,000 tensors per file.');
  }
  let offset = 0;
  for (const tensor of [...tensors].sort((a, b) => a.start - b.start || a.end - b.end)) {
    if (tensor.start !== offset) throw new Error('The safetensors tensor offsets overlap or contain gaps.');
    offset = tensor.end;
  }
  if (offset !== dataBytes) throw new Error('The safetensors header does not describe the complete file.');
  return tensors.map(({ start, end, ...tensor }) => tensor);
}

export async function inspectHubWeights(link: HubLink, signal: AbortSignal, token?: string, fetcher: HubFetch = fetch): Promise<HubWeights> {
  const info = await readJson(`https://huggingface.co/api/models/${link.repo}/revision/${encodeURIComponent(link.revision)}`, token, signal, fetcher);
  if (typeof info.sha !== 'string' || !/^[a-f0-9]{40}$/.test(info.sha)) throw new Error('Hugging Face did not return an immutable model revision.');
  const names = Array.isArray(info.siblings) ? info.siblings.map(item => (item as { rfilename?: unknown })?.rfilename).filter((f): f is string => typeof f === 'string') : [];
  const candidates = names.filter(name => name.endsWith('.safetensors'));
  const file = link.file ?? (names.includes('model.safetensors') ? 'model.safetensors' : candidates.length === 1 ? candidates[0] : undefined);
  if (!file) throw new Error('Select a .safetensors file from the repository’s Files tab and paste its URL. Sharded checkpoints need their complete model setup for execution.');
  segments(file);
  if (!names.includes(file)) throw new Error('The selected safetensors file is not present at this revision.');
  const url = fileUrl(link, info.sha, file);
  const prefix = await range(url, 0, 7, token, signal, fetcher);
  const length = prefix.body.readBigUInt64LE();
  if (length < 2n || length > BigInt(MAX_HEADER) || length + 8n > BigInt(prefix.total)) throw new Error('The safetensors header size is invalid or exceeds 8 MiB.');
  const header = await range(url, 8, 7 + Number(length), token, signal, fetcher);
  if (header.total !== prefix.total) throw new Error('The remote file changed while its header was read.');
  const tensors = parseWeightHeader(header.body, prefix.total - 8 - Number(length));
  const result: HubWeights = { repo: link.repo, revision: link.revision, file, commit: info.sha, url,
    fileBytes: prefix.total, headerBytes: Number(length) + 8, tensors };
  const configFile = `${file.slice(0, file.lastIndexOf('/') + 1)}config.json`;
  if (names.includes(configFile)) result.config = await readJson(fileUrl(link, info.sha, configFile), token, signal, fetcher);
  else result.configNote = 'No config.json beside this file. Use the model’s Python implementation to capture a forward pass.';
  return result;
}

export function architectureFor(weights: HubWeights): string | undefined {
  const architectures = weights.config?.architectures;
  if (weights.config?.auto_map || weights.config?.quantization_config || /-\d+-of-\d+\.safetensors$/.test(weights.file)) return undefined;
  return Array.isArray(architectures) && architectures.length === 1 && typeof architectures[0] === 'string'
    && /^[A-Z][A-Za-z0-9]{0,100}$/.test(architectures[0]) ? architectures[0] : undefined;
}

export function exampleInputs(config?: Record<string, unknown>): ExampleInput[] {
  const type = config?.model_type;
  if (['bert', 'roberta', 'distilbert', 'albert', 'electra', 'gpt2', 'llama', 'mistral', 'qwen2', 'gemma', 'gemma2'].includes(String(type))) {
    const limit = config?.max_position_embeddings;
    const length = typeof limit === 'number' && Number.isSafeInteger(limit) && limit > 0 ? Math.min(16, limit) : 16;
    return [{ name: 'input_ids', shape: [1, length], dtype: 'int64', fill: 'zeros' },
      { name: 'attention_mask', shape: [1, length], dtype: 'int64', fill: 'ones' }];
  }
  if (['vit', 'deit', 'swin'].includes(String(type))) {
    const size = config?.image_size;
    const dims = typeof size === 'number' ? [size, size] : Array.isArray(size) ? size : [];
    if (dims.length === 2 && dims.every(n => Number.isInteger(n) && n > 0 && n <= 1024)) {
      return [{ name: 'pixel_values', shape: [1, Number(config?.num_channels ?? 3), ...dims], dtype: 'float32', fill: 'zeros' }];
    }
  }
  return [];
}

const PYTHON_KEYWORDS = new Set('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self'.split(' '));
export function validateExampleInputs(value: unknown): ExampleInput[] {
  if (!Array.isArray(value) || !value.length || value.length > 16) throw new Error('Add 1–16 named example tensors.');
  const names = new Set<string>();
  let total = 0;
  for (const input of value) {
    if (!input || typeof input.name !== 'string' || !/^[a-zA-Z_][a-zA-Z0-9_]{0,63}$/.test(input.name)
        || PYTHON_KEYWORDS.has(input.name) || names.has(input.name) || !['int64', 'float32', 'bool'].includes(input.dtype)
        || !['zeros', 'ones'].includes(input.fill) || !Array.isArray(input.shape) || input.shape.length > 8
        || input.shape.some((n: unknown) => !Number.isSafeInteger(n) || Number(n) < 1 || Number(n) > 2147483647)) throw new Error('Each input needs a unique Python argument name, positive shape dimensions, dtype and fill.');
    names.add(input.name);
    total += input.shape.reduce((a: number, b: number) => a * b, 1);
  }
  if (total > 4194304) throw new Error('Example inputs exceed 4,194,304 elements. Use a smaller example.');
  return value.map(({ name, shape, dtype, fill }) => ({ name, shape, dtype, fill }));
}

/** Only installed Transformers implementations; downloaded Python and pickle are never loaded. */
export function hubModelFiles(weights: HubWeights, value: unknown): { model: string; profile: string } {
  const architecture = architectureFor(weights);
  if (!architecture) throw new Error('This checkpoint needs an explicit model implementation. Open its local Python model with a checkpoint recipe.');
  const inputs = validateExampleInputs(value);
  const model = `"""Hugging Face weights imported by tensorViz. Example inputs are in tensorviz.toml."""\nfrom pathlib import Path\nimport torch.nn as nn\nimport transformers\nfrom transformers import PreTrainedModel\n\n\nclass HuggingFaceModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        architecture = getattr(transformers, ${JSON.stringify(architecture)}, None)\n        if not isinstance(architecture, type) or not issubclass(architecture, PreTrainedModel):\n            raise ValueError("The installed Transformers library does not provide ${architecture}. Open this model with its local Python implementation and checkpoint recipe.")\n        self.model, loading = architecture.from_pretrained(\n            Path(__file__).parent / "weights", local_files_only=True,\n            use_safetensors=True, trust_remote_code=False, output_loading_info=True,\n        )\n        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):\n            raise ValueError("Checkpoint does not exactly match ${architecture}: " + str(loading))\n\n    def forward(self, ${inputs.map(input => input.name).join(', ')}):\n        return self.model(${inputs.map(input => `${input.name}=${input.name}`).join(', ')})\n`;
  const profile = `# Synthetic example chosen in the Hugging Face weight explorer.\n# This capture covers this input and eval mode only.\nversion = 1\n\n[profiles.example]\nkind = "model"\nmodel = "model:HuggingFaceModel"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\nseed = 0\ntracked_files = ["weights/config.json", "weights/model.safetensors"]\ninit = { args = [], kwargs = {} }\n\n[profiles.example.inputs]\nargs = []\n\n[profiles.example.inputs.kwargs]\n${inputs.map(input => `${input.name} = { kind = "tensor", shape = ${JSON.stringify(input.shape)}, dtype = "${input.dtype}", requires_grad = false, generator = { kind = "full", value = ${input.dtype === 'bool' ? String(input.fill === 'ones') : input.fill === 'ones' ? '1' : '0'} } }`).join('\n')}\n`;
  return { model, profile };
}

export async function downloadHubWeights(weights: HubWeights, destination: string, signal: AbortSignal, token?: string,
  progress?: (bytes: number) => void, fetcher: HubFetch = fetch): Promise<void> {
  if (weights.fileBytes > MAX_DOWNLOAD) throw new Error('This importer downloads files up to 2 GiB. Use a local model setup for larger checkpoints.');
  const response = await request(weights.url, token, signal, fetcher, { 'Accept-Encoding': 'identity' });
  if (!response.body || response.status !== 200) { await response.body?.cancel(); throw new Error('The full checkpoint download was not available.'); }
  let bytes = 0;
  const count = new Transform({ transform(chunk: Buffer, _encoding, callback) {
    bytes += chunk.length;
    if (bytes > weights.fileBytes) return callback(new Error('The downloaded checkpoint exceeds its inspected size.'));
    progress?.(bytes);
    callback(null, chunk);
  } });
  await pipeline(Readable.fromWeb(response.body as import('node:stream/web').ReadableStream), count,
    createWriteStream(destination, { flags: 'wx' }), { signal });
  if (bytes !== weights.fileBytes) throw new Error('The checkpoint download is incomplete.');
}

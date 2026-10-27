import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { architectureFor, downloadHubWeights, exampleInputs, HubAccessError, hubModelFiles,
  inspectHubWeights, parseHubLink, parseWeightHeader, validateExampleInputs, type HubFetch, type HubWeights } from '../../huggingFace';

const commit = 'a'.repeat(40);
const header = Buffer.from(JSON.stringify({ 'encoder.weight': { dtype: 'F32', shape: [2, 3], data_offsets: [0, 24] },
  'encoder.bias': { dtype: 'F32', shape: [2], data_offsets: [24, 32] } }));
const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(header.length));
const fileBytes = 8 + header.length + 32;
const fixture: HubWeights = { repo: 'owner/model', revision: 'main', file: 'model.safetensors', commit,
  url: `https://huggingface.co/owner/model/resolve/${commit}/model.safetensors`, fileBytes, headerBytes: 8 + header.length,
  tensors: parseWeightHeader(header, 32), config: { architectures: ['BertModel'], model_type: 'bert' } };
function network(override?: (url: string, init?: RequestInit) => Response | undefined) {
  const seen: Array<{ url: string; init?: RequestInit }> = [];
  const fetcher: HubFetch = async (input, init) => {
    const url = String(input); seen.push({ url, init });
    const response = override?.(url, init); if (response) return response;
    if (url.includes('/api/models/')) return Response.json({ sha: commit, siblings: [{ rfilename: 'model.safetensors' }, { rfilename: 'config.json' }] });
    if (url.endsWith('config.json')) return Response.json(fixture.config);
    const range = (init?.headers as Record<string, string>)?.Range;
    const body = range === 'bytes=0-7' ? prefix : header;
    const start = range === 'bytes=0-7' ? 0 : 8;
    return new Response(body, { status: 206, headers: { 'content-range': `bytes ${start}-${start + body.length - 1}/${fileBytes}` } });
  };
  return { seen, fetcher };
}

suite('Hugging Face weight import', () => {
  test('accepts the HF file-info link as well as repository, blob and download links', () => {
    assert.deepEqual(parseHubLink('https://huggingface.co/vogent/Vogent-Turn-80M/tree/main?show_file_info=model.safetensors'),
      { repo: 'vogent/Vogent-Turn-80M', revision: 'main', file: 'model.safetensors' });
    assert.deepEqual(parseHubLink('https://huggingface.co/owner/model'), { repo: 'owner/model', revision: 'main' });
    for (const route of ['blob', 'resolve']) assert.deepEqual(parseHubLink(`https://huggingface.co/owner/model/${route}/refs%2Fpr%2F2/sub/model.safetensors?download=true`),
      { repo: 'owner/model', revision: 'refs/pr/2', file: 'sub/model.safetensors' });
  });
  test('rejects unrelated hosts, credentials, non-model URLs and pickle imports', () => {
    for (const url of ['http://huggingface.co/a/b', 'https://huggingface.co.attacker.test/a/b', 'https://token@huggingface.co/a/b',
      'https://huggingface.co:9000/a/b', 'https://huggingface.co/spaces/a/b', 'https://huggingface.co/datasets/a/b',
      'https://huggingface.co/a/b/tree/main?show_file_info=../model.safetensors']) assert.throws(() => parseHubLink(url));
    assert.throws(() => parseHubLink('https://huggingface.co/a/b/blob/main/weights.pt'), /local checkpoint recipe/);
  });
  test('pins metadata reads and fetches two exact ranges, never the whole checkpoint', async () => {
    const { fetcher, seen } = network();
    const result = await inspectHubWeights(parseHubLink('https://huggingface.co/owner/model'), new AbortController().signal, 'test-token', fetcher);
    assert.equal(result.commit, commit); assert.equal(result.fileBytes, fileBytes);
    assert.deepEqual(result.tensors.map(t => [t.name, t.shape, t.elements, t.bytes]), [['encoder.weight', [2, 3], 6, 24], ['encoder.bias', [2], 2, 8]]);
    assert.equal(seen.length, 4);
    assert.equal(seen[1].url, fixture.url);
    assert.deepEqual(seen.slice(1, 3).map(r => (r.init!.headers as Record<string, string>).Range), ['bytes=0-7', `bytes=8-${7 + header.length}`]);
    assert.equal((seen[0].init!.headers as Record<string, string>).Authorization, 'Bearer test-token');
    assert.ok(!JSON.stringify(result).includes('test-token'));
  });
  test('refuses a full-body response to a range request before reading it', async () => {
    let cancelled = false;
    const { fetcher } = network(url => url.endsWith('.safetensors') ? new Response(new ReadableStream({ cancel() { cancelled = true; } })) : undefined);
    await assert.rejects(inspectHubWeights(fixture, new AbortController().signal, undefined, fetcher), /requested header range/);
    assert.ok(cancelled);
  });
  test('rejects oversized, truncated or wrong-position headers', async () => {
    const huge = Buffer.alloc(8); huge.writeBigUInt64LE(1000000000n);
    const oversized = network(url => url.endsWith('.safetensors') ? new Response(huge, { status: 206, headers: { 'content-range': `bytes 0-7/${fileBytes}` } }) : undefined);
    await assert.rejects(inspectHubWeights(fixture, new AbortController().signal, undefined, oversized.fetcher), /header size/);
    const wrong = network(url => url.endsWith('.safetensors') ? new Response(prefix, { status: 206, headers: { 'content-range': `bytes 1-8/${fileBytes}` } }) : undefined);
    await assert.rejects(inspectHubWeights(fixture, new AbortController().signal, undefined, wrong.fetcher), /requested header range/);
    const truncated = network(url => url.endsWith('.safetensors') ? new Response(prefix.subarray(0, 7), { status: 206, headers: { 'content-range': `bytes 0-7/${fileBytes}` } }) : undefined);
    await assert.rejects(inspectHubWeights(fixture, new AbortController().signal, undefined, truncated.fetcher), /incomplete/);
  });
  test('distinguishes gated access from malformed or missing models without exposing server bodies', async () => {
    for (const status of [401, 403]) {
      const { fetcher } = network(() => new Response('secret server detail', { status }));
      await assert.rejects(inspectHubWeights(fixture, new AbortController().signal, undefined, fetcher), HubAccessError);
    }
    const { fetcher } = network(() => new Response('secret server detail', { status: 404 }));
    await assert.rejects(inspectHubWeights(fixture, new AbortController().signal, undefined, fetcher), /not found/);
  });
  test('a checkpoint without config remains inspectable but cannot invent a forward pass', async () => {
    const { fetcher } = network(url => url.includes('/api/models/') ? Response.json({ sha: commit, siblings: [{ rfilename: 'model.safetensors' }] }) : undefined);
    const result = await inspectHubWeights(fixture, new AbortController().signal, undefined, fetcher);
    assert.equal(result.tensors.length, 2); assert.equal(architectureFor(result), undefined);
    assert.match(result.configNote!, /Python implementation/);
  });
  test('does not pick an arbitrary shard or silently change the selected file', async () => {
    const { fetcher } = network(url => url.includes('/api/models/') ? Response.json({ sha: commit, siblings: [{ rfilename: 'a.safetensors' }, { rfilename: 'b.safetensors' }] }) : undefined);
    await assert.rejects(inspectHubWeights({ repo: 'owner/model', revision: 'main' }, new AbortController().signal, undefined, fetcher), /Select a .safetensors/);
    await assert.rejects(inspectHubWeights(fixture, new AbortController().signal, undefined, fetcher), /not present/);
    assert.equal(architectureFor({ ...fixture, file: 'model-00001-of-00002.safetensors' }), undefined);
  });
  test('rejects invalid shapes, overlapping ranges and wrong tensor byte counts', () => {
    for (const tensor of [{ dtype: 'F32', shape: [-1], data_offsets: [0, 4] }, { dtype: 'F32', shape: [2], data_offsets: [0, 4] },
      { dtype: 'F32', shape: [1], data_offsets: [1, 5] }]) assert.throws(() => parseWeightHeader(Buffer.from(JSON.stringify({ x: tensor })), 5));
    assert.throws(() => parseWeightHeader(Buffer.from(JSON.stringify({ a: { dtype: 'F32', shape: [1], data_offsets: [0, 4] },
      b: { dtype: 'F32', shape: [1], data_offsets: [0, 4] } })), 4), /overlap/);
    assert.deepEqual(parseWeightHeader(Buffer.from('{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}'), 4)[0].elements, 1);
  });
  test('preserves explicit input shapes and saves a real profile with no remote code or pickle execution', async () => {
    const { parse } = await import('smol-toml');
    const inputs = [{ name: 'input_ids', shape: [2, 8], dtype: 'int64', fill: 'zeros' }, { name: 'attention_mask', shape: [2, 8], dtype: 'bool', fill: 'ones' }];
    const result = hubModelFiles(fixture, inputs);
    const profile = (parse(result.profile) as any).profiles.example;
    assert.deepEqual(profile.inputs.kwargs.input_ids.shape, [2, 8]);
    // smol-toml 1.9 parses tables as null-prototype objects. Compare the values.
    assert.deepEqual(JSON.parse(JSON.stringify(profile.inputs.kwargs.attention_mask.generator)), { kind: 'full', value: true });
    assert.equal(profile.inputs.kwargs.input_ids.requires_grad, false);
    assert.equal(profile.mode, 'eval'); assert.equal(profile.device, 'cpu');
    assert.deepEqual(profile.tracked_files, ['weights/config.json', 'weights/model.safetensors']);
    assert.match(result.model, /local_files_only=True/); assert.match(result.model, /trust_remote_code=False/);
    assert.match(result.model, /use_safetensors=True/); assert.match(result.model, /missing_keys.*unexpected_keys.*mismatched_keys/);
    assert.match(result.model, /def forward\(self, input_ids, attention_mask\)/);
    assert.ok(!result.model.includes('torch.load'));
  });
  test('does not generate executable code from custom, ambiguous, quantized or injected architecture declarations', () => {
    for (const config of [{ architectures: ['A', 'B'] }, { architectures: ['A\nimport os'] }, { architectures: ['BertModel'], auto_map: {} },
      { architectures: ['BertModel'], quantization_config: {} }, {}]) assert.throws(() => hubModelFiles({ ...fixture, config }, exampleInputs(fixture.config)), /explicit model/);
  });
  test('bounds allocations and rejects duplicate, keyword or injected input names', () => {
    const input = { name: 'x', shape: [1, 4], dtype: 'float32', fill: 'zeros' };
    for (const name of ['x);print(1)', 'self', 'class']) assert.throws(() => validateExampleInputs([{ ...input, name }]));
    assert.throws(() => validateExampleInputs([input, input]));
    assert.throws(() => validateExampleInputs([{ ...input, shape: [4194305] }]), /exceed/);
    assert.throws(() => validateExampleInputs([{ ...input, shape: [0, 4] }]));
    assert.deepEqual(exampleInputs({ model_type: 'llama', max_position_embeddings: 4 })[0].shape, [1, 4]);
    assert.deepEqual(exampleInputs({ model_type: 'unknown' }), []);
  });
  test('downloads exact bytes and never overwrites an existing checkpoint', async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-hf-test-'));
    const file = path.join(root, 'model.safetensors');
    const full = Buffer.concat([prefix, header, Buffer.alloc(32)]);
    try {
      const fetcher: HubFetch = async () => new Response(full);
      await downloadHubWeights(fixture, file, new AbortController().signal, undefined, undefined, fetcher);
      assert.deepEqual(await fs.readFile(file), full);
      await assert.rejects(downloadHubWeights(fixture, file, new AbortController().signal, undefined, undefined, fetcher), /EEXIST/);
      await assert.rejects(downloadHubWeights(fixture, path.join(root, 'short'), new AbortController().signal, undefined, undefined,
        async () => new Response(full.subarray(0, 10))), /incomplete/);
      await assert.rejects(downloadHubWeights(fixture, path.join(root, 'large'), new AbortController().signal, undefined, undefined,
        async () => new Response(Buffer.alloc(full.length + 1))), /exceeds/);
    } finally { await fs.rm(root, { recursive: true, force: true }); }
  });
});

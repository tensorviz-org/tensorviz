import * as assert from 'node:assert/strict';

import { entriesFrom, fetchCatalog, stemFor, templateFor } from '../../catalog';

const CATEGORIES = {
  categories: [
    { id: 'starters', count: 2 },
    { id: 'generative', count: 1 },
  ],
};

const PER = new Map<string, unknown>([
  ['starters', {
    templates: {
      mlp_tiny: { name: 'Tiny MLP', description: 'Two-layer perceptron.' },
      mlp_dropout: { name: 'MLP with Dropout', description: 'With dropout.' },
    },
    popular_models: { resnet18: { name: 'ResNet-18', description: 'Residual net.' } },
    // Deliberately present and deliberately ignored — see entriesFrom.
    module_templates: { resnet_block: { name: 'Residual block' } },
  }],
  ['generative', {
    templates: { gan: { name: 'GAN', description: 'Generator and discriminator.' } },
  }],
]);

suite('catalog.entriesFrom', () => {
  test('offers the gallery starters and the popular architectures', () => {
    const entries = entriesFrom(CATEGORIES, PER);
    const ids = entries.map((e) => e.id).sort();
    assert.deepEqual(ids, ['gan', 'mlp_dropout', 'mlp_tiny', 'resnet18']);
  });

  test('leaves module blocks out: they are canvas insertions, not files', () => {
    // They carry an entry and an exit node and mean nothing as a standalone module —
    // and 15 of the ids that block advertises are not in the registry at all, so a
    // picker entry for one would be a card that 404s when chosen.
    const entries = entriesFrom(CATEGORIES, PER);
    assert.equal(entries.some((e) => e.id === 'resnet_block'), false);
  });

  test('remembers which endpoint each entry answers on', () => {
    const entries = entriesFrom(CATEGORIES, PER);
    assert.equal(entries.find((e) => e.id === 'resnet18')?.group, 'popular');
    assert.equal(entries.find((e) => e.id === 'mlp_tiny')?.group, 'template');
  });

  test('carries the category, because seventy-three items need narrowing', () => {
    const entries = entriesFrom(CATEGORIES, PER);
    assert.equal(entries.find((e) => e.id === 'gan')?.category, 'generative');
  });
});

suite('catalog.stemFor', () => {
  test('turns an id into a file name a person would have typed', () => {
    assert.equal(stemFor('mlp_tiny'), 'mlp_tiny');
    assert.equal(stemFor('ResNet-18'), 'resnet_18');
    assert.equal(stemFor('!!!'), 'model');
  });
});

suite('catalog.templateFor', () => {
  const host = (posts: Record<string, unknown>, seen: string[] = []) => ({
    origin: () => 'http://127.0.0.1:9999',
    headers: () => ({ 'X-tensorViz-Token': 't' }),
    get: async () => ({}),
    post: async (url: string, _h: Record<string, string>, body: unknown) => {
      seen.push(url);
      if (url.includes('generate-code')) {
        return { code: `class ${(body as { class_name: string }).class_name}(nn.Module): pass` };
      }
      return posts[url] ?? { graph: { root_nodes: [], root_edges: [] } };
    },
  });

  test('a popular model and a gallery template use different endpoints', async () => {
    const seen: string[] = [];
    await templateFor(host({}, seen), { id: 'resnet18', label: 'ResNet-18', description: '', group: 'popular' }, 'M');
    assert.ok(seen[0].includes('/api/popular-models/resnet18/instantiate'), seen[0]);

    const seen2: string[] = [];
    await templateFor(host({}, seen2), { id: 'gan', label: 'GAN', description: '', group: 'template' }, 'M');
    assert.ok(seen2[0].includes('/api/templates/gan/instantiate'), seen2[0]);
  });

  test('the class name reaches the generator, so the file agrees with itself', async () => {
    // The generator writes the class name INTO the source. Deciding it afterwards would
    // leave `class GeneratedModel` in a file the scaffolder called something else, and
    // "Open as Graph" would then look for a class that is not there.
    const template = await templateFor(
      host({}), { id: 'mlp_tiny', label: 'Tiny MLP', description: '', group: 'template' }, 'MyNet',
    );
    assert.ok(template.body.includes('class MyNet'), template.body);
  });

  test('refuses loudly when the backend generates nothing', async () => {
    const empty = { ...host({}), post: async (url: string) => (url.includes('generate-code') ? { code: '   ' } : { graph: {} }) };
    await assert.rejects(
      () => templateFor(empty, { id: 'x', label: 'X', description: '', group: 'template' }, 'M'),
      /generated no code/,
    );
  });

  test('says so when the backend is not up, rather than writing an empty file', async () => {
    const down = { ...host({}), origin: () => undefined };
    await assert.rejects(
      () => templateFor(down, { id: 'x', label: 'X', description: '', group: 'template' }, 'M'),
      /not running/,
    );
  });
});

suite('catalog.fetchCatalog', () => {
  test('an unreadable category costs that one, not the other four', async () => {
    const host = {
      origin: () => 'http://127.0.0.1:9999',
      headers: () => ({}),
      post: async () => ({}),
      get: async (url: string) => {
        if (url.endsWith('/categories')) return CATEGORIES;
        if (url.endsWith('/starters')) throw new Error('boom');
        return PER.get('generative');
      },
    };
    const entries = await fetchCatalog(host);
    assert.deepEqual(entries.map((e) => e.id), ['gan']);
  });

  test('no backend means an empty catalogue, not a thrown picker', async () => {
    const host = { origin: () => undefined, headers: () => ({}), post: async () => ({}), get: async () => ({}) };
    assert.deepEqual(await fetchCatalog(host), []);
  });
});

suite('catalog: canonical blocks', () => {
  const { placeBlock } = require('../../catalog');

  test('a block spec becomes nodes the generator accepts', () => {
    // The registry answers with ops and params, not a graph. Without the canvas's own
    // transformation the generator refuses it with `nodes.0.type: missing` — which is
    // what made every one of the 306 look broken when driven straight through.
    const placed = placeBlock({
      nodes: [
        { id: 'a', op: 'input', params: {} },
        { id: 'b', op: 'nn.Linear', params: { in_features: 8, out_features: 8 } },
        { id: 'c', op: 'output', params: {} },
      ],
      edges: [{ id: 'e1', source: 'a', target: 'b' }],
    });
    assert.deepEqual(placed.nodes.map((n: any) => n.type), ['inputNode', 'layerNode', 'outputNode']);
    assert.equal(placed.nodes[1].data.layerType.id, 'linear');
    assert.deepEqual(placed.nodes[1].data.parameters, { in_features: 8, out_features: 8 });
    assert.deepEqual(placed.nodes[0].position, { x: 0, y: 0 });
    assert.equal(placed.edges.length, 1);
  });

  test('the nn. and torch. prefixes are stripped, as the canvas strips them', () => {
    const placed = placeBlock({ nodes: [{ id: 'x', op: 'torch.nn.LayerNorm', params: {} }] });
    assert.equal(placed.nodes[0].data.layerType.id, 'layernorm');
  });

  test('an empty spec is empty, not a crash', () => {
    assert.deepEqual(placeBlock(undefined), { nodes: [], edges: [] });
  });
});

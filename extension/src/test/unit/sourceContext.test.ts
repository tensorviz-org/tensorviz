import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { LocalBrowserSourceEdits } from '../../localBrowserSourceEdits';
import { sha256 } from '../../sourceChanges';
import { McpProposals } from '../../mcpProposals';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import { editorState, reset, resetEditors, state } from '../vscodeStub';

const oracle = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../../tests/fixtures/contracts/source-context-oracle.json'), 'utf8'));
const query = oracle.queries[0].query;
suite('host source dependency context', () => {
  let root: string, host: LocalBrowserSourceEdits, model: McpModel, dependencies: Map<string, string>;
  let requests: Array<{ url: string; body: any; init: RequestInit }>, respond: (body: any) => Response | Promise<Response>;
  let origin: string, token: string;
  setup(() => {
    reset(); resetEditors();
    root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-context-test-')));
    dependencies = new Map();
    for (const source of oracle.sources) {
      fs.writeFileSync(path.join(root, source.file), source.source);
      editorState.documents.set(path.join(root, source.file), source.source);
      dependencies.set(source.file, source.digest);
    }
    model = { target: { workspaceFolder: root, file: 'model.py', cls: 'Model', documentVersion: 1,
      sourceDigest: dependencies.get('model.py')!, backendRevision: 1 }, current: true,
      projection: { document: { semantic: { nodes: [], edges: [] } } } as unknown as ProjectionPayload };
    requests = []; origin = 'http://127.0.0.1:9901'; token = 'private-context-token';
    respond = () => new Response(JSON.stringify(oracle.queries[0].page));
    host = new LocalBrowserSourceEdits({ origin: () => origin, token: () => token,
      context: target => model.current && JSON.stringify(target) === JSON.stringify(model.target) ? { projection: model.projection, dependencies } : undefined,
      channel: { appendLine() {} }, publish() {}, review: async () => { assert.fail('Analysis must not open a source review'); },
      fetchImpl: (async (url, init) => {
        const body = JSON.parse(String(init?.body)); requests.push({ url: String(url), body, init: init! });
        return respond(body);
      }) as typeof fetch,
    });
  });
  teardown(() => { host.dispose(); fs.rmSync(root, { recursive: true, force: true }); });

  test('the production proposal service sends only guarded buffers to the authenticated static endpoint', async () => {
    const graph = new McpGraphService('host-context', () => model), proposals = new McpProposals(host);
    const result = await proposals.querySourceContext(graph, graph.reference(model), query);
    assert.equal(result.evidence, 'source_only'); assert.equal(result.collection.omittedFiles, 0);
    assert.equal(requests[0].url, `${origin}/api/ide/source-context`);
    assert.equal((requests[0].init.headers as any)['X-tensorViz-Token'], token);
    assert.equal(requests[0].init.redirect, 'error');
    assert.deepEqual(requests[0].body.sources.map((item: any) => item.file), ['model.py', '_tensorviz_context_config.py']);
    assert.doesNotMatch(JSON.stringify(result), /private-context-token|pendingDraft/);
    assert.equal(editorState.appliedEdits.length, 0);
  });

  test('conditional configuration analysis uses the same guarded source reader and preserves unknown effects', async () => {
    const changeOracle = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../../tests/fixtures/contracts/source-change-oracle.json'), 'utf8'));
    dependencies.clear();
    for (const source of changeOracle.sources) {
      fs.writeFileSync(path.join(root, source.file), source.source);
      editorState.documents.set(path.join(root, source.file), source.source);
      dependencies.set(source.file, source.digest);
    }
    model.target.sourceDigest = dependencies.get('model.py')!;
    respond = () => new Response(JSON.stringify(changeOracle.queries[0].page));
    const result = await host.querySourceContext(model.target, changeOracle.queries[0].query);
    assert.equal(requests[0].body.query.assumeValue, 16);
    assert.equal(result.changeAnalysis?.evidence, 'conditional_source_arithmetic');
    assert.equal(result.changeAnalysis?.after, 16);
    assert.ok(result.items.some(item => 'configurationChange' in item && item.configurationChange?.kind === 'scalar'
      && item.configurationChange.status === 'unknown'));
    assert.equal(editorState.appliedEdits.length, 0);
  });

  for (const change of ['untrusted', 'unrecorded', 'traversal', 'symlink', 'changed-dependency', 'changed-root', 'oversized']) {
    test(`source query refuses ${change} before contacting the parser`, async () => {
      const file = path.join(root, query.file);
      if (change === 'untrusted') state.isTrusted = false;
      if (change === 'unrecorded') dependencies.delete(query.file);
      if (change === 'changed-dependency') editorState.documents.set(file, 'WIDTH = 16\n');
      if (change === 'changed-root') editorState.documents.set(path.join(root, 'model.py'), '# changed\n');
      if (change === 'oversized') {
        const text = '# ' + 'x'.repeat(262144); editorState.documents.set(file, text); dependencies.set(query.file, sha256(text));
      }
      if (change === 'symlink') { fs.renameSync(file, file + '.original'); fs.symlinkSync(file + '.original', file); }
      await assert.rejects(host.querySourceContext(model.target, { ...query, file: change === 'traversal' ? '../outside.py' : query.file }));
      assert.equal(requests.length, 0); assert.equal(editorState.appliedEdits.length, 0);
    });
  }

  for (const change of ['buffer', 'version', 'trust', 'target', 'inventory', 'digest', 'origin', 'token', 'symlink']) {
    test(`source context is withheld when ${change} changes during the query`, async () => {
      respond = async () => {
        await Promise.resolve();
        const file = path.join(root, query.file);
        if (change === 'buffer') editorState.documents.set(file, 'WIDTH = 16\n');
        if (change === 'version') editorState.versions.set(file, 2);
        if (change === 'trust') state.isTrusted = false;
        if (change === 'target') model.current = false;
        if (change === 'inventory') dependencies.set('new.py', 'f'.repeat(64));
        if (change === 'digest') dependencies.set(query.file, 'f'.repeat(64));
        if (change === 'origin') origin = 'http://127.0.0.1:9902';
        if (change === 'token') token = 'replacement';
        if (change === 'symlink') { fs.renameSync(file, file + '.original'); fs.symlinkSync(file + '.original', file); }
        return new Response(JSON.stringify(oracle.queries[0].page));
      };
      await assert.rejects(host.querySourceContext(model.target, query), /changed/);
      assert.equal(editorState.appliedEdits.length, 0);
    });
  }

  for (const failure of ['http', 'stream-limit', 'invalid-json', 'wrong-index', 'network']) {
    test(`parser ${failure} cannot leak private errors or certify a response`, async () => {
      respond = () => {
        if (failure === 'network') throw new Error('PRIVATE_SOURCE_SECRET');
        if (failure === 'http') return new Response('PRIVATE_SOURCE_SECRET', { status: 422 });
        if (failure === 'stream-limit') return new Response('x'.repeat(65537));
        if (failure === 'invalid-json') return new Response('PRIVATE_SOURCE_SECRET');
        return new Response(JSON.stringify({ ...oracle.queries[0].page, indexId: 'f'.repeat(64) }));
      };
      await assert.rejects(host.querySourceContext(model.target, query), error => {
        assert.doesNotMatch(String(error), /PRIVATE_SOURCE_SECRET/); return true;
      });
    });
  }

  test('known parser refusals remain actionable while unexpected error text stays private', async () => {
    respond = () => new Response(JSON.stringify({ detail: { reason: 'configuration_change_unavailable' } }), { status: 422 });
    await assert.rejects(host.querySourceContext(model.target, { ...query, assumeValue: 16 }), (error: any) => {
      assert.equal(error.code, 'CONFIGURATION_CHANGE_UNAVAILABLE'); assert.match(error.message, /integer assignment/); return true;
    });
    respond = () => new Response(JSON.stringify({ detail: { reason: 'source_index_changed', source: 'PRIVATE_SOURCE_SECRET' } }), { status: 422 });
    await assert.rejects(host.querySourceContext(model.target, query), (error: any) => {
      assert.equal(error.code, 'SOURCE_INDEX_CHANGED'); assert.match(error.message, /new symbol query/);
      assert.doesNotMatch(error.message, /PRIVATE_SOURCE_SECRET/); return true;
    });
    respond = () => new Response(JSON.stringify({ detail: { reason: '__proto__', message: 'PRIVATE_SOURCE_SECRET' } }), { status: 422 });
    await assert.rejects(host.querySourceContext(model.target, query), (error: any) => {
      assert.equal(error.code, 'SOURCE_CONTEXT_UNAVAILABLE'); assert.doesNotMatch(error.message, /PRIVATE_SOURCE_SECRET/); return true;
    });
  });

  test('file and byte limits report partial inventory while retaining the explicitly selected file', async () => {
    for (let i = 0; i < 20; i++) {
      const file = `dep${i.toString().padStart(2, '0')}.py`, source = '# ' + 'x'.repeat(200000) + '\n';
      fs.writeFileSync(path.join(root, file), source); editorState.documents.set(path.join(root, file), source); dependencies.set(file, sha256(source));
    }
    respond = body => {
      const files = [...body.sources].sort((a: any, b: any) => Buffer.compare(Buffer.from(a.file), Buffer.from(b.file))).map(({ file, digest }: any) => ({ file, digest }));
      const indexId = createHash('sha256').update(JSON.stringify([1, files.map((item: any) => [item.file, item.digest])])).digest('hex');
      // Empty page transport fixture: production parser content has a separate oracle.
      return new Response(JSON.stringify({ ...oracle.queries[0].page, files, indexId, items: [], total: 0, nextOffset: null,
        traversal: { entities: 0, boundaries: 0, omittedSeeds: 0 } }));
    };
    const result = await host.querySourceContext(model.target, { file: 'dep19.py', symbol: 'width' });
    const sent = requests[0].body.sources;
    assert.ok(sent.length <= 16); assert.ok(sent.some((item: any) => item.file === 'dep19.py'));
    assert.ok(sent.reduce((size: number, item: any) => size + Buffer.byteLength(item.source), 0) <= 1048576);
    assert.equal(result.collection.omittedFiles, dependencies.size - sent.length); assert.ok(result.collection.omittedFiles > 0);
  });
});

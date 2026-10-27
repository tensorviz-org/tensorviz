import * as assert from 'node:assert/strict';
import { checkCanvasHomeSetup } from '../../canvasHomeSetup';
import { canvasHomeHtml } from '../../canvasHome';
import { reset, state } from '../vscodeStub';

suite('automatic tensorViz runtime readiness', () => {
  setup(() => reset());
  teardown(() => reset());

  test('a ready owned runtime is sufficient, without querying selected project Python', async () => {
    let pythonQueries = 0;
    state.extensions.set('ms-python.python', { activate: async () => { pythonQueries++; throw new Error('incompatible project Pydantic'); } });
    state.workspaceFolders = [];
    const result = await checkCanvasHomeSetup({ signal: new AbortController().signal,
      runtime: async () => ({ state: 'ready', interpreter: '/tool/python' }) });
    assert.equal(result.phase, 'ready'); assert.equal(pythonQueries, 0);
    assert.deepEqual(Object.keys(result).sort(), ['detail', 'phase'], 'home exposes no project environment metadata');
    const html = canvasHomeHtml({ hasWorkspace: false, indexed: false, models: [], setup: result });
    assert.match(html, /tensorViz is ready/);
    assert.doesNotMatch(html, /Choose Python|pydantic|Model Python needs attention|Selected in VS Code/);
  });

  test('missing runtime offers owned setup without suggesting project Python', async () => {
    const result = await checkCanvasHomeSetup({ signal: new AbortController().signal,
      runtime: async () => ({ state: 'absent', why: 'not installed' }) });
    assert.equal(result.phase, 'needs_setup');
    const html = canvasHomeHtml({ hasWorkspace: true, indexed: false, models: [], setup: result });
    assert.match(html, /Set up tensorViz/); assert.doesNotMatch(html, /Choose Python|Missing:/);
  });

  test('broken runtime stays recoverable without echoing raw probe errors', async () => {
    const result = await checkCanvasHomeSetup({ signal: new AbortController().signal,
      runtime: async () => ({ state: 'broken', why: 'private diagnostic' }) });
    assert.equal(result.phase, 'needs_setup'); assert.match(result.detail, /needs repair/);
    assert.doesNotMatch(result.detail, /private diagnostic/);
  });

  test('unsupported runtime offers diagnostics rather than installation', async () => {
    const result = await checkCanvasHomeSetup({ signal: new AbortController().signal,
      runtime: async () => ({ state: 'unavailable', why: 'unsupported host' }) });
    const html = canvasHomeHtml({ hasWorkspace: true, indexed: false, models: [], setup: result });
    assert.equal(result.phase, 'unavailable'); assert.match(html, /Show log/);
    assert.doesNotMatch(html, /data-action="setup"|Choose Python/);
  });

  test('untrusted workspaces run no runtime probe', async () => {
    state.isTrusted = false;
    assert.equal((await checkCanvasHomeSetup({ signal: new AbortController().signal,
      runtime: async () => assert.fail('runtime probe') })).phase, 'untrusted');
  });

  test('already cancelled home checks run no runtime probe', async () => {
    const controller = new AbortController(); controller.abort();
    await assert.rejects(checkCanvasHomeSetup({ signal: controller.signal,
      runtime: async () => assert.fail('cancelled runtime probe') }), /abort/i);
  });

  test('cancelled checks cannot publish a late ready result', async () => {
    const controller = new AbortController();
    let finish!: (result: { state: 'ready'; interpreter: string }) => void;
    const checking = checkCanvasHomeSetup({ signal: controller.signal,
      runtime: () => new Promise(resolve => { finish = resolve; }) });
    controller.abort(); finish({ state: 'ready', interpreter: '/tool/python' });
    await assert.rejects(checking, /abort/i);
  });
});

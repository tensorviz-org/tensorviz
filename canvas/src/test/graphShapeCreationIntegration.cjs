/* global globalThis:readonly */
// Real Editor/store and production ShapeUtil integration. No timing assertions.
const fs = require('fs');
const Module = require('module');
const path = require('path');
const { JSDOM } = require('jsdom');

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { pretendToBeVisual: true, url: 'http://localhost/' });
for (const key of Object.getOwnPropertyNames(window)) {
  if (key in globalThis) continue;
  try { globalThis[key] = window[key]; } catch { /* read-only global */ }
}
// Node's own navigator and (since 25) storage globals would shadow jsdom's.
for (const key of ['window', 'document', 'navigator', 'localStorage', 'sessionStorage']) {
  Object.defineProperty(globalThis, key, { value: window[key], configurable: true, writable: true });
}
window.ResizeObserver = globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
window.matchMedia = globalThis.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });


const ts = require('typescript');
require.extensions['.tsx'] = require.extensions['.ts'] = (module, filename) => {
  const { outputText } = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX },
  });
  module._compile(outputText, filename);
};
require.extensions['.css'] = () => {};
const tldraw = require('tldraw');
const { ModuleShapeUtil } = require('../canvas/ModuleShapeUtil.tsx');
const { LayerShapeUtil } = require('../canvas/LayerShapeUtil.tsx');
const { InputShapeUtil } = require('../canvas/InputShapeUtil.tsx');
const { OutputShapeUtil } = require('../canvas/OutputShapeUtil.tsx');
const { ModelEdgeShapeUtil } = require('../canvas/ModelEdgeShapeUtil.tsx');
const shapeUtils = [...tldraw.defaultShapeUtils, ModuleShapeUtil, LayerShapeUtil, InputShapeUtil, OutputShapeUtil, ModelEdgeShapeUtil];
const assert = require('node:assert/strict');
const React = require('react');
const { createRoot } = require('react-dom/client');
const { act } = React;
const { createGraphShapes } = require('../canvas/graphShapeCreation.ts');
const { nodeIdToShapeId } = require('../canvas/canvasShapeId.ts');
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const editors = [], cleanups = [];
const sid = tldraw.createShapeId;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

function makeEditor(options = {}) {
  const container = document.createElement('div');
  container.getBoundingClientRect = () => ({ x: 0, y: 0, top: 0, left: 0, width: 1080, height: 720, right: 1080, bottom: 720 });
  document.body.appendChild(container);
  const editor = new tldraw.Editor({ shapeUtils, bindingUtils: tldraw.defaultBindingUtils,
    tools: [...tldraw.defaultTools, ...tldraw.defaultShapeTools],
    store: tldraw.createTLStore({ shapeUtils, bindingUtils: tldraw.defaultBindingUtils }),
    getContainer: () => container, initialState: 'select', ...options });
  editors.push(editor);
  editor.createPage({ id: 'page:fixture', name: 'Fixture' });
  editor.setCurrentPage('page:fixture');
  editor.updateViewportScreenBounds(container);
  editor.clearHistory();
  assert(editor.getShapeUtil('module-node') instanceof ModuleShapeUtil);
  assert(editor.getShapeUtil('model-edge') instanceof ModelEdgeShapeUtil);
  return editor;
}
const snapshot = editor => editor.getCurrentPageShapes().sort((a, b) => a.id.localeCompare(b.id));
const indices = editor => snapshot(editor).map(({ id, index, parentId }) => ({ id, index, parentId }));
function compareEditors(a, b) {
  assert.deepEqual(snapshot(a), snapshot(b));
  assert.deepEqual(a.getSortedChildIdsForParent(a.getCurrentPageId()), b.getSortedChildIdsForParent(b.getCurrentPageId()));
  for (const shape of snapshot(a)) {
    assert.deepEqual(a.getShapePageTransform(shape), b.getShapePageTransform(shape.id));
  }
  assert.equal(a.getCanUndo(), b.getCanUndo());
  assert.equal(a.getCanRedo(), b.getCanRedo());
  assert.equal(a.getIsReadonly(), b.getIsReadonly());
  assert.deepEqual(a.getSelectedShapeIds(), b.getSelectedShapeIds());
}
function diagnostics(editor) {
  const events = { edit: 0, created: [], edited: [], after: 0, complete: 0, maxShapes: [] };
  editor.on('max-shapes', event => events.maxShapes.push(event));
  editor.on('created-shapes', shapes => events.created.push(shapes.map(s => ({ ...s }))));
  editor.on('edit', () => events.edit++);
  editor.on('edited-shapes', shapes => events.edited.push(shapes.map(s => s.id)));
  editor.sideEffects.registerAfterChangeHandler('shape', () => events.after++);
  editor.sideEffects.registerOperationCompleteHandler(() => events.complete++);
  return events;
}

// Reconstruct only the two pre-optimization create calls in memory. An optional
// unchanged baseline source path enables independent local differential checks.
function loadBridge(baseline) {
  const candidatePath = path.resolve(__dirname, '../canvas/useTldrawBridge.ts');
  const filename = baseline && process.argv[2] ? path.resolve(process.argv[2]) : candidatePath;
  let source = fs.readFileSync(filename, 'utf8');
  if (baseline && !process.argv[2]) {
    assert.equal((source.match(/createGraphShapes\(editor, /g) ?? []).length, 2);
    source = source.replaceAll('createGraphShapes(editor, ', 'editor.createShapes(');
  }
  const loaded = new Module(filename, module);
  loaded.filename = filename;
  loaded.paths = Module._nodeModulePaths(path.dirname(filename));
  loaded._compile(ts.transpileModule(source, { fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  }).outputText, filename);
  return loaded.exports.useTldrawBridge;
}
async function mountBridge(useBridge, initialNodes, initialEdges, existingEditor, direction) {
  assert(['horizontal', 'vertical'].includes(direction), 'Use a production LayoutDirection');
  const editor = existingEditor ?? makeEditor();
  editor.updateInstanceState({ isReadonly: true }, { history: 'ignore' });
  let api, setNodes, setEdges, nodes;
  function Probe() {
    const [n, sn] = React.useState(initialNodes), [e, se] = React.useState(initialEdges);
    nodes = n; setNodes = sn; setEdges = se;
    api = useBridge({ nodes: n, edges: e, setNodes: sn, setEdges: se, direction });
    return null;
  }
  const host = document.createElement('div'); document.body.appendChild(host);
  const root = createRoot(host);
  await act(async () => { root.render(React.createElement(Probe)); });
  const events = diagnostics(editor);
  const backCalls = [];
  const originalBack = editor.sendToBack.bind(editor);
  editor.sendToBack = ids => { backCalls.push([...ids]); return originalBack(ids); };
  let disposeBridge;
  await act(async () => { disposeBridge = api.handleEditorReady(editor); });
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    disposeBridge?.();
    root.unmount();
    host.remove();
  };
  cleanups.push(dispose);
  await act(async () => { await pause(10); });
  const update = async (nextNodes, nextEdges) => {
    await act(async () => { if (nextNodes) setNodes(nextNodes); if (nextEdges) setEdges(nextEdges); });
    await act(async () => { await pause(80); });
  };
  return { editor, events, backCalls, update, dispose, get nodes() { return nodes; } };
}

const graphTypes = ['module-node', 'layer', 'input-node', 'output-node', 'model-edge'];
const partials = () => graphTypes.map((type, i) => ({ id: sid(`new-${i}`), type,
  x: -30.5 + i * 25, y: 12.25 + i * 10, rotation: i / 10, isLocked: i === 4, meta: { fixture: true } }));
const seed = (editor, shapes) => editor.createShapes(shapes.map(shape => ({ parentId: editor.getCurrentPageId(), ...shape })));
function freezeInputs(shapes) {
  for (const shape of shapes) { if (shape && typeof shape === 'object') Object.freeze(shape); }
  return Object.freeze(shapes);
}
function createPair(a, b, shapes = partials(), options = { history: 'ignore' }) {
  const before = JSON.stringify(shapes);
  freezeInputs(shapes);
  const events = [diagnostics(a), diagnostics(b)];
  let aError, bError;
  try { a.run(() => a.createShapes(shapes), options); } catch (error) { aError = error; }
  try { b.run(() => createGraphShapes(b, shapes), options); } catch (error) { bError = error; }
  assert.equal(bError?.message, aError?.message);
  compareEditors(a, b);
  assert.deepEqual(events[1], events[0]);
  assert.equal(JSON.stringify(shapes), before, 'inputs are not mutated');
  return aError;
}
function withPair(check, options) {
  const a = makeEditor(options), b = makeEditor(options);
  try { check(a, b); } finally { a.dispose(); b.dispose(); editors.splice(editors.indexOf(a), 2); }
}

function helperChecks() {
  const counts = [];
  // Do not wrap the receive method: doing so deliberately disables the guard.
  for (const size of [8, 32, 96]) withPair((a, b) => {
    const existing = Array.from({ length: size }, (_, i) => ({ id: sid(`old-${i}`),
      type: graphTypes[i % graphTypes.length], x: i * 200, y: -i, rotation: i % 2 ? 0.3 : 0 }));
    for (const editor of [a, b]) { seed(editor, existing); editor.clearHistory(); }
    const reads = [0, 0], originals = [a.getShapeUtil, b.getShapeUtil];
    [a, b].forEach((editor, i) => { editor.getShapeUtil = function (...args) { reads[i]++; return originals[i].apply(this, args); }; });
    const batch = Array.from({ length: size }, (_, i) => ({ id: sid(`batch-${i}`),
      type: graphTypes[i % graphTypes.length], x: -i * 0.5, y: i + 0.25 }));
    const frozen = freezeInputs(batch);
    a.run(() => a.createShapes(frozen), { history: 'ignore' });
    b.run(() => createGraphShapes(b, frozen), { history: 'ignore' });
    [a, b].forEach((editor, i) => { editor.getShapeUtil = originals[i]; });
    assert.equal(reads[0] - reads[1], size * size - size);
    compareEditors(a, b);
    counts.push({ existing: size, created: size, baselineUtilReads: reads[0], candidateUtilReads: reads[1] });
  });

  // Predicate and visibility callbacks run only in the original create path,
  // with exactly the same order, counts, errors, and final parent transforms.
  for (const mode of ['accept', 'reject', 'selective', 'stateful', 'throw']) withPair((a, b) => {
    const logs = [[], []];
    [a, b].forEach((editor, i) => {
      seed(editor, [{ id: sid('receiver'), type: 'module-node', x: -100, y: -100,
        props: { w: 1000, h: 1000 } }]);
      let calls = 0;
      editor.getShapeUtil('module-node').canReceiveNewChildrenOfType = (shape, type) => {
        logs[i].push([shape.id, type]); calls++;
        if (mode === 'throw') throw new Error('intentional receiver failure');
        return mode === 'accept' || (mode === 'selective' && type === 'layer') || (mode === 'stateful' && calls % 2 === 0);
      };
      editor.clearHistory();
    });
    const error = createPair(a, b);
    assert.equal(!!error, mode === 'throw');
    assert.deepEqual(logs[1], logs[0]);
    assert(logs[0].length > 0);
  });

  // Accessors are ambiguous overrides. In particular, the baseline never reads
  // a lower shape's getter when a higher receiver already accepts the new shape.
  for (const higherReceiver of [false, true]) withPair((a, b) => {
    const calls = [0, 0];
    [a, b].forEach((editor, i) => {
      seed(editor, [{ id: sid('getter-shape'), type: 'layer', x: -100, y: -100 }]);
      if (higherReceiver) seed(editor, [{ id: sid('higher-frame'), type: 'frame',
        x: -100, y: -100, props: { w: 1000, h: 1000 } }]);
      Object.defineProperty(editor.getShapeUtil('layer'), 'canReceiveNewChildrenOfType', {
        configurable: true, get() { calls[i]++; throw new Error('intentional receiver getter failure'); },
      });
    });
    const error = createPair(a, b, [{ id: sid('new-getter'), type: 'input-node', x: 20, y: 20 }]);
    assert.equal(!!error, !higherReceiver);
    assert.deepEqual(calls, higherReceiver ? [0, 0] : [1, 1]);
  });

  for (const variant of ['normal', 'hidden', 'locked', 'rotated', 'nested']) withPair((a, b) => {
    for (const editor of [a, b]) {
      seed(editor, [
        { id: sid('outer-frame'), type: 'frame', x: -500, y: -500, props: { w: 2000, h: 2000 } },
        { id: sid('frame'), type: 'frame', x: variant === 'nested' ? 400 : -100, y: variant === 'nested' ? 400 : -100,
          rotation: variant === 'rotated' ? 0.3 : 0, isLocked: variant === 'locked',
          ...(variant === 'nested' ? { parentId: sid('outer-frame') } : {}),
          meta: { hidden: variant === 'hidden' }, props: { w: 1000, h: 1000 } },
      ]);
      editor.clearHistory();
    }
    const visibility = [[], []];
    [a, b].forEach((editor, i) => {
      const hidden = editor.isShapeHidden;
      editor.isShapeHidden = function (shape) { visibility[i].push(shape.id ?? shape); return hidden.call(this, shape); };
    });
    createPair(a, b);
    assert.deepEqual(visibility[1], visibility[0]);
  }, { getShapeVisibility: shape => shape.meta.hidden ? 'hidden' : 'inherit' });

  for (const focused of [false, true]) withPair((a, b) => {
    for (const editor of [a, b]) {
      seed(editor, [
        { id: sid('outer-group'), type: 'group', x: 100, y: 200, rotation: 0.2 },
        { id: sid('group'), type: 'group', parentId: sid('outer-group'), x: 30, y: 40, rotation: 0.4 },
        { id: sid('member'), type: 'layer', parentId: sid('group'), x: 20, y: 30 },
        { id: sid('member2'), type: 'layer', parentId: sid('group'), x: 200, y: 30 },
        { id: sid('outer-member'), type: 'layer', parentId: sid('outer-group'), x: 500, y: 30 },
      ]);
      if (focused) editor.setFocusedGroup(sid('group'));
      editor.clearHistory();
    }
    createPair(a, b);
    assert.equal(b.getShape(sid('new-1')).parentId, focused ? sid('group') : b.getCurrentPageId());
  });

  // Explicit page, valid shape, invalid and same-batch parents all fall back as
  // one unchanged batch. Its original array reaches the create API.
  withPair((a, b) => {
    for (const editor of [a, b]) seed(editor, [{ id: sid('parent'), type: 'module-node', x: 80, y: 90 }]);
    const batch = [
      { id: sid('page-child'), type: 'layer', parentId: a.getCurrentPageId(), x: 1, y: 2 },
      { id: sid('shape-child'), type: 'layer', parentId: sid('parent'), x: 2, y: 3 },
      { id: sid('invalid-child'), type: 'layer', parentId: sid('missing'), x: 3, y: 4 },
      { id: sid('inbatch-child'), type: 'layer', parentId: sid('inbatch-parent'), x: 4, y: 5 },
      { id: sid('inbatch-parent'), type: 'module-node', x: 5, y: 6 },
    ];
    const create = b.createShapes; let received;
    b.createShapes = function (shapes) { received = shapes; return create.call(this, shapes); };
    createPair(a, b, batch);
    assert.equal(received, batch);
  });
  for (const inherited of [false, true]) withPair((a, b) => {
    const logs = [[], []];
    const input = index => {
      const fields = {};
      for (const [key, value] of Object.entries({ id: sid('accessor-input'), type: 'layer', x: 10, y: 20 })) {
        Object.defineProperty(fields, key, { enumerable: true, get() {
          logs[index].push(key);
          if (key === 'id') throw new Error('intentional input getter failure');
          return value;
        } });
      }
      return inherited ? Object.create(fields) : fields;
    };
    assert.throws(() => a.createShapes([input(0)]), /intentional input getter failure/);
    assert.throws(() => createGraphShapes(b, [input(1)]), /intentional input getter failure/);
    assert.deepEqual(logs[1], logs[0]); assert.deepEqual(logs[0], ['id']);
    compareEditors(a, b);
  });
  withPair((a, b) => {
    const batch = [{ id: sid('non-enumerable'), type: 'layer', x: 10, y: 20 }];
    Object.defineProperty(batch[0], 'x', { value: 10, enumerable: false });
    const original = b.createShapes; let received;
    b.createShapes = function (shapes) { received = shapes; return original.call(this, shapes); };
    createPair(a, b, batch); assert.equal(received, batch);
  });
  for (const variant of ['index-getter', 'symbol-getter', 'custom-map']) withPair((a, b) => {
    const logs = [[], []];
    const input = index => {
      const shape = { id: sid('array-input'), type: 'layer', x: 10, y: 20 };
      const batch = [shape];
      const throwing = label => () => { logs[index].push(label); throw new Error(`intentional ${label}`); };
      if (variant === 'index-getter') Object.defineProperty(batch, '0', { get: throwing(variant) });
      if (variant === 'symbol-getter') Object.defineProperty(shape, Symbol('custom'), { enumerable: true, get: throwing(variant) });
      if (variant === 'custom-map') batch.map = throwing(variant);
      return batch;
    };
    assert.throws(() => a.createShapes(input(0)), new RegExp(`intentional ${variant}`));
    assert.throws(() => createGraphShapes(b, input(1)), new RegExp(`intentional ${variant}`));
    assert.deepEqual(logs[1], logs[0]); assert.deepEqual(logs[0], [variant]);
    compareEditors(a, b);
  });
  withPair((a, b) => {
    const shape = Object.assign(Object.create(null), { id: sid('null-prototype'), type: 'layer', x: 1.5, y: 2.5 });
    createPair(a, b, [shape]);
  });
  for (const batch of [[], [{ id: sid('unpositioned'), type: 'layer' }], [null],
    [{ id: sid('bad-props'), type: 'layer', x: 0, y: 0, props: { w: 'invalid' } }]]) {
    withPair((a, b) => createPair(a, b, batch));
  }
  withPair((a, b) => {
    assert.throws(() => a.createShapes(null), /must provide an array/);
    assert.throws(() => createGraphShapes(b, null), /must provide an array/);
    b.getCurrentPageShapesSorted = () => { throw new Error('empty/readonly must not scan'); };
    createGraphShapes(b, []);
    for (const editor of [a, b]) editor.updateInstanceState({ isReadonly: true }, { history: 'ignore' });
    createGraphShapes(b, partials());
    assert.equal(b.getCurrentPageShapes().length, 0);
  });
  withPair((a, b) => {
    const events = [diagnostics(a), diagnostics(b)];
    b.getCurrentPageShapesSorted = () => { throw new Error('over-limit must not scan'); };
    a.createShapes(partials()); createGraphShapes(b, partials());
    assert.equal(a.getCurrentPageShapes().length, 0); assert.equal(b.getCurrentPageShapes().length, 0);
    assert.deepEqual(events[1], events[0]);
    assert.equal(events[1].maxShapes.length, 1);
  }, { options: { maxShapesPerPage: 2 } });

  // Check the store after deletes, including receivers made by a delete callback.
  for (const callbackCreatesReceiver of [false, true]) withPair((a, b) => {
    for (const editor of [a, b]) {
      seed(editor, [{ id: sid('retyped'), type: 'frame', x: -100, y: -100 }]);
      if (callbackCreatesReceiver) editor.sideEffects.registerAfterDeleteHandler('shape', () => {
        seed(editor, [{ id: sid('callback-frame'), type: 'frame', x: -100, y: -100, props: { w: 1000, h: 1000 } }]);
      });
      editor.clearHistory();
    }
    const create = (editor, candidate) => editor.run(() => {
      editor.deleteShapes([sid('retyped')]);
      const batch = [{ id: sid('retyped'), type: 'layer', x: 0.5, y: 0.25 }];
      if (candidate) createGraphShapes(editor, batch); else editor.createShapes(batch);
    }, { history: 'ignore' });
    create(a, false); create(b, true); compareEditors(a, b);
    assert.equal(b.getShape(sid('retyped')).parentId, callbackCreatesReceiver ? sid('callback-frame') : b.getCurrentPageId());
  });

  // A receiver introduced by a create callback is not in the current batch's
  // parent snapshot, but must be observed on the very next helper invocation.
  withPair((a, b) => {
    for (const editor of [a, b]) {
      let once = false;
      editor.on('created-shapes', () => {
        if (once) return;
        once = true;
        seed(editor, [{ id: sid('later-frame'), type: 'frame', x: -100, y: -100, props: { w: 1000, h: 1000 } }]);
      });
    }
    createPair(a, b, [{ id: sid('first'), type: 'layer', x: 0, y: 0 }]);
    assert.equal(b.getShape(sid('first')).parentId, b.getCurrentPageId());
    createPair(a, b, [{ id: sid('second'), type: 'layer', x: 20, y: 20 }]);
    assert.equal(b.getShape(sid('second')).parentId, sid('later-frame'));
  });

  // Proof is invocation-local: a changed util, a later base-prototype override,
  // or switching to a page with a receiver must all take the original path.
  withPair((a, b) => {
    for (const editor of [a, b]) seed(editor, [{ id: sid('mutable'), type: 'module-node',
      x: -100, y: -100, props: { w: 1000, h: 1000 } }]);
    createPair(a, b, [{ id: sid('before-change'), type: 'layer', x: 0, y: 0 }]);
    for (const editor of [a, b]) editor.getShapeUtil('module-node').canReceiveNewChildrenOfType = () => true;
    createPair(a, b, [{ id: sid('after-change'), type: 'layer', x: 20, y: 20 }]);
    assert.equal(b.getShape(sid('after-change')).parentId, sid('mutable'));
  });
  withPair((a, b) => {
    for (const editor of [a, b]) seed(editor, [{ id: sid('base-shape'), type: 'layer', x: 0, y: 0 }]);
    const original = tldraw.ShapeUtil.prototype.canReceiveNewChildrenOfType;
    let calls = 0;
    try {
      tldraw.ShapeUtil.prototype.canReceiveNewChildrenOfType = () => { calls++; return false; };
      createPair(a, b, [{ id: sid('after-prototype-change'), type: 'layer', x: 10, y: 10 }]);
      assert.equal(calls, 2, 'each native create invokes the replacement once');
    } finally { tldraw.ShapeUtil.prototype.canReceiveNewChildrenOfType = original; }
  });
  withPair((a, b) => {
    createPair(a, b, [{ id: sid('first-page'), type: 'layer', x: 0, y: 0 }]);
    for (const editor of [a, b]) {
      editor.createPage({ id: 'page:second', name: 'Second' }); editor.setCurrentPage('page:second');
      seed(editor, [{ id: sid('second-page-frame'), type: 'frame', x: -100, y: -100,
        props: { w: 1000, h: 1000 } }]);
    }
    createPair(a, b, [{ id: sid('second-page-child'), type: 'layer', x: 20, y: 20 }]);
    assert.equal(b.getShape(sid('second-page-child')).parentId, sid('second-page-frame'));
    for (const editor of [a, b]) editor.setCurrentPage('page:fixture');
    createPair(a, b, [{ id: sid('returned-to-first'), type: 'layer', x: 20, y: 20 }]);
    assert.equal(b.getShape(sid('returned-to-first')).parentId, 'page:fixture');
  });

  // The input array is not normalized or split; default hooks and metadata run
  // in the same order after all parents and indices have been resolved.
  withPair((a, b) => {
    const logs = [[], []];
    [a, b].forEach((editor, i) => {
      for (const type of graphTypes) editor.getShapeUtil(type).onBeforeCreate = shape => {
        logs[i].push([shape.id, shape.parentId, shape.index]);
        return { ...shape, meta: { ...shape.meta, hook: true } };
      };
    });
    createPair(a, b); assert.deepEqual(logs[1], logs[0]);
  });
  console.log(JSON.stringify({ passed: true, helper: 'real Editor/store and five production ShapeUtils', utilReads: counts, benchmark: false }));
}

async function bridgeChecks() {
  const useBaseline = loadBridge(true), useCandidate = loadBridge(false);
  for (const direction of ['horizontal', 'vertical']) {
    const cleanupStart = cleanups.length, editorStart = editors.length;
    const node = (id, type = 'moduleNode', extra = {}) => ({ id, type, position: { x: 100, y: 100 },
      data: { expanded: true, parameters: {}, layerType: { id: 'module', name: 'Module', parameters: [] } }, ...extra });
    const initial = [node('outer'), node('inner', 'moduleNode', { parentNode: 'outer' }),
      node('layer', 'layerNode', { parentNode: 'inner' }), node('hidden', 'moduleNode', { hidden: true }),
      node('input', 'inputNode', { position: { x: 50, y: 50 } }), node('output', 'outputNode', { position: { x: 500, y: 200 } })];
    const edge = { id: 'connection', source: 'input', target: 'output' };
    const seeded = [makeEditor(), makeEditor()];
    for (const editor of seeded) {
      seed(editor, [{ id: nodeIdToShapeId('outer'), type: 'layer', x: 10, y: 20 }]);
      editor.markHistoryStoppingPoint('seed');
      editor.updateShape({ id: nodeIdToShapeId('outer'), type: 'layer', x: 20 });
      editor.undo(); assert(editor.getCanUndo()); assert(editor.getCanRedo());
    }
    const history = seeded.map(editor => [editor.history.getNumUndos(), editor.history.getNumRedos()]);
    const a = await mountBridge(useBaseline, initial, [edge], seeded[0], direction);
    const b = await mountBridge(useCandidate, initial, [edge], seeded[1], direction);
    const compare = () => { compareEditors(a.editor, b.editor); assert.deepEqual(b.events, a.events); assert.deepEqual(b.backCalls, a.backCalls); };
    compare();
    assert.deepEqual(seeded.map(editor => [editor.history.getNumUndos(), editor.history.getNumRedos()]), history);
    assert.equal(b.editor.getIsReadonly(), true);
    assert.equal(b.editor.getShape(nodeIdToShapeId('outer')).type, 'module-node');
    assert(snapshot(b.editor).filter(shape => shape.type === 'model-edge').every(shape => shape.isLocked));
    const oldOrder = indices(b.editor);
    const moved = initial.map(n => n.id === 'outer' ? { ...n, position: { x: 100.5, y: 100.5 } } : n);
    await a.update(moved); await b.update(moved); compare();
    assert.equal(b.editor.getShape(nodeIdToShapeId('outer')).x, 100.5);
    assert.equal(b.editor.getShape(nodeIdToShapeId('layer')).x, 300.5);
    assert.deepEqual(indices(b.editor), oldOrder);
    // Collapse, repeat unchanged, expand, and delete/replace edges retain all
    // normal sync updates, callbacks and edge ordering calls.
    const collapsed = moved.map(n => ['inner', 'layer'].includes(n.id) ? { ...n, hidden: true } : n);
    await a.update(collapsed, []); await b.update(collapsed, []); compare();
    await a.update([...collapsed], []); await b.update([...collapsed], []); compare();
    await a.update(moved, [edge]); await b.update(moved, [edge]); compare();
    const retyped = moved.map(n => n.id === 'layer' ? { ...n, type: 'outputNode' } : n);
    await a.update(retyped); await b.update(retyped); compare();
    assert.equal(b.editor.getShape(nodeIdToShapeId('layer')).type, 'output-node');
    // Programmatic node updates still respect locks; edge updates retain their
    // existing lock override. The readonly wrapper is restored each time.
    await act(async () => {
      for (const { editor } of [a, b]) {
        editor.updateInstanceState({ isReadonly: false }, { history: 'ignore' });
        editor.run(() => editor.updateShape({ id: nodeIdToShapeId('outer'), type: 'module-node', isLocked: true }), { history: 'ignore' });
        editor.updateInstanceState({ isReadonly: true }, { history: 'ignore' });
      }
    });
    await act(async () => { await pause(80); });
    const lockedMove = retyped.map(n => n.id === 'outer' ? { ...n, position: { x: 110, y: 110 } } : n);
    await a.update(lockedMove); await b.update(lockedMove); compare();
    assert.equal(b.editor.getShape(nodeIdToShapeId('outer')).x, 100.5);
    assert.equal(b.editor.getIsReadonly(), true);
    await act(async () => { a.dispose(); b.dispose(); });
    const reusedA = await mountBridge(useBaseline, lockedMove, [edge], seeded[0], direction);
    const reusedB = await mountBridge(useCandidate, lockedMove, [edge], seeded[1], direction);
    compareEditors(reusedA.editor, reusedB.editor);
    assert.deepEqual(reusedB.events, reusedA.events);
    const freshA = await mountBridge(useBaseline, initial, [edge], undefined, direction);
    const freshB = await mountBridge(useCandidate, initial, [edge], undefined, direction);
    compareEditors(freshA.editor, freshB.editor); assert.deepEqual(freshB.events, freshA.events);
    console.log(JSON.stringify({ passed: true, direction, bridge: 'real hook, readonly projection, locks, history, retyping, collapse/expand, 0.5px updates, cleanup/remount',
      baseline: process.argv[2] ? 'unchanged source' : 'original two createShapes calls', benchmark: false }));
    await act(async () => { cleanups.splice(cleanupStart).forEach(dispose => dispose()); });
    editors.splice(editorStart).forEach(editor => editor.dispose());
  }
}
(async () => { helperChecks(); await bridgeChecks(); })().catch(error => {
  console.error(error); process.exitCode = 1;
}).finally(async () => {
  if (cleanups.length) await act(async () => { cleanups.forEach(dispose => dispose()); });
  editors.forEach(editor => editor.dispose());
  window.close();
});

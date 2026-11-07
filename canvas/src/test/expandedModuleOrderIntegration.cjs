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
const { createRoot } = process.argv.includes('--production-order-only') ? {} : require('react-dom/client');
const { act } = React;
const { planExpandedModuleOrder } = require('../canvas/expandedModuleOrder.ts');
const { nodeIdToShapeId } = require('../canvas/canvasShapeId.ts');
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const editors = [], cleanups = [];
const sid = tldraw.createShapeId;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

function makeEditor() {
  const container = document.createElement('div');
  container.getBoundingClientRect = () => ({ x: 0, y: 0, top: 0, left: 0, width: 1080, height: 720, right: 1080, bottom: 720 });
  document.body.appendChild(container);
  const editor = new tldraw.Editor({ shapeUtils, bindingUtils: tldraw.defaultBindingUtils,
    tools: [...tldraw.defaultTools, ...tldraw.defaultShapeTools],
    store: tldraw.createTLStore({ shapeUtils, bindingUtils: tldraw.defaultBindingUtils }),
    getContainer: () => container, initialState: 'select' });
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
const geometry = editor => snapshot(editor).map(({ index, ...rest }) => rest);
function compareEditors(a, b) {
  if (process.env.NODE_ENV === 'production') {
    assert.deepEqual(geometry(a), geometry(b));
    for (const parentId of new Set(snapshot(a).map(s => s.parentId))) {
      assert.deepEqual(a.getSortedChildIdsForParent(parentId), b.getSortedChildIdsForParent(parentId));
      for (const editor of [a, b]) {
        const siblings = editor.getSortedChildIdsForParent(parentId).map(id => editor.getShape(id));
        assert.equal(new Set(siblings.map(s => s.index)).size, siblings.length);
        siblings.forEach(s => assert.equal(tldraw.getIndicesBetween(s.index, undefined, 1).length, 1));
      }
    }
  } else assert.deepEqual(snapshot(a), snapshot(b));
  assert.equal(a.getCanUndo(), b.getCanUndo());
  assert.equal(a.getCanRedo(), b.getCanRedo());
  assert.equal(a.getIsReadonly(), b.getIsReadonly());
  assert.deepEqual(a.getSelectedShapeIds(), b.getSelectedShapeIds());
}
function diagnostics(editor) {
  const events = { edit: 0, edited: [], after: 0, complete: 0 };
  editor.on('edit', () => events.edit++);
  editor.on('edited-shapes', shapes => events.edited.push(shapes.map(s => s.id)));
  editor.sideEffects.registerAfterChangeHandler('shape', () => events.after++);
  editor.sideEffects.registerOperationCompleteHandler(() => events.complete++);
  return events;
}

// The normal CI invocation reconstructs only the pre-optimization ordering block
// in memory. Local validation can pass an unchanged baseline bridge source path.
// Neither route mocks the Editor, store, ShapeUtils, or projection wrapper.
function loadBridge(baseline) {
  const candidatePath = path.resolve(__dirname, '../canvas/useTldrawBridge.ts');
  const filename = baseline && process.argv[2] ? path.resolve(process.argv[2]) : candidatePath;
  let source = fs.readFileSync(filename, 'utf8');
  if (baseline && !process.argv[2]) {
    const current = `            const moduleIds = nodes.filter(node => node.type === 'moduleNode' && node.data.expanded && !node.hidden)
              .sort((a, b) => depth(b) - depth(a))
              .map(node => nodeIdToShapeId(node.id));
            const moduleOrder = planExpandedModuleOrder(editor, moduleIds);
            if (moduleOrder.length > 0) editor.updateShapes(moduleOrder);`;
    assert(source.includes(current), 'Update the explicit baseline block when the integration seam changes');
    source = source.replace(current, `            nodes.filter(node => node.type === 'moduleNode' && node.data.expanded && !node.hidden)
              .sort((a, b) => depth(b) - depth(a))
              .forEach(node => editor.sendToBack([nodeIdToShapeId(node.id)]));`);
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

async function setModuleLocks(fixtures, isLocked) {
  await act(async () => {
    for (const { editor } of fixtures) {
      editor.updateInstanceState({ isReadonly: false }, { history: 'ignore' });
      editor.run(() => editor.updateShapes(['outer', 'inner', 'sibling'].map(id => ({
        id: nodeIdToShapeId(id), type: 'module-node', isLocked,
      }))), { history: 'ignore', ignoreShapeLock: true });
      editor.updateInstanceState({ isReadonly: true }, { history: 'ignore' });
    }
  });
  await act(async () => { await pause(80); });
}

async function main() {
  // Real immutable-store parent indices, locks, repeated ids and equal index ties.
  for (const equalIndices of [false, true]) {
    const a = makeEditor(), b = makeEditor();
    for (const editor of [a, b]) {
      editor.createShapes([
        { id: sid('parent'), type: 'module-node', index: 'a0' },
        { id: sid('edge'), type: 'model-edge', index: 'a1' },
        { id: sid('outer'), type: 'module-node', index: 'a2', isLocked: true },
        { id: sid('inner'), type: 'module-node', index: equalIndices ? 'a2' : 'a3', isLocked: true },
        { id: sid('child-edge'), type: 'model-edge', index: 'a0', parentId: sid('parent') },
        { id: sid('child'), type: 'module-node', index: 'a1', parentId: sid('parent'), isLocked: true },
      ]);
      editor.clearHistory();
    }
    const before = geometry(a);
    const ids = ['inner', 'missing', 'child', 'outer', 'inner', 'inner', 'child'].map(sid);
    a.run(() => ids.forEach(id => a.sendToBack([id])), { history: 'ignore', ignoreShapeLock: true });
    b.run(() => b.updateShapes(planExpandedModuleOrder(b, ids)), { history: 'ignore', ignoreShapeLock: true });
    compareEditors(a, b);
    assert.deepEqual(geometry(a), before);
    assert.equal(a.getCanUndo(), false);
  }

  // The baseline emits an empty edit for a sole already-bottom module;
  // the candidate deliberately emits none, while final records remain equal.
  {
    const a = makeEditor(), b = makeEditor();
    for (const editor of [a, b]) {
      editor.createShape({ id: sid('alone'), type: 'module-node', index: 'a0' });
      editor.clearHistory();
    }
    const before = snapshot(a), aEvents = diagnostics(a), bEvents = diagnostics(b);
    a.run(() => a.sendToBack([sid('alone')]), { history: 'ignore', ignoreShapeLock: true });
    b.run(() => {
      const updates = planExpandedModuleOrder(b, [sid('alone')]);
      if (updates.length) b.updateShapes(updates);
    }, { history: 'ignore', ignoreShapeLock: true });
    compareEditors(a, b); assert.deepEqual(snapshot(a), before);
    assert.equal(aEvents.edit, 1); assert.equal(bEvents.edit, 0);
    assert.equal(aEvents.after, 0); assert.equal(bEvents.after, 0);
  }

  if (process.argv.includes('--production-order-only')) {
    console.log(JSON.stringify({ passed: true, mode: 'production jitter', checks: 'real-store sibling order, valid unique indices, geometry, history, locks, actual parents, repeated/missing ids and equal-index ties', benchmark: false }));
    return;
  }

  for (const direction of ['horizontal', 'vertical']) {
    const editorStart = editors.length, cleanupStart = cleanups.length;
    const node = (id, type = 'moduleNode', extra = {}) => ({ id, type, position: { x: 100, y: 100 },
      data: { expanded: true, parameters: {}, layerType: { id: 'module', name: 'Module', parameters: [] } }, ...extra });
    const initialNodes = [node('outer'), node('sibling'), node('inner', 'moduleNode', { parentNode: 'outer' }),
      node('collapsed', 'moduleNode', { data: { expanded: false } }), node('hidden', 'moduleNode', { hidden: true }),
      node('input', 'inputNode', { position: { x: 50, y: 50 } }), node('output', 'outputNode', { position: { x: 500, y: 200 } })];
    const edge = { id: 'connection', source: 'input', target: 'output' };
    const a = await mountBridge(loadBridge(true), initialNodes, [edge], undefined, direction);
    const b = await mountBridge(loadBridge(false), initialNodes, [edge], undefined, direction);
    compareEditors(a.editor, b.editor);
    assert.equal(b.editor.getIsReadonly(), true);
    assert.equal(b.editor.getCanUndo(), false);
    assert.equal(b.backCalls.length, 1, 'candidate keeps the edge call, batching module writes');
    assert.equal(b.editor.getShape(b.backCalls[0][0]).type, 'model-edge');
    assert.deepEqual(a.backCalls.slice(1), ['inner', 'outer', 'sibling'].map(id => [nodeIdToShapeId(id)]));
    const ordered = b.editor.getSortedChildIdsForParent(b.editor.getCurrentPageId());
    assert(ordered.indexOf(nodeIdToShapeId('sibling')) < ordered.indexOf(b.backCalls[0][0]));
    assert(b.events.edit < a.events.edit, 'batched edit notifications intentionally differ');
    const initialEventCounts = JSON.parse(JSON.stringify({ baseline: a.events, candidate: b.events }));
    assert(snapshot(b.editor).filter(s => s.type === 'model-edge').every(s => s.isLocked));

    // Ordinary user edits remain blocked after the real readonly wrapper restores.
    const shapeId = nodeIdToShapeId('outer'), originalX = b.editor.getShape(shapeId).x;
    b.editor.updateShape({ id: shapeId, type: 'module-node', x: 999 });
    assert.equal(b.editor.getShape(shapeId).x, originalX);
    // Geometry-only correction, including a subpixel state change, is not skipped.
    const moved = initialNodes.map(n => n.id === 'outer' ? { ...n, position: { x: 100.5, y: 100.5 } } : n);
    const oldOrder = indices(b.editor), oldBackCalls = b.backCalls.length;
    await a.update(moved); await b.update(moved);
    compareEditors(a.editor, b.editor);
    assert.equal(b.editor.getShape(shapeId).x, 100.5);
    assert.equal(b.editor.getShape(nodeIdToShapeId('inner')).x, 200.5);
    assert.deepEqual(indices(b.editor), oldOrder);
    assert.equal(b.backCalls.length, oldBackCalls);
    // Delete-only trigger and repeat are exercised through the full bridge.
    await a.update(null, []); await b.update(null, []); compareEditors(a.editor, b.editor);
    const repeated = indices(b.editor);
    await a.update([...moved], []); await b.update([...moved], []); compareEditors(a.editor, b.editor);
    assert.deepEqual(indices(b.editor), repeated);
    // Locked expanded modules must still reorder through the real projection wrapper.
    await setModuleLocks([a, b], true);
    // Restore edges to force another normal edge-first ordering operation.
    await a.update(null, [edge]); await b.update(null, [edge]); compareEditors(a.editor, b.editor);
    assert.equal(b.editor.getIsReadonly(), true);
    assert.equal(b.editor.getCanUndo(), false);

    for (const id of ['outer', 'inner', 'sibling']) assert.equal(b.editor.getShape(nodeIdToShapeId(id)).isLocked, true);
    // Unlock before later geometry tests: baseline node synchronization deliberately
    // respects module locks, independently of the ordering transaction's override.
    await setModuleLocks([a, b], false);

    // A real util hook failure is caught by the ordering boundary and readonly is
    // restored. Partial index progress and notification counts are not compared.
    const util = b.editor.getShapeUtil('module-node');
    const inherited = util.onBeforeUpdate;
    let throws = 0;
    util.onBeforeUpdate = (previous, next) => {
      if (previous.index !== next.index) { throws++; throw new Error('intentional ordering failure'); }
      return inherited?.call(util, previous, next);
    };
    await b.update(null, []);
    util.onBeforeUpdate = inherited;
    assert(throws > 0);
    assert.equal(b.editor.getIsReadonly(), true);
    const later = moved.map(n => n.id === 'outer' ? { ...n, position: { x: 104, y: 102 } } : n);
    await b.update(later, [edge]);
    assert.equal(b.editor.getShape(shapeId).x, 104);
    assert.equal(b.editor.getShape(nodeIdToShapeId('inner')).x, 204);
    assert.equal(b.editor.getCanUndo(), false);

    // Preserve pre-existing undo AND redo stacks, rather than only empty history.
    const seeded = [makeEditor(), makeEditor()];
    for (const editor of seeded) {
      editor.createShape({ id: shapeId, type: 'module-node' });
      editor.markHistoryStoppingPoint('seed');
      editor.updateShape({ id: shapeId, type: 'module-node', x: 20 });
      editor.undo();
      assert(editor.getCanUndo()); assert(editor.getCanRedo());
    }
    const historyBefore = seeded.map(editor => [editor.history.getNumUndos(), editor.history.getNumRedos()]);
    const seededA = await mountBridge(loadBridge(true), initialNodes, [edge], seeded[0], direction);
    const seededB = await mountBridge(loadBridge(false), initialNodes, [edge], seeded[1], direction);
    compareEditors(seededA.editor, seededB.editor);
    assert.deepEqual(seeded.map(editor => [editor.history.getNumUndos(), editor.history.getNumRedos()]), historyBefore);

    // StrictMode-like cleanup/remount on the SAME Editor, then a real user move.
    // A stale after-change subscription would carry nested children twice.
    await act(async () => { seededA.dispose(); seededB.dispose(); });
    const reusedA = await mountBridge(loadBridge(true), initialNodes, [edge], seeded[0], direction);
    const reusedB = await mountBridge(loadBridge(false), initialNodes, [edge], seeded[1], direction);
    compareEditors(reusedA.editor, reusedB.editor);
    const childX = reusedB.editor.getShape(nodeIdToShapeId('inner')).x;
    await act(async () => {
      for (const { editor } of [reusedA, reusedB]) {
        editor.updateInstanceState({ isReadonly: false }, { history: 'ignore' });
        editor.updateShape({ id: shapeId, type: 'module-node', x: 110 });
      }
    });
    await act(async () => { await pause(100); });
    compareEditors(reusedA.editor, reusedB.editor);
    assert.equal(reusedB.editor.getShape(nodeIdToShapeId('inner')).x, childX + 10);

    // Fresh mount repeats initial behavior; no retained planner/cache state.
    const remount = await mountBridge(loadBridge(false), initialNodes, [edge], undefined, direction);
    const reference = await mountBridge(loadBridge(true), initialNodes, [edge], undefined, direction);
    compareEditors(remount.editor, reference.editor);
    console.log(JSON.stringify({ passed: true, direction, integration: 'real Editor/store, five production ShapeUtils, real React bridge projection wrapper',
      baseline: process.argv[2] ? 'unchanged baseline source' : 'explicit pre-optimization ordering block',
      initialEventCounts,
      caveats: ['edit event count and exceptional partial progress intentionally differ', 'JSDOM state integration is not a browser visual check', 'no performance benchmark'] }));
    await act(async () => { cleanups.splice(cleanupStart).forEach(dispose => dispose()); });
    editors.splice(editorStart).forEach(editor => editor.dispose());
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; }).finally(async () => {
  if (cleanups.length) await act(async () => { cleanups.forEach(dispose => dispose()); });
  editors.forEach(editor => editor.dispose());
  window.close();
  // React's production scheduler creates a global MessageChannel on import.
  // It is not owned by any Editor and keeps a standalone JSDOM process alive.
  // All editors and React roots are already disposed; close only these harness
  // ports, rather than force process.exit() and hide pending test failures.
  if (process.argv.includes('--production-order-only')) {
    for (const handle of process._getActiveHandles()) {
      if (handle.constructor.name === 'MessagePort') handle.close();
    }
  }
});

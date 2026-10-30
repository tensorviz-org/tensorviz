// Real tldraw gestures in plain Node, for checks the CRA Jest transform cannot run:
// tldraw does not load under it, so tests run this file in a child process (see
// doubleClickTarget.test.ts). jsdom stands in for the browser, and tldraw's own
// Driver sends pointer events through the editor's click detector, select tool
// and hit test exactly as the canvas does.
//
// Usage: node tldrawGestures.cjs '{"shapes": [...], "gestures": [...]}'
//   shape:   { id, type: 'layer' | 'module-node', x, y, w, h }, in creation (z) order
//   gesture: { click: [x, y] } | { doubleClick: [x, y], shift?, onShape? }
//   onShape sends the pointer events with target 'shape' instead of 'canvas'.
// Prints { activations, clicks, selected } and exits. Evidence scripts require
// createEditor, loadTs and settle to drive other handlers.
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
const tldraw = require('tldraw');
const { Driver } = require('@tldraw/driver');

/** Compile one of the app's TypeScript modules for this process. */
function loadTs(file) {
  const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
    fileName: file,
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  }).outputText;
  const loaded = new Module(file, module);
  loaded.filename = file;
  loaded.paths = Module._nodeModulePaths(path.dirname(file));
  loaded._compile(compiled, file);
  return loaded.exports;
}

// Stand-ins for the card utils. Those render React and CSS that plain Node cannot
// load, and hit-test with BaseBoxShapeUtil's filled box, as these do.
const card = type => class extends tldraw.BaseBoxShapeUtil {
  static type = type;
  static props = { w: tldraw.T.number, h: tldraw.T.number };
  getDefaultProps() { return { w: 200, h: 100 }; }
  component() { return null; }
  getIndicatorPath() { return undefined; }
};

/** A select-tool editor on a 1080×720 viewport at zoom 1, holding these shapes. */
function createEditor(shapes) {
  const shapeUtils = [...tldraw.defaultShapeUtils, card('layer'), card('module-node')];
  const container = document.createElement('div');
  container.getBoundingClientRect = () => ({ x: 0, y: 0, top: 0, left: 0, width: 1080, height: 720, right: 1080, bottom: 720 });
  document.body.appendChild(container);
  const editor = new tldraw.Editor({
    shapeUtils,
    bindingUtils: tldraw.defaultBindingUtils,
    tools: [...tldraw.defaultTools, ...tldraw.defaultShapeTools],
    store: tldraw.createTLStore({ shapeUtils, bindingUtils: tldraw.defaultBindingUtils }),
    getContainer: () => container,
    initialState: 'select',
    options: { createTextOnCanvasDoubleClick: false },
  });
  editor.updateViewportScreenBounds(container);
  for (const { id, type, x, y, w, h } of shapes) {
    editor.createShape({ id: tldraw.createShapeId(id), type, x, y, props: { w, h } });
  }
  return { editor, driver: new Driver(editor) };
}

/** Wait until the click detector has reported the 'settle' phase. */
const settle = editor => new Promise(resolve => setTimeout(resolve, editor.options.multiClickDurationMs + 100));

async function perform({ shapes, gestures }) {
  const { doubleClickedShapeId } = loadTs(path.join(__dirname, '../canvas/doubleClickTarget.ts'));
  const { editor, driver } = createEditor(shapes);

  // Wired as SelectionWatcher wires it.
  const activations = [];
  const clicks = [];
  editor.on('event', info => {
    if (info.name === 'double_click') clicks.push({ phase: info.phase, target: info.target, shape: info.shape?.id });
    const shapeId = doubleClickedShapeId(editor, info);
    if (shapeId) activations.push(shapeId);
  });

  for (const gesture of gestures) {
    const [x, y] = gesture.click ?? gesture.doubleClick;
    const target = gesture.onShape ? tldraw.createShapeId(gesture.onShape) : undefined;
    driver.pointerMove(x, y);
    driver.click(x, y, target, { shiftKey: Boolean(gesture.shift) });
    if (gesture.doubleClick) driver.click(x, y, target, { shiftKey: Boolean(gesture.shift) });
  }
  await settle(editor);
  return { activations, clicks, selected: editor.getSelectedShapeIds() };
}

module.exports = { createEditor, loadTs, settle };

if (require.main === module) {
  perform(JSON.parse(process.argv[2])).then(
    // tldraw keeps background timers running; the result is complete.
    result => process.stdout.write(JSON.stringify(result), () => process.exit(0)),
    error => { console.error(error); process.exit(1); },
  );
}

import fs from 'fs';
import path from 'path';
import {
  isModelShape,
  isManagedCanvasShape,
  EDGE_SHAPE_TYPE,
} from '../../canvas/canvasShapePolicy';

/**
 * Regression tests for "Canvas correctness (tldraw bridge)" — PRs #41, #47, #58.
 *
 *   #41  template selection renders on canvas (tldraw mount-order race)
 *   #47  undo/redo buttons stuck disabled (H5) + new-graph work lost on reload (H7)
 *   #58  group-as-module now renders on the canvas (H6)
 *
 * The fixes live almost entirely inside useTldrawBridge.ts /
 * ModelBuilder.tsx, all of which import `tldraw` and therefore cannot load under the
 * CRA jest transform. What IS tldraw-free and directly exercised by the bridge's
 * React->store sync, edge-sync, and stray-shape cleanup paths:
 *
 *   - canvas/canvasShapePolicy    the shape classifier the sync effects branch on
 *
 * Everything that can only be observed through a live tldraw Editor (undo stack
 * state, mount-order race, on-canvas render of a grouped module) is specified as a
 * Playwright scenario in the QA summary "E2E SCENARIOS" note. The undo-pollution
 * defect that used to be parked here as an empty it.skip now runs for real, in
 * tests/e2e/qa-canvas-h6-h7.spec.ts. Nothing in this file is skipped.
 *
 * Live module layout and expandability are covered by canvasLayout.test.ts and
 * visibleLayout.test.tsx.
 */

describe('canvasShapePolicy — the classifier the bridge sync effects branch on', () => {
  // The React->store sync enumerates *existing* shapes with isModelShape
  // (useTldrawBridge.ts:217) to decide what to update vs delete. Every shape type
  // the adapter emits (input-node/output-node/module-node/layer) must be recognised;
  // if one were renamed, its live shape would fall out of the model set and the sync
  // would either fail to update it or treat it as a removed node — a silent desync.
  it('recognises all four node shape types the adapter emits as model shapes', () => {
    expect(isModelShape({ type: 'input-node' })).toBe(true);
    expect(isModelShape({ type: 'output-node' })).toBe(true);
    expect(isModelShape({ type: 'module-node' })).toBe(true);
    expect(isModelShape({ type: 'layer' })).toBe(true);
  });

  it('does NOT classify edge / chat / arbitrary shapes as model shapes', () => {
    // Edges and chat nodes are real canvas shapes but must be excluded from the
    // model-shape set, else the nodes-effect would count them as graph nodes and
    // never delete them through the edge path.
    expect(isModelShape({ type: EDGE_SHAPE_TYPE })).toBe(false);
    expect(isModelShape({ type: 'arrow' })).toBe(false);
    expect(isModelShape({ type: 'chat-node' })).toBe(false);
    expect(isModelShape({ type: 'geo' })).toBe(false);
  });

  it('treats edge shapes as MANAGED but NOT model — the crux of edge sync vs stray cleanup', () => {
    // The edges effect deletes exactly the `EDGE_SHAPE_TYPE || 'arrow'` shapes
    // before recreating them (useTldrawBridge.ts:330-343). Meanwhile the from-editor
    // handler deletes any shape that is NOT isManagedCanvasShape as a stray
    // (useTldrawBridge.ts:114-119). So edges must be (a) excluded from isModelShape
    // (so they are not mistaken for nodes) yet (b) included in isManagedCanvasShape
    // (so the first user interaction does not wipe every rendered edge as a stray).
    for (const edgeType of [EDGE_SHAPE_TYPE, 'arrow']) {
      expect(isModelShape({ type: edgeType })).toBe(false);
      expect(isManagedCanvasShape({ type: edgeType })).toBe(true);
    }
  });

  it('classifies every node shape type as managed too (never deleted as a stray)', () => {
    for (const t of ['input-node', 'output-node', 'module-node', 'layer']) {
      expect(isManagedCanvasShape({ type: t })).toBe(true);
    }
  });

  it('pins EDGE_SHAPE_TYPE to the exact string the ShapeUtil and edge effect share', () => {
    // Load-bearing constant: the edges effect creates/deletes shapes of this type
    // and ModelEdgeShapeUtil registers it. If it drifted, the delete pass would miss
    // the old edge shapes and duplicates would accumulate on every resync.
    expect(EDGE_SHAPE_TYPE).toBe('model-edge');
  });

  it('returns false (no throw) for a missing shape', () => {
    // The bridge calls isModelShape(editor?.getShape(id)) on possibly-missing shapes
    // (selection filter, useTldrawBridge.ts:411) and iterates possibly-stale shapes
    // during cleanup. A throw or a truthy result on null would break selection sync
    // or delete real shapes.
    expect(isModelShape(undefined)).toBe(false);
    expect(isModelShape(null)).toBe(false);
    expect(isManagedCanvasShape(undefined)).toBe(false);
    expect(isManagedCanvasShape(null)).toBe(false);
  });
});

// ---- REGRESSIONS FOR CONFIRMED BUGS ------------------------------------------

describe('canvas-correctness regressions', () => {
  // FIXED. This was an it.skip whose body was a comment and nothing else, so
  // un-skipping it would only have produced a test that can never fail. The
  // defect it described was real:
  //
  //   Programmatic React->tldraw sync recorded onto tldraw's undo stack, so undo
  //   was enabled with no user edit, and pressing it erased shapes that merely
  //   mirrored unchanged React state — with no re-sync path, because the sync
  //   effects only re-run on nodes/edges changes and an undo does not change
  //   React state.
  //
  // Every programmatic batch in useTldrawBridge.ts now runs history:'ignore', and
  // ChatShapeUtil's legacy-anchor size migration — which runs on LOAD, so it lit
  // the affordance on a graph nobody had touched — now does too. The one
  // deliberate exception is the module-drag child carry, which is a user gesture
  // and must undo as one step.
  //
  // The observable behaviour needs a live Editor and an undo stack, so it is
  // asserted in the browser by tests/e2e/qa-canvas-h6-h7.spec.ts. What is
  // checkable here is that no programmatic mutation site loses the flag again:
  // a whole-file scan, not a character-offset window, so it survives a reflow.
  const CANVAS_DIR = path.resolve(__dirname, '../..', 'canvas');
  const readCanvas = (file: string) => fs.readFileSync(path.join(CANVAS_DIR, file), 'utf8');

  /**
   * The options argument of every `editor.run(...)` batch in a file.
   *
   * Scans to the matching close paren rather than to the next `;` — a batch body
   * is full of semicolons — so the answer does not depend on how the call is
   * formatted, only on where it ends.
   */
  const runBatches = (source: string): Array<{ start: number; end: number; options: string }> => {
    const batches: Array<{ start: number; end: number; options: string }> = [];
    let from = source.indexOf('editor.run(');
    while (from !== -1) {
      let parens = 0;
      let braces = 0;
      let lastArgAt = -1;
      let index = from + 'editor.run'.length;
      for (; index < source.length; index += 1) {
        const ch = source[index];
        if (ch === '(') parens += 1;
        else if (ch === ')') {
          parens -= 1;
          if (parens === 0) break;
        } else if (ch === '{') braces += 1;
        else if (ch === '}') braces -= 1;
        // A comma between the callback and the options object, not one inside
        // either of them.
        else if (ch === ',' && parens === 1 && braces === 0) lastArgAt = index;
      }
      batches.push({
        start: from,
        end: index,
        options: lastArgAt === -1 ? '' : source.slice(lastArgAt, index),
      });
      from = source.indexOf('editor.run(', index);
    }
    return batches;
  };

  const MUTATORS = /editor\.(createShapes?|updateShapes?|deleteShapes)\(/g;

  it('runs every programmatic editor mutation outside the undo stack', () => {
    for (const file of ['useTldrawBridge.ts']) {
      const source = readCanvas(file);
      const batches = runBatches(source);
      // Every batch declares its history intent…
      for (const batch of batches) {
        expect(batch.options).toMatch(/history: 'ignore'|ignoreShapeLock: true/);
      }
      // …and no mutation escapes a batch. ChatShapeUtil's legacy-anchor size
      // migration used to be a bare editor.updateShape() in a load-time effect,
      // which is a recorded edit the user never made.
      MUTATORS.lastIndex = 0;
      let call = MUTATORS.exec(source);
      let seen = 0;
      while (call !== null) {
        seen += 1;
        const inside = batches.some((b) => call!.index > b.start && call!.index < b.end);
        expect(`${file}:${call[0]} batched=${inside}`).toBe(`${file}:${call[0]} batched=true`);
        call = MUTATORS.exec(source);
      }
      expect(seen).toBeGreaterThan(0);
    }
  });

  it('keeps exactly one deliberate exception, and it is the user drag', () => {
    // The module-drag child carry is left ON the stack on purpose: undoing a
    // module drag must return the frame AND its children as one step. If a second
    // exception appears, it is far more likely to be an oversight than a decision.
    const bridge = readCanvas('useTldrawBridge.ts');
    const recorded = runBatches(bridge).filter((b) => !b.options.includes("history: 'ignore'"));
    expect(recorded).toHaveLength(1);
    expect(bridge).toContain("Left on the undo stack (NOT history:'ignore') so undoing a module drag");
  });

});

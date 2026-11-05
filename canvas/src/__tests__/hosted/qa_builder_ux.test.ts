/**
 * QA regression suite — area: "Builder chrome, layout & shortcuts".
 * Targets the last-2-days PRs: #20 (minimal builder chrome — kill left rail,
 * 40px top bar, logo-mark nav), #21/#22/#23 (P2b–P2d floating chrome, canvas
 * gate, keyboard shortcuts) and #37 (visual-system tightening / single-accent).
 *
 * This area is almost entirely UI/CSS + tldraw, which the CRA jest transform
 * cannot mount (importing ChatShapeUtil/tldraw breaks jest). So this file only
 * imports the one PROVEN tldraw-free pure module in the area
 * (selectPaletteTemplateItems), and pins every other behavior with the same
 * source-contract technique the repo already uses in ui2InteractionAudit.test.ts
 * — reading the shipped source and asserting the invariant, so a regression in
 * the shortcut map or the chrome layout FAILS the build. Full interactive
 * coverage lives in the Playwright specs (see "E2E SCENARIOS" in the task notes).
 *
 * Part 4 collects the bugs this suite confirmed. They were pinned with it.skip
 * and a "// BUG:" note while they were open; every one of them has since been
 * fixed and un-skipped, and each keeps a "// FIXED:" note naming the defect it
 * closes. Nothing in this file is skipped.
 */
import fs from 'fs';
import path from 'path';
import ts from 'typescript';
import { selectPaletteTemplateItems } from '../../components/ui2/BuilderPaletteV2';
import {
  PaperTemplateInventoryResponse,
  PaperTemplateInventoryTemplate,
} from '../../services/apiService';

const UI2_DIR = path.resolve(__dirname, '../..', 'components', 'ui2');
const FRONTEND_ROOT = path.resolve(__dirname, '../..', '..');

const BUILDER_PAGE = path.join(UI2_DIR, 'BuilderPageV2.tsx');
const WORKBENCH_SHELL = path.join(UI2_DIR, 'WorkbenchShell.tsx');
const UI2_TOKENS = path.join(FRONTEND_ROOT, 'src', 'styles', 'ui2.tokens.css');
const CANVAS_DIR = path.resolve(__dirname, '../..', 'canvas');
const TLDRAW_BRIDGE = path.join(CANVAS_DIR, 'useTldrawBridge.ts');
const TLDRAW_CANVAS = path.join(CANVAS_DIR, 'TldrawCanvas.tsx');
const TLDRAW_THEME = path.join(CANVAS_DIR, 'tldraw-theme.css');

const read = (p: string) => fs.readFileSync(p, 'utf8');

// ---------------------------------------------------------------------------
// Part 1 — Layer palette fuzzy search (pure logic; drives the Shift+A drawer).
// The existing builderPaletteTemplateSearch.test.ts already covers (a) searching
// the full inventory before the visible limit and (b) the unfiltered 28-item cap.
// These target the UNCOVERED behaviors: sort order, case-insensitivity,
// id-vs-name matching, whitespace-only queries, and null inventory.
// ---------------------------------------------------------------------------

function template(
  template_id: string,
  name: string,
  source: PaperTemplateInventoryTemplate['source']
): PaperTemplateInventoryTemplate {
  return {
    template_id,
    name,
    source,
    required_nodes: 1,
    required_edges: 0,
    training_minimums: 0,
    input_expectation: 'rank-3 tensor [batch, sequence, features]',
  };
}

function inventory(templates: PaperTemplateInventoryTemplate[]): PaperTemplateInventoryResponse {
  return {
    summary: {
      template_total: templates.length,
      template_builtin: templates.filter((t) => t.source === 'builtin').length,
      template_dynamic: templates.filter((t) => t.source === 'dynamic').length,
      mapping_total: 0,
      mapping_dynamic: 0,
      staging_pending: 0,
      staging_approved: 0,
      staging_rejected: 0,
    },
    templates,
    dynamic_template_ids: [],
    dynamic_mappings: {},
  };
}

describe('selectPaletteTemplateItems — palette search ordering & matching', () => {
  const mixed = inventory([
    template('z_dyn', 'Zebra Block', 'dynamic'),
    template('a_built', 'Alpha Block', 'builtin'),
    template('m_dyn', 'Mango Block', 'dynamic'),
  ]);

  it('sorts dynamic templates ahead of builtin ones', () => {
    const items = selectPaletteTemplateItems(mixed, '');
    // dynamic entries first (Mango, Zebra by name), then builtin (Alpha) — a
    // regression that dropped the source tiebreak would put Alpha first.
    expect(items.map((t) => t.template_id)).toEqual(['m_dyn', 'z_dyn', 'a_built']);
    expect(items[items.length - 1].source).toBe('builtin');
  });

  it('sorts alphabetically by name within the same source', () => {
    const dynamicOnly = inventory([
      template('t3', 'Transformer', 'dynamic'),
      template('t1', 'Attention', 'dynamic'),
      template('t2', 'Bottleneck', 'dynamic'),
    ]);
    expect(selectPaletteTemplateItems(dynamicOnly, '').map((t) => t.name)).toEqual([
      'Attention',
      'Bottleneck',
      'Transformer',
    ]);
  });

  it('matches case-insensitively', () => {
    const items = selectPaletteTemplateItems(mixed, 'ZEB');
    expect(items).toHaveLength(1);
    expect(items[0].template_id).toBe('z_dyn');
  });

  it('matches on template_id substring, not just display name', () => {
    // Query "a_built" is present only in the id; the name is "Alpha Block".
    const items = selectPaletteTemplateItems(mixed, 'a_built');
    expect(items.map((t) => t.template_id)).toEqual(['a_built']);
  });

  it('treats a whitespace-only query as unfiltered (query is trimmed)', () => {
    expect(selectPaletteTemplateItems(mixed, '   ')).toHaveLength(3);
  });

  it('returns an empty list for a null inventory instead of throwing', () => {
    expect(selectPaletteTemplateItems(null, 'anything')).toEqual([]);
  });

  it('returns an empty list when nothing matches the query', () => {
    expect(selectPaletteTemplateItems(mixed, 'no-such-template')).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Part 2 — Builder keyboard shortcut map (PRs #21–#23, P2b–P2d).
// Only Cmd/Ctrl+P and Cmd/Ctrl+\ are covered by builder-shortcuts.spec.ts.
// Cmd+Enter (verify), Cmd+L (layout), Cmd+0 (fit-view), Shift+A (palette) and
// "i" (inspector toggle) are unguarded. This pins the whole mapping plus the
// "prevent input shortcut conflicts" fix (bail while typing) at the source
// level, since the handler lives inside a tldraw-mounting component.
// ---------------------------------------------------------------------------

function keydownHandlerSource(): string {
  const src = read(BUILDER_PAGE);
  const start = src.indexOf('const onKeyDown = (event: KeyboardEvent) =>');
  const end = src.indexOf("window.addEventListener('keydown', onKeyDown)");
  expect(start).toBeGreaterThan(-1);
  expect(end).toBeGreaterThan(start);
  return src.slice(start, end);
}

/** True iff `action` appears within 120 chars after `keyCondition` (same branch). */
function branchMaps(handler: string, keyCondition: string, action: string): boolean {
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(esc(keyCondition) + '[\\s\\S]{0,120}?' + esc(action)).test(handler);
}

describe('builder keyboard shortcut command map', () => {
  const handler = keydownHandlerSource();

  it('ignores shortcuts while the user is typing (prevents input conflicts)', () => {
    // Fix from "prevent input shortcut conflicts": typing in an input/textarea/
    // contentEditable must not fire canvas shortcuts.
    expect(handler).toContain(
      "const isTyping = tag === 'input' || tag === 'textarea' || target?.isContentEditable;"
    );
    expect(handler).toContain('if (isTyping) return;');
  });

  it('maps Cmd/Ctrl+Enter to verify', () => {
    expect(branchMaps(handler, "event.key.toLowerCase() === 'enter'", "dispatchBuilderCommand('verify')")).toBe(true);
  });

  it('maps Cmd/Ctrl+L to auto-layout', () => {
    expect(branchMaps(handler, "event.key.toLowerCase() === 'l'", "dispatchBuilderCommand('layout')")).toBe(true);
  });

  it('maps Cmd/Ctrl+0 to fit-view', () => {
    expect(branchMaps(handler, "event.key.toLowerCase() === '0'", "dispatchBuilderCommand('fit-view')")).toBe(true);
  });

  it('maps Cmd/Ctrl+P to the quick-jump palette', () => {
    expect(branchMaps(handler, "event.key.toLowerCase() === 'p'", 'setQuickJumpOpen(true)')).toBe(true);
  });

  it.each(['metaKey', 'ctrlKey'])('preserves modified host shortcuts while keeping %s+P for graph search', (modifier) => {
    const openSearch = jest.fn();
    // Execute the shipped listener, including its modifier checks, without
    // loading tldraw into CRA's Jest transform. Native VS Code verifies routing.
    const onKeyDown = new Function('readOnlyLocalBrowser', 'dispatchBuilderCommand',
      'setChromeHidden', 'setQuickJumpOpen', 'setPaletteOpen',
      `${ts.transpile(handler)}; return onKeyDown;`)(true, jest.fn(), jest.fn(), openSearch, jest.fn());
    for (const extra of [{ shiftKey: true }, { altKey: true }]) {
      const event = new KeyboardEvent('keydown', { key: 'P', [modifier]: true, ...extra, cancelable: true });
      onKeyDown(event);
      expect(event.defaultPrevented).toBe(false);
      expect(openSearch).not.toHaveBeenCalled();
    }
    const plain = new KeyboardEvent('keydown', { key: 'p', [modifier]: true, cancelable: true });
    onKeyDown(plain);
    expect(plain.defaultPrevented).toBe(true);
    expect(openSearch).toHaveBeenCalledWith(true);
  });

  it('maps Shift+A (no meta/ctrl/alt) to opening the layer palette', () => {
    expect(
      branchMaps(handler, "event.shiftKey && event.key.toLowerCase() === 'a'", 'setPaletteOpen(true)')
    ).toBe(true);
  });

  // The Inspector is gone — parameters are edited ON the node — so the two
  // shortcuts that drove it ("i" to toggle the panel, "e" to open its Properties
  // tab for the selection) have nothing to open. Neither may come back as a
  // silent no-op: a key that appears to do nothing is worse than no key.
  it('has no shortcut left that reaches for the removed inspector', () => {
    expect(handler).not.toContain('setInspectorState');
    expect(handler).not.toContain('setInspectorTab');
    // The selection count was read only by the "e" branch; a dep left behind
    // would re-bind the whole keydown listener on every selection change.
    expect(read(BUILDER_PAGE)).toContain('}, [readOnlyLocalBrowser]);');
  });

  it('wires Cmd/Ctrl+\\ to hide/restore all floating chrome', () => {
    // The exact key literal is covered by builder-shortcuts.spec.ts; here we pin
    // that the hide-all-chrome toggle is driven from the keyboard handler.
    expect(handler).toContain('setChromeHidden((v) => !v)');
  });
});

// ---------------------------------------------------------------------------
// Part 3 — Minimal builder chrome (PR #20).
// The builder route must drop the persistent left rail, use a 40px (h-10) top
// bar, and move nav behind a logo-mark dropdown menu, so the canvas goes
// full-bleed. These invariants have no unit surface (WorkbenchShell mounts
// UserMenu + router), so they are pinned against the shipped source.
// ---------------------------------------------------------------------------

describe('minimal builder chrome layout', () => {
  const shell = read(WORKBENCH_SHELL);
  const builder = read(BUILDER_PAGE);

  // 'ide' joined the switch when the IDE stopped mounting the hosted product's
  // routes: it is 'minimal' plus the removal of the nav menu and the account
  // menu, so it must keep the minimal LAYOUT — hence the `|| chrome === 'ide'`
  // pinned here rather than a looser "contains minimal" check. 'local' joined
  // for the external read-only page: same minimal layout, no top bar at all,
  // because that page's own strip is the single truth header.
  it('WorkbenchShell exposes a full|minimal|ide|local chrome switch defaulting to full', () => {
    expect(shell).toContain("chrome?: 'full' | 'minimal' | 'ide' | 'local';");
    expect(shell).toContain("chrome = 'full',");
    expect(shell).toContain("const minimal = chrome === 'minimal' || chrome === 'ide' || chrome === 'local';");
    expect(shell).toContain("const ideChrome = chrome === 'ide';");
  });

  it('drops the left nav rail in minimal mode', () => {
    // The <aside className="ui2-sidebar"> rail is rendered only when NOT minimal.
    expect(/\{!minimal && \(\s*<aside className="ui2-sidebar/.test(shell)).toBe(true);
  });

  it('uses a 40px (h-10) top bar with a logo-mark dropdown nav menu in minimal mode', () => {
    expect(shell).toContain('ui2-topbar h-10 px-4');
    expect(shell).toContain('aria-haspopup="menu"');
    expect(shell).toContain('role="menu"');
  });

  it('the builder page opts into minimal chrome, ide chrome in the IDE, and local chrome on the read-only page', () => {
    // The local page must win over inIde: inside this bundle BOTH flags are
    // true there, so the ternary's order is the contract.
    expect(builder).toContain(`chrome={readOnlyLocalBrowser ? 'local' : inIde ? 'ide' : 'minimal'}`);
  });
});

// ---------------------------------------------------------------------------
// Part 4 — Regression tests for bugs that WERE confirmed here and have since
// been fixed. Each keeps its "// FIXED:" note so the shipped behaviour is
// traceable to the defect it closes. Nothing in this file is skipped.
// ---------------------------------------------------------------------------

describe('builder-chrome regressions', () => {
  // FIXED: this used to be a skipped pin asserting that ui-v2-visual.spec.ts had
  // stopped naming removed chrome. That spec was itself `test.describe.skip`ped
  // and appeared in no npm spec list, so the pin was a switched-off test policing
  // a switched-off spec: two layers of off, guarding nothing at all. Retargeted
  // at the gate that actually runs. (The spec has since been deleted outright —
  // it never had a visual baseline to restore — so nothing is left to point at.)
  //
  // The smoke set is the spec list in package.json, not a file this test picks.
  // That is the part that rotted last time — the guard went on pointing at a
  // spec after the gate moved — so the list is READ, and a name in it that no
  // longer exists fails here rather than silently guarding nothing.
  it('the e2e smoke gate exists and does not assert chrome the builder no longer renders', () => {
    const scripts = JSON.parse(read(path.join(FRONTEND_ROOT, 'package.json'))).scripts;

    // Follow `npm run <script>` indirection before looking for spec names. The
    // smoke script delegates to its :auto twin so that a single definition routes
    // through run-e2e.mjs — no npm script may call `playwright test` directly,
    // because Playwright exits 0 when everything it collected was skipped. Reading
    // only the top-level string would find zero specs and fail for the wrong
    // reason; resolving the alias asks the same question of the same specs.
    const resolve = (name: string, depth = 0): string => {
      const body = String(scripts[name] || '');
      const alias = body.match(/^npm run ([\w:-]+)$/);
      if (alias && depth < 5) return resolve(alias[1], depth + 1);
      return body;
    };
    const smokeScript = resolve('test:e2e:ui2:smoke');
    const smokeSpecs = smokeScript
      .split(/\s+/)
      .filter((token: string) => token.endsWith('.spec.ts'));
    expect(smokeSpecs.length).toBeGreaterThan(0);

    // The guard itself: exiting on `executed === 0` is what makes a green smoke
    // run mean something. A script that shells straight to `playwright test` is
    // how the deleted ui2-gate reported success having run nothing.
    expect(smokeScript).toContain('run-e2e.mjs');

    // "Inspector" as a button on load, a "Shortcuts:" legend and a "Flow
    // Overview" panel were all removed by the minimal-chrome rewrite (#20-#23).
    const removedChrome = [/Flow Overview/, /Shortcuts:/, /name: 'Inspector', exact: true/];
    for (const spec of smokeSpecs) {
      const file = path.join(FRONTEND_ROOT, spec);
      expect(fs.existsSync(file)).toBe(true);
      const source = read(file);
      // A gate that is skipped guards nothing; that is how the last one rotted.
      expect(source).not.toMatch(/test\.describe\.skip\(/);
      for (const pattern of removedChrome) expect(source).not.toMatch(pattern);
    }
  });

  // FIXED: Graph<->Code view toggle used to unmount tldraw (ModelBuilder was
  // mounted only in the graph branch of a "viewTab === 'graph' ? (…) : (…)"
  // ternary), so a Code View peek destroyed undo history, camera/zoom, and
  // selection. The persistenceKey fix direction lives in TldrawCanvas (out of
  // the builder-ux file scope); the shipped fix instead keeps <ModelBuilder>
  // mounted and hides the graph container via CSS, so the canvas is never
  // unmounted and its editor state survives the toggle.
  it('keeps the tldraw canvas mounted across the Graph<->Code toggle (hidden, not unmounted)', () => {
    const src = read(BUILDER_PAGE);
    // Canvas is no longer swapped out for the code panel by the view ternary…
    expect(src.includes("viewTab === 'graph' ? (")).toBe(false);
    // …instead the graph container's visibility is driven by viewTab while the
    // <ModelBuilder> mount stays put, and the code panel is a separate sibling.
    expect(src).toContain("display: viewTab === 'graph' ? undefined : 'none'");
    expect(src).toContain("{viewTab === 'code' && (");
  });

  // FIXED: the top-level "Code View" tab showed only the SELECTED layer's
  // snippet (codePreview is derived from snapshot.selectedNode), not the
  // full-model PyTorch — full-model code is produced by the "Export Code"
  // backend flow, not this preview. Rather than build a separate full-model
  // generator in the frontend (risking incorrect code for branching graphs),
  // the tab is relabeled to honestly reflect its per-layer scope.
  it('the code-view tab honestly labels its selected-layer snippet, not a full-model "Code View"', () => {
    const src = read(BUILDER_PAGE);
    const btnIdx = src.indexOf('data-testid="ui2-view-code"');
    expect(btnIdx).toBeGreaterThan(-1);
    // The label is now mode-dependent, so it is read from the one place that decides it
    // rather than from a character window after the testid — a window that broke the moment
    // the literal moved into a variable, without anything about the claim changing.
    expect(src).toContain("const codeTabLabel = inIde ? 'Reveal in editor' : 'Layer Code';");
    // The rendered label must use the mode-specific value.
    expect(src).toContain('<FileCode2 size={14} /> <span className="ui2-tool-label">{codeTabLabel}</span>');
    // The claim being pinned, in both modes: neither label calls this a whole-model view.
    // Scoped to what is RENDERED — the phrase is legitimate in a comment explaining why the
    // tab is not called that, and a file-wide ban would fail on its own rationale.
    expect(src).not.toMatch(/(?:>|})\s*Code View/);
  });

  // In the IDE the real file is a pane away, so the button goes there instead of rendering a
  // snippet the frontend synthesised — the surface the First-Run audit found inventing
  // PyTorch that the backend registry would never emit. Outside the IDE there is no editor to
  // reveal, so the snippet tab stays exactly as it was.
  it('inside the IDE the code button reveals the real line instead of a synthesised snippet', () => {
    const src = read(BUILDER_PAGE);
    const start = src.indexOf('const showSelectedLayerCode = useCallback(');
    const end = src.indexOf('}, [inIde, selectedNodeId]);', start);
    expect(start).toBeGreaterThan(-1);
    expect(end).toBeGreaterThan(start);
    const handler = src.slice(start, end);
    expect(handler).toContain('reportNodeClick(selectedNodeId)');
    expect(handler).toContain("setViewTab('code')");
    // Nothing to reveal without a selection, so the button says so by being disabled
    // rather than by doing nothing when pressed.
    expect(src).toContain('disabled={inIde && !selectedNodeId}');
  });

  // FIXED: the nav (full sidebar rail + minimal-chrome dropdown menu) used to
  // highlight the active route with the pre-migration green rgba(87,242,135,...)
  // instead of the cyan single-accent token (--ui2-cyan / --ui2-accent). The
  // highlight is now tinted from var(--ui2-accent).
  it('active nav highlight uses the cyan accent token, not the old green', () => {
    const shell = read(WORKBENCH_SHELL);
    expect(shell.includes('rgba(87,242,135')).toBe(false);
    expect(shell).toContain('color-mix(in srgb, var(--ui2-accent) 10%, transparent)');
  });

  // FIXED: dragging a node left its edges behind. The during-drag recompute in
  // useTldrawBridge was correct but a silent no-op: edge shapes are created with
  // isLocked: true, and tldraw's updateShapes DROPS partials for locked shapes
  // unless the batch opts out of the lock.
  it('repositions edges during a drag without being blocked by the edge shape lock', () => {
    const bridge = read(TLDRAW_BRIDGE);
    const idx = bridge.indexOf('editor.run(() => editor.updateShapes(shapesToUpdate)');
    expect(idx).toBeGreaterThan(-1);
    const call = bridge.slice(idx, idx + 160);
    expect(call).toContain('ignoreShapeLock: true');
    // …and still off the undo stack, so a drag stays one undo step.
    expect(call).toContain("history: 'ignore'");
  });

  // FIXED: the inspector opened on the pointer-DOWN that begins a drag, because
  // tldraw's SelectTool selects before it knows the gesture is a drag. A click is
  // now a pointer_up still in 'select.pointing_shape' (PointingShape leaves for
  // 'translating' the moment the drag threshold is crossed).
  it('treats only a drag-free pointer_up as a shape click', () => {
    const canvas = read(TLDRAW_CANVAS);
    expect(canvas).toContain("info.name !== 'pointer_up'");
    expect(canvas).toContain("editor.getPath() !== 'select.pointing_shape'");
    // The old selection-derived detector must be gone.
    expect(canvas.includes("registerAfterChangeHandler(\n      'instance_page_state'")).toBe(false);
  });

  // The panel opened on the pointer-DOWN that begins a drag, because tldraw's
  // SelectTool selects before it can know the gesture is a drag — so moving a card
  // put a panel over the canvas you were moving it on. Parameters now open from
  // the same drag-free click the shape-click channel already distinguishes, and
  // from nothing else.
  it('opens the parameter editor from a real node click, not from selection', () => {
    const canvas = read(TLDRAW_CANVAS);
    const start = canvas.indexOf("if (info.name !== 'pointer_up' || editor.getPath() !== 'select.pointing_shape') return;");
    expect(start).toBeGreaterThan(-1);
    const handler = canvas.slice(start, start + 600);
    expect(handler).toContain('openNodeParameters(selected[0])');
    // Read-only pages do not edit source, and a non-model shape (the chat card)
    // has no parameters to show.
    expect(handler).toContain('canEditParameters && isModelShape(');
    expect(canvas).toContain('canEditParameters={!readOnly || !!local?.sourceEdits}');
    // It belongs to one card: any other selection takes it away.
    expect(canvas).toContain(
      "if (target && (selectedIds.length !== 1 || selectedIds[0] !== target)) closeNodeParameters();"
    );
  });

  // The expanded Inspector was the one piece of chrome that could not float: it
  // was opaque and 320px wide, so the canvas ran underneath it and the cards in
  // that strip could not be clicked (qa.md CV-11). Reserving the width was the
  // fix. With the panel gone the builder reserves nothing, and the popover that
  // replaced it floats — so that whole class of dead strip cannot return.
  it('reserves no canvas edge on the builder, because nothing is docked any more', () => {
    const builder = read(BUILDER_PAGE);
    const styles = read(UI2_TOKENS);
    const canvasCss = read(TLDRAW_THEME);

    expect(builder).not.toContain('ui2-inspector-open');
    expect(styles).not.toContain('--ui2-canvas-inset-left: 352px');
    expect(styles).not.toContain('--ui2-canvas-inset-right');
    // The variable itself stays: the project rail is docked and still sets it.
    expect(styles).toContain('--ui2-canvas-inset-left: var(--ui2-project-rail-width, 248px);');
    expect(canvasCss).toContain('.tldraw-canvas-wrapper .model-node-params {');
    expect(canvasCss).toContain('pointer-events: auto;');
  });

  it('gates the node-click channel to model shapes so the chat card cannot open it', () => {
    expect(read(TLDRAW_BRIDGE)).toContain(
      "if (isModelShape(editorRef.current?.getShape(shapeId))) onNodeClick?.();"
    );
  });

  // FIXED: tldraw's own Select/Hand toolbar rendered bottom-centre on top of our
  // builder toolbar, duplicating our Select button.
  it("hides tldraw's built-in toolbar so it cannot duplicate our own", () => {
    // Anchored to a line start: plain 'Toolbar: null' also matches the
    // pre-existing 'RichTextToolbar: null' and would pass without the fix.
    expect(read(TLDRAW_CANVAS)).toMatch(/^\s*Toolbar: null,$/m);
  });

  // FIXED: the selected layer's code rendered twice at once — once in the
  // inspector's "Code Preview" section and once in the "Selected Layer Code"
  // panel behind the Layer Code tab, both from the same codePreview memo.
  it('renders the selected-layer code exactly once', () => {
    const builder = read(BUILDER_PAGE);
    expect(builder.match(/\{codePreview\}/g) || []).toHaveLength(1);
    expect(builder).not.toContain('Code Preview');
  });

  // FIXED: a FLOPs metric that no code path can ever populate (nothing in the
  // repo computes FLOPs) sat permanently at an em-dash.
  it('does not advertise a FLOPs metric that is never computed', () => {
    expect(read(BUILDER_PAGE)).not.toContain('FLOPs');
  });

  // FIXED: two permanent bottom bars carried 27 controls. The diagnostics drawer
  // is no longer permanent chrome — it renders only while open, opened from the
  // single toolbar — and situational tools moved into a More menu.
  it('keeps only one permanent bottom bar (diagnostics render only while open)', () => {
    const builder = read(BUILDER_PAGE);
    const idx = builder.indexOf('className="ui2-diagnostics-drawer');
    expect(idx).toBeGreaterThan(-1);
    // The drawer element is inside a !logsCollapsed guard.
    expect(builder.slice(Math.max(0, idx - 400), idx)).toContain('{!readOnlyLocalBrowser && !logsCollapsed && (');
  });

  it('keeps the situational tools reachable in the More menu, with their testids', () => {
    const builder = read(BUILDER_PAGE);
    for (const id of [
      'ui2-action-refresh-shapes', 'ui2-action-view-blocks', 'ui2-action-view-expanded',
      'ui2-action-focus', 'ui2-action-zoom-in', 'ui2-action-zoom-out', 'ui2-action-fullscreen',
    ]) {
      expect(builder).toContain(id);
    }
    // …and they are rendered from the menu list, not as bar buttons.
    expect(builder).toContain('moreTools.map((tool)');
  });
  it('does not offer built-in chat from the toolbar', () => {
    expect(read(BUILDER_PAGE)).not.toContain('ui2-action-ai-chat');
  });
});

import fs from 'fs';
import path from 'path';
import { localBrowserMode } from '../../ide/projection';

const readSource = (...parts: string[]) => fs.readFileSync(path.join(__dirname, '../..', ...parts), 'utf8');

describe('the external extension page is read-only', () => {
  afterEach(() => {
    delete (window as { __TENSORVIZ_LOCAL_BROWSER__?: boolean }).__TENSORVIZ_LOCAL_BROWSER__;
  });

  it('recognizes the external-page marker independently of generic IDE mode', () => {
    expect(localBrowserMode()).toBe(false);
    (window as { __TENSORVIZ_LOCAL_BROWSER__?: boolean }).__TENSORVIZ_LOCAL_BROWSER__ = true;
    expect(localBrowserMode()).toBe(true);
  });

  it('blocks both manual and automatic shape inference before an API call', () => {
    const builder = readSource('components', 'ModelBuilder.tsx');
    const manual = builder.slice(
      builder.indexOf('const triggerShapeInference = useCallback'),
      builder.indexOf('const inferShapesRef = useRef'),
    );
    const automatic = builder.slice(
      builder.indexOf('const inferShapesRef = useRef'),
      builder.indexOf('Apply the repairs the backend suggested'),
    );
    expect(manual.indexOf('if (localBrowserMode() || showingProjectGraph()) return;')).toBeLessThan(
      manual.indexOf('ApiService.inferShapes(request)'),
    );
    expect(automatic).toContain('if (localBrowserMode() || showingProjectGraph()) return;');
  });

  it('keeps navigation/view controls and gates write controls in the local page', () => {
    const page = readSource('components', 'ui2', 'BuilderPageV2.tsx');
    expect(page).toContain('const readOnlyLocalBrowser = localBrowserMode();');
    expect(page).not.toContain('data-testid="ui2-action-ai-chat"');
    expect(page).toContain('data-testid="ui2-action-layout"');
    expect(page).toContain('data-testid="ui2-action-fit-view"');
    expect(page).toContain('{!readOnlyLocalBrowser && <button className="ui2-primary-action"');
    expect(page).toContain('{!readOnlyLocalBrowser && paletteOpen && (');
    expect(page).toContain('onRenameModel={readOnlyLocalBrowser ? undefined');
  });

  it('hides the Blocks family legend and add palette in the local page', () => {
    const overlay = readSource('canvas', 'GraphFocusOverlay.tsx');
    expect(overlay).toContain('const local = React.useContext(LocalBrowserContext);');
    expect(overlay).toContain('{!local && <FamilyLegend nodes={graphNodes.get()} />}');

    const palette = readSource('components', 'ui2', 'BuilderPaletteV2.tsx');
    expect(palette).toContain('if (localBrowserMode()) return null;');
  });

  it('keeps node source editing and the chat rail out of local mode', () => {
    const builder = readSource('components', 'ModelBuilder.tsx');
    expect(builder).not.toContain('LocalBrowserAskRail');
    expect(builder).not.toContain('SourceEdit');
    expect(builder).toContain('sidePanelOpen={false}');
  });

  it('keeps saved-graph storage and its chrome out of the local page', () => {
    const page = readSource('components', 'ui2', 'BuilderPageV2.tsx');
    expect(page).not.toContain('Ask about this model');
    const panel = page.indexOf('<HistoryPanel open={historyOpen}');
    expect(panel).toBeGreaterThan(-1);
    expect(page.slice(Math.max(0, panel - 140), panel)).toContain('{!readOnlyLocalBrowser && HistoryPanel && (');
  });

  it('enforces read-only and focused projection guards for keyboard, command and canvas paths', () => {
    const builder = readSource('components', 'ModelBuilder.tsx');
    const canvas = readSource('canvas', 'TldrawCanvas.tsx');
    // Focus is a display projection: it must share the local page's source-write
    // protection while retaining navigation and presentation-only commands.
    expect(builder).toContain('const mutatesReadOnlyProjection = (readOnlyLocalBrowser || moduleFocusRef.current.active)');
    expect(builder).toContain('!LOCAL_BROWSER_VIEW_COMMANDS.has(command)');
    expect(builder).toContain('if (moduleFocusRef.current.active && !LOCAL_BROWSER_VIEW_COMMANDS.has(command))');
    expect(builder).not.toContain('applyCodeArtifacts');
    expect(builder).toContain('readOnly={readOnlyLocalBrowser || moduleFocus.active}');
    expect(canvas).toContain('commit: readOnly ? () => {}');
    // The canvas is NOT globally locked. Read-only means the source cannot change;
    // layout is presentation state, and this contract keeps layout/fit available —
    // but tldraw's isReadonly is all-or-nothing and took node dragging with it.
    expect(canvas).toContain('editor.updateInstanceState({ isReadonly: false }');
    expect(canvas).not.toContain('isReadonly: readOnly');
    const bridge = readSource('canvas', 'useTldrawBridge.ts');
    expect(bridge).toContain('withProjectionWriteAccess(editor');
    // Deletion was the one destructive gesture the global lock uniquely covered —
    // blockDrawingShortcuts blocks tool letters, not Delete — so it is refused here
    // instead, and only for a user gesture: the sync effects delete stray and
    // replaced shapes while isSyncingToEditor is set.
    expect(bridge).toContain('registerBeforeDeleteHandler');
    expect(bridge).toContain(
      "if (source !== 'user' || isSyncingToEditor.current || !(localBrowserMode() || graphIsReadOnly(editor) || (shape.props as any).nodeData?.focusBoundary)) return;",
    );
    expect(builder).toContain('if (localBrowserMode()) return;');

  });

  it('canvas ports and status pills make no affordance or health claims on the local page', () => {
    const port = readSource('canvas', 'NodePort.tsx');
    const inertGate = port.indexOf('if (localBrowserMode() || graphIsReadOnly(editor) || editor.getIsReadonly())');
    const inertClass = port.indexOf('model-node-port-inert');
    const affordanceTitle = port.indexOf("'Drag to connect'");
    const gesture = port.indexOf('onPointerDown={(event)');
    expect(inertGate).toBeGreaterThan(-1);
    expect(inertGate).toBeLessThan(inertClass);
    expect(inertClass).toBeLessThan(affordanceTitle);
    expect(affordanceTitle).toBeLessThan(gesture);
    expect(port.match(/onPointerDown=\{\(event\)/g)).toHaveLength(1);

    const theme = readSource('canvas', 'tldraw-theme.css');
    const inertRule = theme.indexOf('.tldraw-canvas-wrapper .model-node-port-inert {');
    expect(inertRule).toBeGreaterThan(-1);
    expect(theme.slice(inertRule, inertRule + 140)).toContain('pointer-events: none');

    const moduleShape = readSource('canvas', 'ModuleShapeUtil.tsx');
    expect(moduleShape).toContain('export function statusLabelForModule');
    expect(moduleShape).toContain("ok: 'No issues recorded'");
    expect(moduleShape).toContain("if (status === 'ok' && localBrowserMode()) return 'Source-only';");
    expect(moduleShape).toContain('statusLabelForModule(moduleStatus, data)');
  });

  it('replaces runtime diagnostics with the projection static warnings on the local page', () => {
    const builder = readSource('components', 'ModelBuilder.tsx');
    // The health probe's gate stays narrow on purpose: only the EXTERNAL page must
    // not probe, because the broker owns session truth there. The panel wants
    // backend health like any other surface — what it must not do is validate a
    // projection, which is the two widened gates above.
    const healthGate = builder.indexOf('if (localBrowserMode()) return;');
    expect(healthGate).toBeGreaterThan(-1);
    expect(builder.indexOf('ApiService.checkBackendHealth()')).toBeGreaterThan(healthGate);
    expect(builder).toContain('staticDiagnostics: readOnlyLocalBrowser ? currentProjection()?.diagnostics ?? [] : [],');

    const page = readSource('components', 'ui2', 'BuilderPageV2.tsx');
    expect(page).toContain("import { DIAGNOSTIC_LABELS } from '../../ide/localBrowserPresentation';");
    expect(page).toContain('snapshot.staticDiagnostics.map(');
    expect(page).toContain('DIAGNOSTIC_LABELS[diagnostic.code]');

    // These two panels used to be the read-only page's exception: the Inspector
    // rendered them for everyone else and hid them here. The Inspector is gone,
    // so nobody gets a runtime shape table or a quick-action stack beside the
    // canvas — which is the same guarantee, now unconditional.
    expect(page).not.toContain('Shape Inference</h4>');
    expect(page).not.toContain('Quick Actions</h4>');

    // the drawer shows one truthful static list, not four runtime tabs
    expect(page).toContain('Static resolution warnings (');
    expect(page).toContain("!readOnlyLocalBrowser && diagnosticsTab === 'logs'");
    expect(page).toContain("!readOnlyLocalBrowser && diagnosticsTab === 'shapes'");
    expect(page).toContain("!readOnlyLocalBrowser && diagnosticsTab === 'errors'");
    expect(page).toContain("(readOnlyLocalBrowser || diagnosticsTab === 'warnings')");
    expect(page).toContain('No static resolution warnings.');
  });

  it('bounds the local page to the viewport and keeps one truth header', () => {
    const app = readSource('ide', 'LocalBrowserApp.tsx');
    expect(app).toContain('<div className="local-browser-content">');
    expect(app).not.toContain('<div style={{ flex: 1, minHeight: 0 }}>');

    const css = readSource('ide', 'LocalBrowserApp.css');
    const contentRule = css.indexOf('.local-browser-content {');
    expect(contentRule).toBeGreaterThan(-1);
    expect(css.slice(contentRule, contentRule + 160)).toContain('min-height: 0');
    expect(css).toContain('.local-browser-content > .ui2-app-bg.h-screen {\n  height: 100%;\n}');

    const shell = readSource('components', 'ui2', 'WorkbenchShell.tsx');
    expect(shell).toContain("chrome?: 'full' | 'minimal' | 'ide' | 'local';");
    expect(shell).toContain("chrome === 'minimal' || chrome === 'ide' || chrome === 'local'");
    expect(shell).toContain("const localChrome = chrome === 'local';");
    expect(shell).toContain('{!localChrome && (minimal ? (');

    // The local page asks for the headerless chrome BEFORE the IDE check —
    // inside this bundle both flags are true, so the order is the contract.
    const page = readSource('components', 'ui2', 'BuilderPageV2.tsx');
    expect(page).toContain("chrome={readOnlyLocalBrowser ? 'local' : inIde ? 'ide' : 'minimal'}");
  });

  it('dismisses its overlays with Escape and restores focus to the opener', () => {
    const app = readSource('ide', 'LocalBrowserApp.tsx');
    expect(app).toContain('const warningToggleRef = React.useRef<HTMLButtonElement | null>(null);');
    expect(app).toContain('ref={warningToggleRef}');
    expect(app).toContain("if (event.key !== 'Escape') return;");
    expect(app).toContain('warningToggleRef.current?.focus();');

    const page = readSource('components', 'ui2', 'BuilderPageV2.tsx');
    expect(page).toContain('if (!moreOpen && !paletteOpen && logsCollapsed) return;');
    expect(page).toContain('if (quickJumpOpen) return;');
    expect(page).toContain('if (moreOpen) moreButtonRef.current?.focus();');
    expect(page).toContain(`document.querySelector('[data-testid="ui2-toggle-logs"]')`);
  });

  it('delivers focus to the canvas itself and names quick-jump results uniquely', () => {
    const canvas = readSource('canvas', 'TldrawCanvas.tsx');
    // tldraw blurs its skip-to-main-content button first, then focuses the
    // container on a timer — and only when a shape exists. The canvas must land
    // that promised focus itself, or the keyboard user ends on BODY.
    const listener = canvas.indexOf("ownerDoc.addEventListener('click', onSkipToMainContent, true)");
    const cleanup = canvas.indexOf("ownerDoc.removeEventListener('click', onSkipToMainContent, true)");
    expect(listener).toBeGreaterThan(-1);
    expect(cleanup).toBeGreaterThan(listener);
    expect(canvas).toContain(".closest('.tl-skip-to-main-content')");
    expect(canvas).toContain('ownerDoc.activeElement !== container');

    const page = readSource('components', 'ui2', 'BuilderPageV2.tsx');
    // Registered paths identify captured modules; opaque IDs distinguish repeat invocations.
    expect(page).toContain('`${n.title} — ${n.path ?? n.id}${n.path && n.path !== n.id ? ` — ${n.id}`');
    expect(page).toContain('<span className="local-canvas-search-path">{n.path ?? n.id}</span>');
    expect(page).toContain('<span className="ui2-quickjump-name">{n.title}</span>');
    expect(page).toContain('<span className="ui2-quickjump-id">{n.id}</span>');
    expect(page).toContain('<span className="ui2-quickjump-type ui2-muted">{n.title}</span>');

    const tokens = readSource('styles', 'ui2.tokens.css');
    expect(tokens).toContain('.ui2-quickjump-id {');
    expect(tokens).toContain('.ui2-quickjump-type {');
  });
});

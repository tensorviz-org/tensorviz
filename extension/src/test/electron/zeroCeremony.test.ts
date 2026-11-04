import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';

import * as vscode from 'vscode';

import { WorkspaceClass } from '../../workspaceClassIndex';
import { OPEN_AS_GRAPH } from '../../commandIds';
import { ModelsNode, ModelsTree } from '../../views';

const EXTENSION_ID = 'tensorviz.tensorviz-vscode';

/** The checkout, three directories up from out/test/electron. */
const REPO = path.resolve(__dirname, '..', '..', '..', '..');

/**
 * The folder this suite is about, and it must be THE open workspace folder.
 *
 * Every other Electron suite copies the fixture to a temp directory and opens no folder at
 * all, which is fine for what they test and fatal for this one: with no folder open the
 * class index is keyed by the file's own directory, so opening `models/resid.py` indexes
 * `models/` and finds four classes. Four is the number a broken scan and a healthy one both
 * produce. Nine is the number that says the whole folder was searched.
 */
const FIXTURE = process.env.TENSORVIZ_ZERO_CEREMONY_FIXTURE
  ?? path.join(REPO, 'backend', 'tests', 'fixtures', 'ide_repo');

/** Every `nn.Module` in the fixture: four under `models/`, five under `custom/`. `Cfg` is a
 *  dataclass and is deliberately absent — the scan lists models, not classes. */
const EVERY_CLASS = [
  'Branchy',
  'Configured',
  'Custom',
  'Declared',
  'Gate',
  'MyBlock',
  'Net',
  'Symbolic',
  'Transposed',
];

/** task.md §8.3: time-to-first-graph on `ide_repo`, from folder open, cold backend. */
const FIRST_GRAPH_BUDGET_MS = 30_000;

interface AppMessage {
  type: string;
  [key: string]: unknown;
}

interface TestApi {
  messages(): AppMessage[];
  backendOrigin(): string | undefined;
  classesIn(root: string): WorkspaceClass[] | undefined;
  localBrowserTarget(): { cls: string } | undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Poll until `read` returns something truthy, or fail with `what`. */
async function until<T>(what: string, timeoutMs: number, read: () => T | undefined): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = read();
    if (value) {
      return value;
    }
    assert.ok(Date.now() < deadline, `timed out waiting for ${what}`);
    await sleep(100);
  }
}

/** The newest projectionApplied the app has posted. */
function lastApplied(api: TestApi): (AppMessage & { nodeCount: number }) | undefined {
  const applied = api.messages().filter((message) => message.type === 'projectionApplied');
  return applied[applied.length - 1] as never;
}

/** The `tensorviz.*` settings this build contributes, without the section prefix. Read out of
 *  the manifest rather than listed here, so a setting added later is covered by "none of them
 *  is set" instead of quietly escaping it. */
function contributedSettings(packageJSON: unknown): string[] {
  const properties = (
    packageJSON as { contributes?: { configuration?: { properties?: Record<string, unknown> } } }
  )?.contributes?.configuration?.properties;
  return Object.keys(properties ?? {})
    .filter((key) => key.startsWith('tensorviz.'))
    .map((key) => key.slice('tensorviz.'.length));
}

/**
 * Open a folder; the models are listed; the graph is beside the code. Nothing configured
 * (task.md P0).
 *
 * P0 is a milestone about *absence*, so almost everything here is asserted about things that
 * did not happen, and each one has a shape that would otherwise pass unnoticed:
 *
 *  * the Models view fills in with no `tensorviz.*` setting set. A run that inherited
 *    `tensorviz.repoPath` from a sibling suite would prove the opposite of the milestone, so
 *    the setting is cleared and its absence asserted rather than assumed;
 *  * nobody was asked to press Start or to type a path. The old Models view had a "Start
 *    tensorViz to scan" row and the old backend resolution asked for `repoPath`; both are
 *    deleted, and a deleted prompt only stays deleted if something fails when it comes back;
 *  * the folder is searched, not the file's directory. Nine classes, not four;
 *  * and the number the milestone is judged on gets printed: folder open to first graph,
 *    against the 30 s budget in §8.3.
 *
 * What this suite CANNOT prove on its own is which of `resolveBackendDir`'s sources answered.
 * That ordering is pinned by `src/test/unit/resolveBackendDir.test.ts`; what is checked here
 * is the precondition that makes the shipped answer available at all — a backend staged
 * inside the extension — and that no setting was needed on top of it.
 */
suite('zeroCeremony', () => {
  let api: TestApi;
  /** Wall clock from the earliest moment this suite can observe, which is as close to
   *  "the folder was opened" as an in-host test can stand. */
  let openedAt: number;
  let residUri: vscode.Uri;
  /** Every message the extension put in front of the user while this suite ran. */
  const shown: string[] = [];
  const restore: Array<() => void> = [];
  /** `tensorviz.*` global values found on the way in, put back on the way out: the settings
   *  store is shared with every other suite in the run. */
  const cleared = new Map<string, unknown>();

  /**
   * Record what a `window.show*Message` was called with, and answer nothing.
   *
   * Deliberately not calling through. The product's own calls are `void`-ed and would not
   * block, but a prompt that is never answered is a promise that never settles, and the whole
   * premise here is that nothing should be asking. If something does ask, the assertion below
   * says so by name; letting the real dialog open as well would only add a notification
   * nobody in CI will ever dismiss.
   */
  function capture(name: 'showInformationMessage' | 'showWarningMessage' | 'showErrorMessage'): void {
    const surface = vscode.window as unknown as Record<string, unknown>;
    const original = surface[name];
    const spy = (message: unknown): Thenable<undefined> => {
      shown.push(String(message));
      return Promise.resolve(undefined);
    };
    surface[name] = spy;
    // If the namespace were frozen the assignment would be a silent no-op and every
    // assertion about notifications below would pass without observing anything.
    assert.equal(surface[name], spy, `vscode.window.${name} could not be captured`);
    restore.push(() => {
      surface[name] = original;
    });
  }

  suiteSetup(async function () {
    this.timeout(240_000);
    openedAt = Date.now();

    // Never skipped (task.md M2, "Test infrastructure rule"): an Electron run without an
    // interpreter proves nothing, and a skip that reads as green is the failure mode the
    // rule exists to prevent.
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'not run: TENSORVIZ_TEST_PYTHON unset');
    assert.ok(
      fs.existsSync(path.join(REPO, 'extension', 'media', 'app', 'index.html')),
      'not run: extension/media/app/index.html is missing — run `npm run build:webview` first',
    );
    // The one thing that has to exist for a folder with no settings to work at all: source 2
    // of `resolveBackendDir`, the backend packaged inside the extension. Sources 1 (a bundle)
    // and 3 (`tensorviz.repoPath`) are both unavailable here on purpose.
    // Skip, not assert, for the same reason as the workspace folder below: `extension/backend/`
    // is generated and gitignored, so a clean checkout does not have it and a run that has not
    // packaged is the normal case rather than a job configured wrong. It cannot simply be
    // staged here either — its presence changes which source `resolveBackendDir` picks for
    // EVERY suite in the run, so a test that created it would quietly change what the others
    // are testing.
    if (!fs.existsSync(path.join(REPO, 'extension', 'backend', 'server_v2.py'))) {
      console.log(
        'not run: extension/backend/server_v2.py is missing — run `npm run stage:backend` ' +
          'first. Without it there is no backend a folder with no tensorviz.* settings can ' +
          'reach, and this suite would be measuring the absence of the staging step, not zero ' +
          'ceremony.',
      );
      this.skip();
    }
    process.env.BACKEND_PYTHON = python;

    // The one precondition this suite cannot assert its way to. Every other suite makes its own
    // temp workspace; this one is ABOUT opening a folder with nothing configured, so the folder
    // has to be opened by the host — which needs its own entry in .vscode-test.mjs with
    // `workspaceFolder` pointing at the fixture. The shared config has no such entry yet, so
    // under it this suite skips with the reason printed rather than failing a hook and taking
    // the rest of the run down with it. Skipping is right here and wrong for
    // TENSORVIZ_TEST_PYTHON: a missing interpreter is a job configured wrong, while this is a
    // suite the harness simply did not launch.
    const folders = vscode.workspace.workspaceFolders ?? [];
    const opened = folders.map((folder) => fs.realpathSync(folder.uri.fsPath));
    if (opened.length !== 1 || opened[0] !== fs.realpathSync(FIXTURE)) {
      console.log(
        `not run: zeroCeremony needs ${FIXTURE} as the single open workspace folder; saw ` +
          `${JSON.stringify(opened)}. Give it its own config in .vscode-test.mjs with ` +
          '`workspaceFolder` set to the fixture, and it will run.',
      );
      this.skip();
    }
    assert.ok(
      vscode.workspace.isTrusted,
      'not run: the workspace is restricted, and P0 is about the editor\'s own trust dialog ' +
        'being the only one. Launch the host with --disable-workspace-trust.',
    );

    const extension = vscode.extensions.getExtension(EXTENSION_ID);
    assert.ok(extension, `${EXTENSION_ID} was not loaded by the extension host`);
    api = ((await extension.activate()) as { __test: TestApi }).__test;

    // Nothing configured. Cleared rather than merely asserted, because every other Electron
    // suite writes `tensorviz.repoPath` at global scope and the settings store outlives them.
    // A workspace-scoped value is asserted away instead of cleared: clearing one would write
    // a .vscode/settings.json into the fixture the repository ships.
    const settings = vscode.workspace.getConfiguration('tensorviz');
    for (const key of contributedSettings(extension.packageJSON)) {
      const inspected = settings.inspect(key);
      assert.equal(
        inspected?.workspaceValue,
        undefined,
        `tensorviz.${key} is set in the fixture's own .vscode/settings.json; delete it`,
      );
      assert.equal(
        inspected?.workspaceFolderValue,
        undefined,
        `tensorviz.${key} is set for the workspace folder; delete it`,
      );
      if (inspected?.globalValue !== undefined) {
        cleared.set(key, inspected.globalValue);
        await settings.update(key, undefined, vscode.ConfigurationTarget.Global);
      }
    }

    capture('showInformationMessage');
    capture('showWarningMessage');
    capture('showErrorMessage');

    // The user's whole contribution: they open one of their model files. Everything after
    // this is the extension's job.
    residUri = vscode.Uri.file(path.join(FIXTURE, 'models', 'resid.py'));
    await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(residUri));
  });

  suiteTeardown(async () => {
    for (const undo of restore) {
      undo();
    }
    const settings = vscode.workspace.getConfiguration('tensorviz');
    for (const [key, value] of cleared) {
      await settings.update(key, value, vscode.ConfigurationTarget.Global);
    }
    await vscode.commands.executeCommand('workbench.action.closeAllEditors');
  });

  test('the Models view lists every class in the folder, with nothing configured', async function () {
    this.timeout(60_000);
    const settings = vscode.workspace.getConfiguration('tensorviz');
    for (const key of contributedSettings(
      vscode.extensions.getExtension(EXTENSION_ID)!.packageJSON,
    )) {
      const inspected = settings.inspect(key);
      assert.equal(
        inspected?.globalValue ?? inspected?.workspaceValue ?? inspected?.workspaceFolderValue,
        undefined,
        `tensorviz.${key} is set, so this is not the no-configuration path any more`,
      );
    }

    // The real Models view over the real index. `ModelsTree` is what the activity bar
    // renders, and its rows come from `roots()` — the open workspace folders — crossed with
    // the class index. Reading `classesIn` directly would skip exactly that crossing, which
    // is the half of this that decides between nine classes and four.
    const changed = new vscode.EventEmitter<void>();
    const models = new ModelsTree({
      classesFor: (root) => api.classesIn(root),
      // The tree never calls either of these; it reads the index the extension keeps and
      // redraws when the extension says so. The adapter is a read-only window onto that
      // index, not a second one, which is why refreshing through it does nothing.
      refresh: async () => undefined,
      onDidChangeClasses: changed.event,
    });

    const modelRows = (parent?: ModelsNode): ModelsNode[] => models.getChildren(parent).flatMap(
      (node) => node.kind === 'model' ? [node] : modelRows(node),
    );
    const listed = await until(
      `the Models view to list ${EVERY_CLASS.length} classes`,
      FIRST_GRAPH_BUDGET_MS,
      () => {
        const rows = modelRows();
        return rows.length >= EVERY_CLASS.length ? rows : undefined;
      },
    );
    assert.deepEqual([...listed.map((row) => row.name)].sort(), EVERY_CLASS);
    assert.deepEqual(models.getChildren().map((row) => row.name), ['custom', 'models']);

    // A listed model has to be one click from a graph, so the row carries the command and
    // the file it was found in — the two things a click needs.
    const net = listed.find((row) => row.name === 'Net')!;
    const item = models.getTreeItem(net);
    assert.equal(item.command?.command, OPEN_AS_GRAPH);
    assert.deepEqual(item.command?.arguments, [
      { uri: vscode.Uri.file(path.join(FIXTURE, 'models', 'resid.py')).toString(), cls: 'Net' },
    ]);

    models.dispose();
    changed.dispose();
  });

  test('nobody was asked to start anything or to type a path', () => {
    // The two sentences P0 deleted. "Start tensorViz to scan" was the Models view's own row
    // and `repoPath` was the question the backend used to ask for; both had an answer the
    // user was expected to supply, and both are now the extension's problem. This is the
    // assertion that keeps them deleted.
    const ceremony = shown.filter((message) => /Start tensorViz|repoPath/.test(message));
    assert.deepEqual(ceremony, [], `the user was asked for something: ${JSON.stringify(shown)}`);
  });

  test('Open as Graph draws the selected model on the current Canvas', async function () {
    this.timeout(120_000);
    const askedAt = Date.now();
    await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: residUri.toString(), cls: 'Net' });
    assert.ok(api.backendOrigin(), 'the backend never started');

    const applied = await until('the app to post projectionApplied', 60_000, () => lastApplied(api));
    assert.ok(applied.nodeCount > 0);
    assert.equal(api.localBrowserTarget()?.cls, 'Net');

    // task.md §8.3: printed, not asserted — a number over budget schedules a fix in the PR's
    // status row, it does not fail the gate. Two numbers, because they answer different
    // questions: the budget is about folder open to first graph, and the second says how
    // much of it was the click itself, so a slow index and a slow backend are told apart.
    // The first number includes whatever earlier tests in this suite waited for, so read it
    // as an upper bound when they are red.
    const total = Date.now() - openedAt;
    // eslint-disable-next-line no-console
    console.log(
      `[zeroCeremony] time-to-first-graph ${(total / 1000).toFixed(1)}s ` +
        `against the ${FIRST_GRAPH_BUDGET_MS / 1000}s budget (task.md §8.3); ` +
        `${((Date.now() - askedAt) / 1000).toFixed(1)}s of it after Open as Graph`,
    );
  });
});

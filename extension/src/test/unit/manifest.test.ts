import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { DEFAULT_MCP_PORT } from '../../mcpServer';
import { WALKTHROUGH_ID } from '../../walkthrough';
import { MODELS_VIEW, NEW_MODEL_COMMAND, TEMPLATES_VIEW } from '../../views';

/** The checkout, four directories up from out/test/unit. */
const REPO = path.resolve(__dirname, '..', '..', '..', '..');
const EXTENSION = path.join(REPO, 'extension');

const manifest = JSON.parse(
  fs.readFileSync(path.join(EXTENSION, 'package.json'), 'utf8'),
) as Record<string, any>;
const contributes = manifest.contributes as Record<string, any>;

/**
 * The manifest is a contract with three different readers — VS Code, the fork's product.json,
 * and the code in src/ — and none of them fails in a way you can see from a passing build.
 * A theme id that stopped matching ide/patches/0003 boots the fork on the upstream dark theme;
 * a walkthrough step whose media file was renamed renders as an empty panel; a
 * `configurationDefaults` key that is not ours changes a setting in somebody else's editor.
 * Each of those is one line here.
 */
suite('extension manifest', () => {
  test('sidebar actions follow actual visibility without taking over other views', () => {
    assert.equal(manifest.engines.vscode, '^1.106.0');
    assert.ok(contributes.viewsContainers.secondarySidebar.some((container: any) => container.id === 'tensorviz-canvas'));
    assert.equal(contributes.viewsContainers.secondarySidebar.find((container: any) => container.id === 'tensorviz-canvas').title, 'tensorViz');
    assert.deepEqual(contributes.views['tensorviz-canvas'].map((view: any) => [view.id, view.type, view.when]),
      [['tensorviz.sidebarCanvas', 'webview', undefined]]);
    const editor = contributes.menus['editor/title'];
    for (const [command, when] of [['tensorviz.showSidebarCanvas', '!tensorviz.canvasVisible'], ['tensorviz.hideCanvas', 'tensorviz.canvasVisible']]) {
      const entries = editor.filter((entry: any) => entry.command === command);
      assert.equal(entries.length, 2, 'toolbar and overflow both expose the visibility action');
      assert.ok(entries.every((entry: any) => entry.when === when));
      assert.ok(entries.some((entry: any) => entry.group.startsWith('navigation')));
      assert.ok(entries.some((entry: any) => !entry.group.startsWith('navigation')));
    }
    for (const command of ['tensorviz.hideCanvas', 'tensorviz.openCanvasInBrowser']) {
      assert.ok(contributes.menus['view/title'].some((entry: any) => entry.command === command && entry.when === 'view == tensorviz.sidebarCanvas'));
    }
    assert.deepEqual(contributes.keybindings.find((entry: any) => entry.command === 'tensorviz.toggleCanvas'),
      { command: 'tensorviz.toggleCanvas', key: 'ctrl+cmd+g', mac: 'ctrl+cmd+g', win: 'ctrl+alt+g', linux: 'ctrl+alt+g', when: '!terminalFocus' });
    const preference = contributes.configuration.properties['tensorviz.graphSurface'];
    assert.equal(preference.default, 'sidebar');
    assert.match(preference.deprecationMessage, /All values now open the sidebar/);
  });

  test('the MCP endpoint keeps a constant default port the code also uses', () => {
    // The setting default and the extension's config fallback are read by different
    // readers; a drift makes the endpoint bind a different port than the manifest
    // promises, so a client URL stops matching on reconnect.
    assert.equal(contributes.configuration.properties['tensorviz.mcpPort'].default, DEFAULT_MCP_PORT);
  });

  test('does not contribute analytics settings', () => {
    const properties = contributes.configuration.properties as Record<string, unknown>;
    assert.equal(properties['tensorviz.analytics.enabled'], undefined);
    assert.equal(properties['tensorviz.analytics.projectToken'], undefined);
    assert.equal(properties['tensorviz.analytics.host'], undefined);
  });
  test('the deprecated capture-on-save setting stays in the manifest and does not execute', () => {
    const setting = contributes.configuration.properties['tensorviz.traceOnSave'];
    assert.equal(setting.default, false);
    assert.equal(contributes.configurationDefaults['tensorviz.traceOnSave'], false);
    assert.match(setting.deprecationMessage, /Saving refreshes source only/);
  });

  test('contributes the first-run walkthrough the code opens', () => {
    const walkthroughs = contributes.walkthroughs as Array<Record<string, any>>;
    assert.equal(walkthroughs?.length, 1);
    assert.equal(walkthroughs[0].id, WALKTHROUGH_ID);
    assert.ok(walkthroughs[0].steps.length >= 1);
  });

  test('every walkthrough step points at a media file that exists', () => {
    for (const step of contributes.walkthroughs[0].steps as Array<Record<string, any>>) {
      const markdown = step.media?.markdown as string | undefined;
      assert.ok(markdown, `step ${step.id} has no markdown media`);
      assert.ok(
        fs.existsSync(path.join(EXTENSION, markdown)),
        `step ${step.id} names ${markdown}, which is not in the extension`,
      );
      // extension/media/ is gitignored (it is where the built React app is staged), so a
      // walkthrough asset there would pass locally and be missing from every clone.
      assert.ok(
        !markdown.startsWith('media/'),
        `${markdown} is under extension/media/, which .gitignore excludes`,
      );
    }
  });

  test('the Hugging Face panel assets are tracked outside the generated media build', () => {
    for (const name of ['hugging-face.js', 'hugging-face.css']) {
      assert.ok(fs.existsSync(path.join(EXTENSION, 'webview', name)), `${name} is missing from extension/webview`);
    }
  });

  test('keeps model tools without built-in chat or provider setup', () => {
    assert.deepEqual(contributes.viewsContainers.activitybar.map((v: any) => v.id), ['tensorviz']);
    assert.deepEqual(contributes.viewsContainers.secondarySidebar.map((v: any) => v.id), ['tensorviz-canvas']);
    assert.equal(manifest.engines.vscode, '^1.106.0');
    assert.deepEqual(contributes.views['tensorviz-canvas'].map((v: any) => [v.id, v.type, v.when]),
      [['tensorviz.sidebarCanvas', 'webview', undefined]]);
    assert.deepEqual(contributes.views.tensorviz.map((v: any) => v.id), [MODELS_VIEW, TEMPLATES_VIEW]);
    assert.equal(contributes.views['tensorviz-agent'], undefined);
    assert.equal(contributes.viewsWelcome, undefined);
    const commands = contributes.commands.map((c: any) => c.command);
    for (const obsolete of ['tensorviz.fixWithAgent', 'tensorviz.setApiKey', 'tensorviz.removeApiKey',
      'tensorviz.runRecipe', 'tensorviz.refreshRuns', 'tensorviz.compareRuns',
      'tensorviz.manageAnalytics', 'tensorviz.showAnalyticsData', 'tensorviz.resetAnalyticsIdentity',
      'tensorviz.giveAnalyticsFeedback']) {
      assert.ok(!commands.includes(obsolete));
    }
    assert.ok(commands.includes('tensorviz.connectMcp'));
    assert.ok(commands.includes('tensorviz.signInToWebsite'));
    assert.ok(commands.includes('tensorviz.signOutOfWebsite'));
  });

  test('configurationDefaults carries our own settings and nothing else', () => {
    // The acceptance grep in task.md F1, as a test. Anything else here — usage reporting, the startup
    // editor, the colour theme — is a product default and belongs in ide/patches/0003, because
    // this manifest also installs into stock VS Code and Cursor.
    const defaults = contributes.configurationDefaults as Record<string, unknown>;
    const leaked = Object.keys(defaults).filter(
      (key) => !key.startsWith('tensorviz.') && !key.startsWith('['),
    );
    assert.deepEqual(leaked, []);
  });

  test('no language-scoped default changes how somebody else’s files look', () => {
    // The rule above allows a `[python]` block, and that turned out to be the hole: it says
    // nothing about what goes INSIDE one. What went inside was `editor.rulers: [88]`, which drew
    // a vertical line down every Python file in every workspace of everyone who installed this
    // extension, whether or not they ever opened a graph. A user noticed it and asked what it
    // was, which is the whole argument.
    //
    // The companion key, `editor.codeLens: true`, was measured in a real VS Code: the built-in
    // default is already true, so it could never enable anything — it could only override
    // somebody who had deliberately turned lenses off.
    //
    // If a language-scoped default is ever genuinely needed, this test is where the case gets
    // made. Deleting it to add one is the decision, and it should look like one.
    const defaults = contributes.configurationDefaults as Record<string, unknown>;
    const scoped = Object.keys(defaults).filter((key) => key.startsWith('['));
    assert.deepEqual(
      scoped,
      [],
      `language-scoped defaults change files this extension was never opened for: ${scoped.join(', ')}`,
    );
  });

  test('no API proposal is declared here — the fork grants them', () => {
    // Declaring one trips `vsce package` and disables the extension on a released build, so
    // this must stay absent no matter how convenient a proposal looks (task.md F1).
    assert.equal(manifest.enabledApiProposals, undefined);
  });

  test('every command a view or a walkthrough invokes is contributed', () => {
    const contributed = new Set(
      (contributes.commands as Array<Record<string, any>>).map((entry) => entry.command),
    );
    const referenced = new Set<string>([NEW_MODEL_COMMAND]);
    for (const items of Object.values(contributes.menus as Record<string, Array<Record<string, any>>>)) {
      items.forEach((item) => referenced.add(item.command));
    }
    const linked = JSON.stringify(contributes).match(/command:([A-Za-z0-9_.]+)/g) ?? [];
    for (const link of linked) {
      const command = link.slice('command:'.length);
      // Workbench commands are the host's, not ours; only our own must be contributed.
      if (command.startsWith('tensorviz.')) {
        referenced.add(command);
      }
    }
    const missing = [...referenced].filter((command) => !contributed.has(command));
    assert.deepEqual(missing, [], 'these are invoked but never contributed');
  });

  test('every command this manifest contributes is registered somewhere in src', () => {
    // The other direction, and the one that fails silently. A contributed command
    // with no `registerCommand` appears in the palette, on a menu, in the SCM
    // gutter — and does nothing at all when pressed. VS Code logs "command not
    // found" to a console nobody has open, the build passes, and the feature looks
    // shipped. P6 added two commands and three menus in one change, which is
    // exactly the shape of edit that leaves one behind.
    //
    // Registration is matched statically because activating the extension needs a
    // real editor. Ids reach `registerCommand` either as a literal or as a
    // constant, so exported string constants are resolved first.
    const sources = new Map<string, string>();
    const walk = (dir: string): void => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          if (entry.name !== 'test') walk(full);
        } else if (entry.name.endsWith('.ts')) {
          sources.set(full, fs.readFileSync(full, 'utf8'));
        }
      }
    };
    walk(path.join(EXTENSION, 'src'));
    const all = [...sources.values()].join('\n');

    const constants = new Map<string, string>();
    for (const match of all.matchAll(/const\s+([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=\s*'([^']+)'/g)) {
      constants.set(match[1], match[2]);
    }

    const registered = new Set<string>();
    for (const match of all.matchAll(/registerCommand\(\s*(?:'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))/g)) {
      const [, literal, identifier] = match;
      if (literal) registered.add(literal);
      else if (identifier && constants.has(identifier)) registered.add(constants.get(identifier)!);
    }

    const unregistered = (contributes.commands as Array<Record<string, any>>)
      .map((entry) => entry.command as string)
      .filter((command) => !registered.has(command));
    assert.deepEqual(
      unregistered,
      [],
      'contributed, shown to the user, and nothing happens when pressed',
    );
  });

  test('the theme the fork boots on exists, and this extension still ships its own', () => {
    // Two claims. Our theme must remain real and loadable, because it is still
    // contributed and a user can pick it. And the theme the fork STARTS on must name
    // something that exists — a string matching nothing falls back to upstream's dark
    // theme in silence, and the fork looks like Code - OSS with a different name.
    const theme = (contributes.themes as Array<Record<string, any>>)[0];
    assert.equal(theme.id, 'tensorViz Dark');
    assert.equal(theme.uiTheme, 'vs-dark');

    const themeFile = path.join(EXTENSION, theme.path);
    assert.ok(fs.existsSync(themeFile), `${theme.path} does not exist`);
    const parsed = JSON.parse(fs.readFileSync(themeFile, 'utf8')) as Record<string, unknown>;
    assert.equal(parsed.type, 'dark');
    assert.ok(Object.keys(parsed.colors as object).length > 0);
  });

  test('Set Color Theme offers loadable tensorViz light and dark themes with readable light text', () => {
    const themes = contributes.themes as Array<{ id: string; label: string; uiTheme: string; path: string }>;
    assert.deepEqual(themes.map(theme => [theme.id, theme.label, theme.uiTheme]), [
      ['tensorViz Dark', 'tensorViz Dark', 'vs-dark'],
      ['tensorViz Light', 'tensorViz Light', 'vs'],
    ]);
    for (const theme of themes) {
      const parsed = JSON.parse(fs.readFileSync(path.join(EXTENSION, theme.path), 'utf8'));
      assert.equal(parsed.name, theme.label);
      assert.equal(parsed.type, theme.uiTheme === 'vs' ? 'light' : 'dark');
      assert.ok(Object.keys(parsed.colors).length > 0);
    }
    const light = JSON.parse(fs.readFileSync(path.join(EXTENSION, themes[1].path), 'utf8'));
    const luminance = (hex: string) => {
      const channels = [1, 3, 5].map(offset => parseInt(hex.slice(offset, offset + 2), 16) / 255)
        .map(channel => channel <= .04045 ? channel / 12.92 : ((channel + .055) / 1.055) ** 2.4);
      return channels[0] * .2126 + channels[1] * .7152 + channels[2] * .0722;
    };
    const background = luminance(light.colors['editor.background']);
    assert.ok(background > .9, 'Light must actually render a light editor surface');
    for (const foreground of [light.colors['editor.foreground'], ...light.tokenColors.map((token: any) => token.settings.foreground)]) {
      assert.ok((background + .05) / (luminance(foreground) + .05) >= 4.5, `${foreground} must be readable on the light editor`);
    }
  });

  test('activates on startup and on a Python workspace, and on nothing else', () => {
    // onStartupFinished so the views and the walkthrough exist before a command is run.
    // workspaceContains:**/*.py is task.md P0: a folder with Python in it is a folder
    // where the Models view should already be filling in, and the backend now starts
    // itself rather than waiting for the user to press something.
    //
    // Pinned as an exact list rather than a contains-check, because every entry added
    // here wakes the extension in somebody else's window: this manifest ships to stock
    // VS Code and Cursor too.
    assert.deepEqual(manifest.activationEvents, ['onStartupFinished', 'workspaceContains:**/*.py']);
  });
});

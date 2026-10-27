import * as assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as vscode from 'vscode';

import {
  MARKDOWN_MESSAGE_PROPOSAL,
  ModelsTree,
  ModelsNode,
  MODEL_TEMPLATES,
  ScaffoldHost,
  classNameFor,
  markdownMessageGranted,
  modelsMessage,
  renderTemplate,
  scaffoldModel,
} from '../../views';
import { WorkspaceClass } from '../../workspaceClassIndex';
import { reset, state } from '../vscodeStub';

/** Any Python will do: this parses source with `ast`, which needs no torch. `python3` is a
 *  hard requirement of ide/setup.sh and of docs/check_links.py, so it is not an assumption. */
const PYTHON = process.env.TENSORVIZ_TEST_PYTHON || 'python3';

suite('markdownMessageGranted', () => {
  setup(reset);

  test('true only when the host granted the proposal this build asks for', () => {
    assert.equal(markdownMessageGranted({ enabledApiProposals: [MARKDOWN_MESSAGE_PROPOSAL] }), true);
  });

  test('false in every channel that did not — which is the degraded path', () => {
    // Stock VS Code and Cursor never see the grant: it lives in the fork's product.json, and
    // this extension is forbidden from declaring it in package.json.
    assert.equal(markdownMessageGranted({}), false);
    assert.equal(markdownMessageGranted(undefined), false);
    assert.equal(markdownMessageGranted({ enabledApiProposals: [] }), false);
    assert.equal(markdownMessageGranted({ enabledApiProposals: ['treeViewReveal'] }), false);
    // A string rather than a list is what a hand-edited product.json produces.
    assert.equal(markdownMessageGranted({ enabledApiProposals: MARKDOWN_MESSAGE_PROPOSAL }), false);
  });
});

suite('modelsMessage', () => {
  test('says nothing at all once there are classes to list', () => {
    assert.equal(modelsMessage({ markdown: true, hasWorkspace: true, indexed: true, classCount: 4 }), undefined);
  });

  test('the degraded path keeps the sentence and loses only the link', () => {
    const granted = modelsMessage({ markdown: true, hasWorkspace: true, indexed: true, classCount: 0 });
    const degraded = modelsMessage({ markdown: false, hasWorkspace: true, indexed: true, classCount: 0 });

    assert.ok(granted && degraded);
    assert.equal(granted.markdown, true);
    assert.equal(degraded.markdown, false);
    assert.ok(granted.text.includes('command:tensorviz.newModelFromTemplate'));
    assert.ok(
      !degraded.text.includes('command:'),
      'a command: link in a plain string renders as literal markup, which is worse than no link',
    );
    assert.ok(
      degraded.text.includes('nn.Module'),
      'the explanation must survive the degrade; only the shortcut is allowed to go',
    );
  });

  test('distinguishes "no folder", "indexing" and "nothing here"', () => {
    // The middle state used to tell the user to start tensorViz. Nothing needs
    // starting since task.md P0 — the backend comes up on its own — so the honest
    // sentence is that the scan is running, and the empty state names the folder
    // that was actually searched rather than saying "here".
    const noFolder = modelsMessage({ markdown: false, hasWorkspace: false, indexed: false, classCount: 0 });
    const indexing = modelsMessage({
      markdown: false, hasWorkspace: true, indexed: false, classCount: 0, folderName: 'mini-lm',
    });
    const empty = modelsMessage({
      markdown: false, hasWorkspace: true, indexed: true, classCount: 0, folderName: 'mini-lm',
    });

    assert.ok(noFolder?.text.includes('Open the folder'));
    assert.ok(indexing?.text.includes('Indexing mini-lm'));
    assert.equal(indexing?.text.includes('Start tensorViz'), false);
    assert.ok(empty?.text.includes('No nn.Module classes found in mini-lm'));
  });
});

suite('classNameFor', () => {
  test('turns a file stem into a class name', () => {
    assert.equal(classNameFor('mlp'), 'Mlp');
    assert.equal(classNameFor('my_net'), 'MyNet');
    assert.equal(classNameFor('resid-block'), 'ResidBlock');
  });

  test('falls back rather than producing something that is not an identifier', () => {
    // The result is pasted straight into a `class` statement, so "2d" would be a SyntaxError
    // in the file the user is about to open.
    assert.equal(classNameFor('2d'), 'Model');
    assert.equal(classNameFor(''), 'Model');
    assert.equal(classNameFor('---'), 'Model');
  });
});

suite('model templates', () => {
  test('every template renders the class name the file name implies', () => {
    for (const template of MODEL_TEMPLATES) {
      const rendered = renderTemplate(template, 'MyNet');
      assert.ok(rendered.includes('class MyNet(nn.Module):'), `${template.id} lost its class name`);
      assert.ok(!rendered.includes('{cls}'), `${template.id} left a placeholder behind`);
    }
  });

  test('every template is valid Python that defines exactly one nn.Module', function () {
    // The whole promise of Templates is "a file you can Open as Graph immediately". A template
    // that does not parse breaks that at the first click, and only Python can say whether it
    // parses — so this shells out rather than eyeballing the string. It never skips: a missing
    // python3 fails the suite with the spawn error, which is the honest outcome.
    this.timeout(30_000);

    for (const template of MODEL_TEMPLATES) {
      const source = renderTemplate(template, 'MyNet');
      const summary = execFileSync(
        PYTHON,
        [
          '-c',
          'import ast,sys;t=ast.parse(sys.stdin.read());' +
            'print(",".join(n.name for n in t.body if isinstance(n, ast.ClassDef)))',
        ],
        { input: source, encoding: 'utf8' },
      ).trim();
      assert.equal(summary, 'MyNet', `${template.id} should define exactly one class`);
    }
  });
});

interface ScaffoldSpy {
  host: ScaffoldHost;
  written: Array<{ fsPath: string; contents: string }>;
  opened: string[];
  warnings: string[];
}

function scaffoldHost(options: { root?: string; answer?: string; existing?: string[] } = {}): ScaffoldSpy {
  const written: Array<{ fsPath: string; contents: string }> = [];
  const opened: string[] = [];
  const warnings: string[] = [];
  return {
    written,
    opened,
    warnings,
    host: {
      workspaceRoot: () => options.root,
      ask: async (_prompt, value) => (options.answer === undefined ? value : options.answer || undefined),
      exists: async (fsPath) => (options.existing ?? []).some((name) => fsPath.endsWith(name)),
      write: async (fsPath, contents) => {
        written.push({ fsPath, contents });
      },
      open: async (fsPath) => {
        opened.push(fsPath);
      },
      warn: async (message) => {
        warnings.push(message);
      },
    },
  };
}

suite('scaffoldModel', () => {
  const template = MODEL_TEMPLATES[0];

  test('writes models/<name>.py and opens it', async () => {
    const spy = scaffoldHost({ root: '/ws', answer: 'my_net' });

    const result = await scaffoldModel(spy.host, template);

    assert.deepEqual(result, { ok: true, fsPath: '/ws/models/my_net.py', className: 'MyNet' });
    assert.equal(spy.written.length, 1);
    assert.ok(spy.written[0].contents.includes('class MyNet(nn.Module):'));
    assert.deepEqual(spy.opened, ['/ws/models/my_net.py']);
  });

  test('never overwrites an existing file', async () => {
    // The .py is the source of truth for the whole product; a template that can silently
    // replace one is the single bug this feature must not have.
    const spy = scaffoldHost({ root: '/ws', answer: 'mlp', existing: ['models/mlp.py'] });

    assert.deepEqual(await scaffoldModel(spy.host, template), { ok: false, reason: 'exists' });
    assert.deepEqual(spy.written, []);
    assert.equal(spy.warnings.length, 1);
  });

  test('a typed ".py" suffix does not become models/name.py.py', async () => {
    const spy = scaffoldHost({ root: '/ws', answer: 'mlp.py' });

    const result = await scaffoldModel(spy.host, template);
    assert.deepEqual(result, { ok: true, fsPath: '/ws/models/mlp.py', className: 'Mlp' });
  });

  test('refuses with an explanation when no folder is open', async () => {
    const spy = scaffoldHost({ answer: 'mlp' });

    assert.deepEqual(await scaffoldModel(spy.host, template), { ok: false, reason: 'no-workspace' });
    assert.deepEqual(spy.written, []);
    assert.equal(spy.warnings.length, 1);
  });

  test('a cancelled prompt writes nothing', async () => {
    const spy = scaffoldHost({ root: '/ws', answer: '' });

    assert.deepEqual(await scaffoldModel(spy.host, template), { ok: false, reason: 'cancelled' });
    assert.deepEqual(spy.written, []);
  });
});

suite('ModelsTree empty state', () => {
  setup(() => reset());
  teardown(() => reset());

  /** An index that has never answered for this root, which is what "the backend is
   *  not running" looks like from here. */
  const unscanned = {
    classesFor: () => undefined,
    refresh: async () => undefined,
    onDidChangeCodeLenses: () => ({ dispose: () => undefined }),
  } as never;

  const scanned = (classes: unknown[]) =>
    ({
      classesFor: () => classes,
      refresh: async () => undefined,
      onDidChangeCodeLenses: () => ({ dispose: () => undefined }),
    } as never);

  test('an unscanned folder lists nothing at all, not a button', () => {
    // This replaces the "offers a row you can click" case. That row existed because
    // the user had to start the backend by hand and the message telling them to was
    // unclickable outside the fork. task.md P0 removes the cause: the backend starts
    // itself, so the tree shows the "Indexing…" message and then fills in. A play
    // button here would be an action with nothing left to do.
    state.workspaceFolders = [{ uri: { fsPath: '/work' } }];
    const tree = new ModelsTree(unscanned);

    assert.deepEqual(tree.getChildren(), []);
  });

  test('a folder that HAS been scanned lists its classes, not the button', () => {
    // The other half: once the scan has answered, the button must be gone even when
    // the answer was empty, or "no models here" becomes permanently unreachable.
    state.workspaceFolders = [{ uri: { fsPath: '/work' } }];

    assert.deepEqual(new ModelsTree(scanned([])).getChildren(), []);
    const tree = new ModelsTree(
      scanned([{ name: 'Net', file: 'models/resid.py', line: 14, bases: ['nn.Module'], ctor_required: 0 }]),
    );
    const one = tree.getChildren();
    assert.equal(one.length, 1);
    assert.equal(tree.getTreeItem(one[0]).label, 'models');
    const file = tree.getChildren(one[0])[0];
    assert.equal(tree.getTreeItem(file).label, 'resid.py');
    assert.equal(tree.getTreeItem(tree.getChildren(file)[0]).label, 'Net');
  });

  test('no folder open offers no button either', () => {
    // There is nothing to scan, so a button that starts a scan would be a lie about
    // what pressing it achieves.
    state.workspaceFolders = undefined;
    assert.deepEqual(new ModelsTree(unscanned).getChildren(), []);
  });
});

suite('ModelsTree source hierarchy', () => {
  setup(reset);
  teardown(reset);

  const model = (file: string, name: string, line = 1): WorkspaceClass => ({
    file, name, line, bases: ['nn.Module'], ctor_required: 0,
  });
  const treeFor = (classes: Map<string, WorkspaceClass[]>) => new ModelsTree({
    classesFor: (root) => classes.get(root),
    refresh: async () => undefined,
    onDidChangeClasses: new vscode.EventEmitter<void>().event,
  });
  const labels = (tree: ModelsTree, parent?: ModelsNode) =>
    tree.getChildren(parent).map((node) => tree.getTreeItem(node).label);
  const leaves = (tree: ModelsTree, parent?: ModelsNode): ModelsNode[] =>
    tree.getChildren(parent).flatMap((node) =>
      tree.getTreeItem(node).command ? [node] : leaves(tree, node));

  test('folders and files contain every class, with folders first and sorted siblings', () => {
    state.workspaceFolders = [{ uri: { fsPath: '/work' } }];
    const tree = treeFor(new Map([['/work', [
      model('models/heads/depth.py', 'ZHead', 20),
      model('train.py', 'RootNet'),
      model('models/stem.py', 'Stem'),
      model('models/heads/depth.py', 'AHead', 4),
      model('baseline/net.py', 'Baseline'),
    ]]]));
    assert.deepEqual(labels(tree), ['baseline', 'models', 'train.py']);
    const models = tree.getChildren()[1];
    assert.deepEqual(labels(tree, models), ['heads', 'stem.py']);
    const heads = tree.getChildren(models)[0];
    const file = tree.getChildren(heads)[0];
    assert.deepEqual(labels(tree, heads), ['depth.py']);
    assert.deepEqual(labels(tree, file), ['AHead', 'ZHead']);
    for (const group of [models, heads, file]) {
      const item = tree.getTreeItem(group);
      assert.equal(item.collapsibleState, vscode.TreeItemCollapsibleState.Collapsed);
      assert.equal(item.command, undefined, 'expanding a group must not capture a model');
    }
    assert.equal(tree.getTreeItem(models).description, '3 models');
    assert.equal(leaves(tree).length, 5);
    const head = tree.getChildren(file)[0];
    assert.deepEqual(tree.getChildren(head), []);
    assert.equal(tree.getTreeItem(head).description, 'line 4');
    assert.equal(tree.getTreeItem(head).contextValue, 'tensorviz.model');
    assert.deepEqual(tree.getTreeItem(head).command, {
      title: 'Open as Graph', command: 'tensorviz.openAsGraph',
      arguments: [{ uri: '/work/models/heads/depth.py', cls: 'AHead' }],
    });
  });

  test('identical class names and paths in different workspace folders open their own files', () => {
    state.workspaceFolders = [
      { uri: { fsPath: '/one/project' } }, { uri: { fsPath: '/two/project' } },
    ];
    const tree = treeFor(new Map([
      ['/one/project', [model('models/net.py', 'Net')]],
      ['/two/project', [model('models/net.py', 'Net')]],
    ]));
    const roots = tree.getChildren();
    assert.equal(roots.length, 2);
    assert.notEqual(tree.getTreeItem(roots[0]).id, tree.getTreeItem(roots[1]).id);
    assert.deepEqual(leaves(tree).map((node) => tree.getTreeItem(node).command?.arguments), [
      [{ uri: '/one/project/models/net.py', cls: 'Net' }],
      [{ uri: '/two/project/models/net.py', cls: 'Net' }],
    ]);
    assert.equal(new Set(leaves(tree).map((node) => tree.getTreeItem(node).id)).size, 2);
  });

  test('refresh keeps group identities while additions, moves and removals follow the index', () => {
    state.workspaceFolders = [{ uri: { fsPath: '/work' } }];
    const classes = new Map([['/work', [model('models/net.py', 'Net')]]]);
    const tree = treeFor(classes);
    const folderId = tree.getTreeItem(tree.getChildren()[0]).id;
    const fileId = tree.getTreeItem(tree.getChildren(tree.getChildren()[0])[0]).id;
    let changes = 0;
    tree.onDidChangeTreeData(() => changes++);
    classes.set('/work', [model('models/net.py', 'Net', 8), model('models/block.py', 'Block')]);
    tree.refresh();
    const folder = tree.getChildren()[0];
    assert.equal(changes, 1);
    assert.ok(folderId);
    assert.equal(tree.getTreeItem(folder).id, folderId);
    assert.deepEqual(labels(tree, folder), ['block.py', 'net.py']);
    assert.equal(tree.getTreeItem(tree.getChildren(folder)[1]).id, fileId);
    classes.set('/work', [model('other/moved.py', 'Net')]);
    tree.refresh();
    assert.deepEqual(labels(tree), ['other']);
    assert.equal(leaves(tree).length, 1);
    classes.set('/work', []);
    tree.refresh();
    assert.deepEqual(tree.getChildren(), []);
  });

  test('a large model index stays compact until its folders and files are expanded', () => {
    state.workspaceFolders = [{ uri: { fsPath: '/work' } }];
    const entries = Array.from({ length: 1500 }, (_, i) =>
      model(`models/area_${i % 10}/net_${i}.py`, `Net${i}`));
    const tree = treeFor(new Map([['/work', entries]]));
    assert.deepEqual(labels(tree), ['models']);
    const models = tree.getChildren()[0];
    assert.equal(tree.getChildren(models).length, 10);
    assert.equal(tree.getTreeItem(models).description, '1500 models');
    assert.equal(leaves(tree).length, entries.length, 'grouping must not hide or truncate models');
    assert.deepEqual(entries.map((entry) => entry.name),
      Array.from({ length: 1500 }, (_, i) => `Net${i}`), 'the shared index must not be reordered');
  });
});

suite('scaffoldModel: a name already asked for is not asked for twice', () => {
  const template = { id: 't', label: 'T', description: '', fileStem: 'suggested', body: 'class {cls}: pass\n' };

  test('uses the given name and never prompts', async () => {
    // The catalogue path settles the file name first, because the backend writes the
    // class name into the source it generates. Without this the user answered the same
    // question twice in a row.
    const asked: string[] = [];
    const written: Array<[string, string]> = [];
    const result = await scaffoldModel(
      {
        workspaceRoot: () => '/ws',
        ask: async (prompt: string) => { asked.push(prompt); return 'from-the-prompt'; },
        exists: async () => false,
        write: async (p: string, c: string) => { written.push([p, c]); },
        open: async () => {},
        warn: async () => {},
      },
      template,
      'already-chosen',
    );
    assert.equal(asked.length, 0);
    assert.ok(result.ok);
    assert.ok(written[0][0].endsWith('already-chosen.py'), written[0][0]);
  });

  test('still prompts when the caller has no name', async () => {
    // The two built-in starters go through the original path, and must keep asking.
    const asked: string[] = [];
    await scaffoldModel(
      {
        workspaceRoot: () => '/ws',
        ask: async (prompt: string) => { asked.push(prompt); return 'typed'; },
        exists: async () => false,
        write: async () => {},
        open: async () => {},
        warn: async () => {},
      },
      template,
    );
    assert.deepEqual(asked, ['File name for the new model']);
  });
});

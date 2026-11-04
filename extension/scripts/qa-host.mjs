#!/usr/bin/env node
/**
 * Launch a disposable, remote-controllable VS Code host with this extension loaded.
 *
 * `docs/platform-testing.md` calls a portable end-to-end host launcher follow-up work:
 * the drivers (`qa-local-browser.cjs`, `check-installed-workspace.cjs`) expect a
 * `TV_QA_ROOT` laid out just so, and nothing built it. This does, so the
 * installed-page Playwright specs — and a human, and an agent — can drive the real
 * extension in a real editor instead of approximating it in jsdom.
 *
 * What it guarantees, because the drivers assert all three:
 *   - the workspace is a COPY under TV_QA_ROOT/fixture, never anyone's real tree;
 *   - `evidence/fixture-before.json` is captured BEFORE VS Code starts, so the
 *     purity check can prove activation wrote nothing into the workspace;
 *   - the editor profile is disposable (`--user-data-dir`), so a run cannot inherit
 *     or corrupt a developer's real profile.
 *
 * Usage:
 *   node scripts/qa-host.mjs                     # boot and idle, printing the handoff
 *   node scripts/qa-host.mjs --root <dir>        # reuse a specific TV_QA_ROOT
 *   node scripts/qa-host.mjs --driver <file>     # a different extensionTestsPath
 *
 * Drive it by writing TV_QA_ROOT/control/command.json (see qa-local-browser.cjs for
 * the verbs) and reading the result file it names back. `{"type":"finish"}` ends it.
 *
 * TENSORVIZ_TEST_PYTHON must name an interpreter with the backend's dependencies; the
 * host starts a real backend and there is no point pretending otherwise.
 */
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runTests, resolveCliArgsFromVSCodeExecutablePath } from '@vscode/test-electron';

const here = dirname(fileURLToPath(import.meta.url));
const extensionDir = resolve(here, '..');
const repoRoot = resolve(extensionDir, '..');

function arg(name, fallback) {
  const at = process.argv.indexOf(name);
  return at !== -1 && process.argv[at + 1] ? process.argv[at + 1] : fallback;
}

const root = resolve(arg('--root', join(repoRoot, '.qa-host')));
const driver = resolve(arg('--driver', join(here, 'qa-local-browser.cjs')));

/** The workspace the host opens. Small on purpose: every file here is read by the
 *  purity check, and the drivers name `models/net.py` and
 *  `complex_models/network.py` specifically. */
const FIXTURE = {
  'models/__init__.py': '',
  'models/net.py': `import torch.nn as nn


# tensorviz: input=2,8
class Net(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.up = nn.Linear(width, width * 2)
        self.act = nn.ReLU()
        self.down = nn.Linear(width * 2, width)

    def forward(self, x):
        return self.down(self.act(self.up(x)))
`,
  'complex_models/__init__.py': '',
  // The cross-file case, and the only one in this fixture that tells the two
  // projection tiers apart: `Block` lives in its OWN file, so a single-buffer
  // parse can never see inside it and the graph differs by exactly its layers.
  // `Network` below does not work for that — its `Block` is built inside an
  // `nn.ModuleList` comprehension, which stays one opaque card on both tiers.
  'complex_models/blocks.py': `import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
`,
  'complex_models/layered.py': `import torch
import torch.nn as nn

from complex_models.blocks import Block


# tensorviz: input=1,3,32,32
class Layered(nn.Module):
    def __init__(self, dim=16, classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.block = Block(dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.pool(self.block(self.stem(x)))
        return self.head(torch.flatten(x, 1))
`,
  'complex_models/network.py': `import torch
import torch.nn as nn


# tensorviz: input=1,3,32,32
class Network(nn.Module):
    """Wide enough to exercise grouping, repeats and a parameter source."""

    def __init__(self, dim=16, depth=3, classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        # A ModuleList of LEAF layers: the tracer folds these onto one card with
        # _repeat_count, which is the case a repeat frame has to wrap.
        self.stack = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.act(self.norm(self.stem(x)))
        for block in self.blocks:
            x = block(x)
        x = x + self.bias
        x = self.pool(x)
        x = torch.flatten(x, 1)
        for layer in self.stack:
            x = layer(x)
        x = x.reshape(x.shape[0], -1, 1, 1)
        return self.head(torch.flatten(x, 1))


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x))) + x
`,
};

function manifest(dir) {
  const out = {};
  const walk = (current) => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const file = join(current, entry.name);
      if (entry.isDirectory()) walk(file);
      else out[relative(dir, file)] = createHash('sha256').update(readFileSync(file)).digest('hex');
    }
  };
  walk(dir);
  return out;
}

function layout() {
  // A stale control/ or a stale profile is how a "fresh" run inherits the last one's
  // answer, so the root is rebuilt rather than topped up.
  rmSync(root, { recursive: true, force: true });
  mkdirSync(root, { recursive: true, mode: 0o700 });
  for (const part of ['fixture', 'control', 'evidence', 'profile/logs', 'extensions']) {
    mkdirSync(join(root, part), { recursive: true });
  }
  for (const [file, text] of Object.entries(process.argv.includes('--cold-runtime') ? {} : FIXTURE)) {
    const target = join(root, 'fixture', file);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, text, 'utf8');
  }
  const graphExamples = join(repoRoot, 'examples', 'graph_ui');
  if (!process.argv.includes('--cold-runtime') && existsSync(graphExamples)) cpSync(graphExamples, join(root, 'fixture', 'graph_ui'), { recursive: true });
  // BEFORE the editor starts: that ordering is the whole point of the purity check.
  writeFileSync(
    join(root, 'evidence/fixture-before.json'),
    JSON.stringify(manifest(join(root, 'fixture')), null, 2),
    'utf8',
  );
}

async function main() {
  const vsix = arg('--vsix', '');
  const executable = arg('--vscode', '');
  if (vsix && (!executable || !existsSync(resolve(vsix)))) {
    throw new Error('--vsix requires an existing package and --vscode executable');
  }
  if (!vsix && !existsSync(join(extensionDir, 'out/extension.js'))) {
    console.error('qa-host: extension/out is missing — run `npm run compile` first.');
    process.exit(1);
  }
  if (!vsix && !existsSync(join(extensionDir, 'media/app/index.html'))) {
    console.error('qa-host: extension/media/app is missing — run `npm run build:webview` first.');
    process.exit(1);
  }
  if (!process.env.TENSORVIZ_TEST_PYTHON) {
    console.error('qa-host: set TENSORVIZ_TEST_PYTHON to an interpreter with the backend deps.');
    process.exit(1);
  }

  layout();
  let launchExtension = extensionDir;
  if (vsix) {
    const [cli, ...cliArgs] = resolveCliArgsFromVSCodeExecutablePath(resolve(executable));
    execFileSync(cli, [...cliArgs, '--user-data-dir', join(root, 'profile'), '--extensions-dir', join(root, 'extensions'),
      '--install-extension', resolve(vsix), '--force'], { stdio: 'inherit' });
    const installed = readdirSync(join(root, 'extensions')).filter(name => name.startsWith('tensorviz.tensorviz-vscode-'));
    if (installed.length !== 1) throw new Error('Expected one installed tensorViz VSIX in the disposable profile');
    launchExtension = join(root, 'extensions', installed[0]);
    writeFileSync(join(root, 'evidence', 'package.json'), JSON.stringify({
      sha256: createHash('sha256').update(readFileSync(resolve(vsix))).digest('hex'),
      extensionPath: launchExtension,
    }, null, 2));
  }
  console.log(`qa-host: root      ${root}`);
  console.log(`qa-host: workspace ${join(root, 'fixture')}`);
  console.log(`qa-host: control   ${join(root, 'control')}`);
  console.log(`qa-host: opens     ${join(root, 'control/opened-links.txt')}`);
  console.log(`qa-host: driver    ${driver}`);

  try {
    await runTests({
      // Use an installed/cached editor without a network version lookup.
      ...(arg('--vscode', '') ? { vscodeExecutablePath: resolve(arg('--vscode', '')) } : {}),
      extensionDevelopmentPath: launchExtension,
      extensionTestsPath: driver,
      // A disposable profile, and nothing else loaded: the run must mean the same
      // thing on a machine with fifty extensions as on a bare one.
      launchArgs: [
        join(root, 'fixture'),
        '--user-data-dir', join(root, 'profile'),
        '--extensions-dir', join(root, 'extensions'),
        '--disable-workspace-trust',
        '--skip-welcome',
        '--skip-release-notes',
        ...(arg('--debug-port', '') ? [`--remote-debugging-port=${Number(arg('--debug-port', ''))}`, '--remote-debugging-address=127.0.0.1'] : []),
      ],
      extensionTestsEnv: {
        TV_QA_ROOT: root,
        TV_QA_DEBUG_PORT: arg('--debug-port', ''),
        // Every page the extension would hand to a browser is appended here instead,
        // one URL per line. Without it a driver cannot tell "opened a tab" from
        // "silently did nothing", which is exactly the bug class this host exists to
        // catch — and it keeps a QA run from throwing real browser windows at whoever
        // is watching.
        TENSORVIZ_LOCAL_BROWSER_LINK_FILE: join(root, 'control/opened-links.txt'),
        TENSORVIZ_TEST_PYTHON: process.env.TENSORVIZ_TEST_PYTHON,
        TENSORVIZ_TEST_PRESEED_MANAGED: process.argv.includes('--cold-runtime') ? '0' : process.env.TENSORVIZ_TEST_PRESEED_MANAGED ?? '1',
        ...(process.argv.includes('--cold-runtime') ? { BACKEND_PYTHON: '', TENSORVIZ_TEST_RUNTIME_ARCHIVE: '', TENSORVIZ_TEST_RUNTIME_MANIFEST: '' } : {}),
        PLANNER_PROVIDER: process.env.PLANNER_PROVIDER ?? 'fixture',
        KMP_DUPLICATE_LIB_OK: 'TRUE',
      },
    });
    console.log('qa-host: driver finished');
  } catch (error) {
    console.error(`qa-host: driver failed — ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

await main();

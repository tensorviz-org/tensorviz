import * as assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { promisify } from 'node:util';

suite('development build', () => {
  test('a fresh output directory builds the webview before backend staging validates it', async () => {
    const manifest = JSON.parse(await readFile(path.resolve(__dirname, '../../../package.json'), 'utf8'));
    const root = await mkdtemp(path.join(os.tmpdir(), 'tensorviz-dev-build-'));
    try {
      await writeFile(path.join(root, 'step.cjs'), `
const fs = require('node:fs');
const step = process.argv[2];
if (step === 'webview') { fs.mkdirSync('media/app', { recursive: true }); fs.writeFileSync('media/app/index.html', 'canvas'); }
if (step === 'backend') { fs.readFileSync('media/app/index.html'); fs.writeFileSync('backend-ready', ''); }
if (step === 'compile') fs.writeFileSync('host-ready', '');
if (step === 'check') { fs.readFileSync('backend-ready'); fs.readFileSync('host-ready'); fs.readFileSync('media/app/index.html'); }
`);
      await writeFile(path.join(root, 'package.json'), JSON.stringify({ private: true, scripts: {
        'dev:build': manifest.scripts['dev:build'], 'build:webview': 'node step.cjs webview',
        'stage:backend': 'node step.cjs backend', compile: 'node step.cjs compile', 'check:staged': 'node step.cjs check',
      } }));
      await promisify(execFile)(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'dev:build'], { cwd: root });
      assert.equal(await readFile(path.join(root, 'media/app/index.html'), 'utf8'), 'canvas');
    } finally { await rm(root, { recursive: true, force: true }); }
  });
});

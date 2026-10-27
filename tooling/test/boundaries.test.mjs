import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { contractViolations, hostViolations, reachableSources } from '../check-boundaries.mjs';

const directory = path.resolve('packages/contracts/src');
const check = source => contractViolations(path.join(directory, 'protocol.ts'), source, directory);

test('contracts reject host dependencies even when erased from the runtime bundle', () => {
  assert.deepEqual(check("import type { Uri } from 'vscode';"), ['vscode']);
  assert.deepEqual(check("type Uri = import('vscode').Uri;"), ['vscode']);
  assert.deepEqual(check("export * from '../../../extension/src/protocol';"), ['../../../extension/src/protocol']);
});

test('contracts cannot acquire platform or dynamically loaded application code', () => {
  assert.deepEqual(check("import fs from 'node:fs';"), ['node:fs']);
  assert.deepEqual(check("const host = require('../../../plugin/src/host');"), ['../../../plugin/src/host']);
  assert.deepEqual(check("const ui = import('../../../canvas/src/App');"), ['../../../canvas/src/App']);
});

test('internal contracts remain usable and prose does not count as an import', () => {
  assert.deepEqual(check("import type { Model } from './protocol'; export * from './captureReader'; // import 'vscode'"), []);
});

test('shared execution can use Node and declared libraries without acquiring editor types', () => {
  const host = path.resolve('packages/graph-host/src');
  const checkHost = source => hostViolations(path.join(host, 'worker.ts'), source, host, ['@tensorviz/contracts', 'zod']);
  assert.deepEqual(checkHost("import { spawn } from 'node:child_process'; import { z } from 'zod'; import type { RunSpec } from '@tensorviz/contracts/protocol';"), []);
  assert.deepEqual(checkHost("import type { Uri } from 'vscode';"), ['vscode']);
  assert.deepEqual(checkHost("export * from '../../../extension/src/worker';"), ['../../../extension/src/worker']);
  assert.deepEqual(checkHost("import React from 'react';"), ['react']);
});

test('a local entry cannot conceal hosted dependencies behind helpers, type imports or cycles', () => {
  const directory = path.resolve('test-frontend');
  const files = new Map(Object.entries({
    'local.tsx': "import { helper } from './helper';",
    'helper.ts': "import type { User } from './hosted/account'; import './local';",
    'hosted/account.ts': "export type User = string;",
  }).map(([name, source]) => [path.join(directory, name), source]));
  assert.deepEqual(reachableSources([path.join(directory, 'local.tsx')], filename => files.get(filename),
    filename => files.has(filename)), [...files.keys()]);
});

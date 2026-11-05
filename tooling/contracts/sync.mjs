#!/usr/bin/env node
/** Checked browser copies preserve its independent compiler without shared-code forks. */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const copies = [
  ['packages/contracts/src/protocol.ts', 'canvas/src/ide/protocol.generated.ts'],
  ['packages/contracts/src/validationContracts.generated.ts', 'canvas/src/ide/validationContracts.generated.ts'],
  ...['captureReader', 'captureCorrespondence', 'captureComparison', 'initializerArguments', 'graphHelpRequest', 'sourceShapeCall', 'graphSetupWorkflow', 'nodeExplanation'].map(name =>
    [`packages/contracts/src/${name}.ts`, `canvas/src/ide/${name}.generated.ts`]),
  ['packages/graph-host/src/viewTransport.ts', 'canvas/src/ide/localBrowserViewTransport.ts'],
  ['packages/graph-host/src/viewWire.ts', 'canvas/src/ide/localBrowserWire.ts'],
];

const check = process.argv.includes('--check');
const stale = [];
for (const [source, target] of copies) {
  const header = source.endsWith('validationContracts.generated.ts') ? ''
    : `/* GENERATED from ${source}; npm run protocol:sync. */\n`;
  const content = readFileSync(resolve(root, source), 'utf8')
    .replaceAll("'./semanticContracts.generated'", "'../types/semanticContracts.generated'")
    .replaceAll("'./protocol'", "'./protocol.generated'")
    .replaceAll("'@tensorviz/contracts/protocol'", "'./protocol.generated'")
    .replaceAll("'./captureCorrespondence'", "'./captureCorrespondence.generated'")
    .replaceAll("'./nodeExplanation'", "'./nodeExplanation.generated'")
    .replaceAll("'./viewWire'", "'./localBrowserWire'");
  const generated = header + content;
  let current;
  try { current = readFileSync(resolve(root, target), 'utf8'); } catch { /* First generation. */ }
  if (current === generated) continue;
  if (check) { stale.push(target); continue; }
  mkdirSync(dirname(resolve(root, target)), { recursive: true });
  writeFileSync(resolve(root, target), generated);
  console.log(`protocol: wrote ${target}`);
}
if (stale.length) {
  throw new Error(`Outdated browser copies: ${stale.join(', ')}. Run npm run protocol:sync from the repository root.`);
}
if (check) console.log('protocol: all browser contracts and transport copies are current');

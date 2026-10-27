#!/usr/bin/env node
/** Read-only verification of Python corpus documents + portable payloads through
 * the production file importer/exporter. Optional outputs are local QA files. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const ts = require('typescript');
const root = path.resolve(__dirname, '..');
require.extensions['.ts'] = (module, filename) => {
  if (!filename.startsWith(path.join(root, 'src') + path.sep)) throw new Error('Only local frontend sources may be transpiled.');
  const output = ts.transpileModule(fs.readFileSync(filename, 'utf8'), { compilerOptions: {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true,
  } });
  module._compile(output.outputText, filename);
};
const { createGraphFile, parseGraphFile } = require('../src/services/graphFile.ts');
const { graphDocumentV2ToModelGraph, modelGraphToDocumentV2 } = require('../src/services/graphDocumentAdapter.ts');
const { initialGraphFileView } = require('../src/viewer/graphFileView.ts');
const { reconcileModuleView } = require('../src/services/canvasLayout.ts');
const { requirePortableCapture } = require('../src/services/portableCapture.ts');
const directory = process.argv[2], output = process.argv[3];
if (!directory || !output) throw new Error('Usage: node scripts/check-portable-corpus.cjs INPUT_DIRECTORY OUTPUT_REPORT.json');
const files = fs.readdirSync(directory).filter(name => name.endsWith('.portable.json')).sort();
assert.ok(files.length, 'No corpus pairs were found.');
const results = [];
for (const filename of files) {
  const name = filename.slice(0, -'.portable.json'.length);
  const capture = requirePortableCapture(JSON.parse(fs.readFileSync(path.join(directory, filename), 'utf8')));
  const document = JSON.parse(fs.readFileSync(path.join(directory, name + '.json'), 'utf8'));
  const start = performance.now();
  const converted = graphDocumentV2ToModelGraph(document);
  const canvas = reconcileModuleView(converted.nodes, converted.edges, { relayout: true });
  const file = createGraphFile(canvas, { name: document.semantic.name, status: 'traced', capture });
  const serialized = JSON.stringify(file);
  const restored = parseGraphFile(serialized);
  assert.deepEqual(restored.capture, capture);
  assert.deepEqual(restored.document.semantic.edges, document.semantic.edges);
  const portMap = doc => new Map(doc.semantic.nodes.map(node => [node.node_id, node.metadata.capturePorts]));
  assert.deepEqual(portMap(restored.document), portMap(document));
  const graph = initialGraphFileView(restored);
  assert.deepEqual(modelGraphToDocumentV2(graph, { name: document.semantic.name, revision: 0 }).semantic.edges, document.semantic.edges);
  fs.writeFileSync(path.join(directory, name + '.tensorviz.json'), serialized);
  results.push({ case: name, matched: true, nodes: graph.nodes.length, edges: document.semantic.edges.length,
    status: capture.capture.status, bytes: Buffer.byteLength(serialized), wallMs: Math.round(performance.now() - start) });
}
fs.writeFileSync(output, JSON.stringify({ version: 1, scope: 'Production portable file export/import and initial viewer projection; no installed host or browser certification.', results }, null, 2) + '\n');
console.log(JSON.stringify(results));

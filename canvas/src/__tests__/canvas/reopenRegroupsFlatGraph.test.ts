/**
 * A saved graph must be re-grouped on reopen.
 *
 * F5-TTS ships FLAT: `/api/templates/f5_tts/instantiate` returns 290 root nodes and ZERO
 * modules, so every block card on the canvas is produced by `autoModularize`, not by the
 * template. That grouping is applied in memory when the template lands.
 *
 * The reopen path used to skip grouping whenever `currentRevision > 0` — which is every
 * graph after its first autosave. Measured on a flat-saved F5-TTS at revision 1:
 *
 *     without the fix   290 layers, 0 "Expand Block"   (one unreadable row)
 *     with the fix      294 layers, 4 "Expand Block"
 *
 * Grouping does persist when it gets saved (a grouped F5-TTS round-trips as 294 nodes
 * with its 4 wrappers), so the guard protected nothing — but a graph stored flat, because
 * the autosave beat the in-memory grouping, could never recover.
 *
 * ModelBuilder cannot be mounted under the CRA jest transform (importing tldraw breaks
 * jest — see qa_builder_ux.test.ts), so this pins the source contract, the technique the
 * repo already uses in ui2InteractionAudit.test.ts.
 */
import fs from 'fs';
import path from 'path';

const SOURCE = fs.readFileSync(
  path.join(__dirname, '../..', 'components', 'ModelBuilder.tsx'),
  'utf8',
);

describe('reopening a saved graph re-derives its module structure', () => {
  it('does not gate grouping on the revision number', () => {
    // The exact shape of the defect: a revision test choosing between the grouping
    // helper and the plain one.
    const gated = /if\s*\(\s*initialGraph\.currentRevision\s*>\s*0\s*\)[\s\S]{0,400}?applyLogicalAutoGroupedRef/.test(
      SOURCE,
    );
    expect(gated).toBe(false);
  });

  it('routes the reopen through the grouping helper', () => {
    expect(SOURCE).toMatch(
      /applyLogicalAutoGroupedRef\.current\?\.\(\s*logicalNodes,\s*logicalEdges/,
    );
  });

  it('still passes the graph metadata, so declared blocks win over inferred ones', () => {
    const call = SOURCE.match(
      /applyLogicalAutoGroupedRef\.current\?\.\(\s*logicalNodes,\s*logicalEdges,([\s\S]{0,120}?)\)/,
    );
    expect(call).not.toBeNull();
    expect(call![1]).toContain('metadata');
  });

  it('keeps re-running safe by leaving an already-modular graph alone', () => {
    // The guarantee that makes unconditional re-grouping correct: autoModularize bails
    // when wrappers or parented children already exist. If that early return goes, this
    // change would re-wrap an already-grouped graph on every open.
    const auto = fs.readFileSync(
      path.join(__dirname, '../..', 'services', 'autoModularize.ts'),
      'utf8',
    );
    expect(auto).toMatch(
      /if\s*\(\s*nodes\.some\(\(n\)\s*=>\s*n\.type\s*===\s*'moduleNode'\s*\|\|\s*n\.data\?\.parentId\)\)\s*return\s+noop\(\)/,
    );
  });
});

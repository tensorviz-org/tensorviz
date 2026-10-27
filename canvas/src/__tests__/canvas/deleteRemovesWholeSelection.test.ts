/**
 * Delete and Backspace must remove the WHOLE selection.
 *
 * The key handler called removeNode(selectedNodeId) — a single id — so shift-selecting
 * two cards and pressing Delete removed exactly one of them (measured 5 layers -> 4,
 * not 3), and Cmd+A then Delete removed a single node. The toolbar's trash button
 * already routed through deleteSelectedNodes, so the two disagreed on the same
 * selection. That helper also heals the edges around what it removes; removeNode does
 * not, which is how a delete could leave the graph wired through a node that was gone.
 *
 * ModelBuilder cannot be mounted under the CRA jest transform (importing tldraw breaks
 * jest — see qa_builder_ux.test.ts), so this pins the source contract, as
 * ui2InteractionAudit.test.ts does.
 */
import fs from 'fs';
import path from 'path';

const SOURCE = fs.readFileSync(
  path.join(__dirname, '../..', 'components', 'ModelBuilder.tsx'),
  'utf8',
);

const handler = (() => {
  const start = SOURCE.indexOf("if (event.key === 'Delete' || event.key === 'Backspace')");
  return start === -1 ? '' : SOURCE.slice(start, start + 700);
})();

describe('the Delete key removes every selected node', () => {
  it('binds Delete and Backspace to the same path the toolbar uses', () => {
    expect(handler).not.toBe('');
    expect(handler).toContain('deleteSelectedNodes()');
  });

  it('no longer deletes a single id', () => {
    // The exact defect: one id deleted regardless of how many were selected. Match a
    // real CALL, not the comment that records what the code used to do.
    const calls = handler.replace(/\/\/[^\n]*/g, '');
    expect(calls).not.toMatch(/removeNode\(selectedNodeId\)/);
  });

  it('still refuses to delete while the user is typing', () => {
    // Backspace in a parameter field must edit text, not delete the model.
    expect(handler).toMatch(/if\s*\(\s*!isTyping\s*\)/);
  });

  it('keeps the handler subscribed to the helper it now calls', () => {
    const deps = SOURCE.slice(SOURCE.indexOf(handler) + handler.length);
    const list = deps.slice(0, deps.indexOf(']);') + 3);
    expect(list).toContain('deleteSelectedNodes');
  });

  it('the shared helper deletes all selected nodes and heals their edges', () => {
    const helper = SOURCE.slice(
      SOURCE.indexOf('const deleteSelectedNodes = useCallback'),
      SOURCE.indexOf('const deleteSelectedNodes = useCallback') + 800,
    );
    expect(helper).toMatch(/filter\(\(node\) => node\.selected \|\| node\.data\?\.isSelected\)/);
    expect(helper).toContain('healEdgesAfterDeletion');
  });
});

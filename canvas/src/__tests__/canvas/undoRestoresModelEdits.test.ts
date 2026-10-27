/**
 * Undo must restore model edits, not just canvas moves.
 *
 * Undo delegated to editorRef.current.undo() — tldraw's history. But every model edit
 * (delete, add layer, parameter change) mutates React state through setNodes/setEdges,
 * which tldraw never observes, so its undo stack was empty for exactly the edits users
 * wanted back. Measured: select-all, trash, then five Undos restored nothing, and the
 * empty model then survived a reload. A second report had undo appear to restore a card
 * with no edges, so the snapshot has to carry nodes AND edges together.
 *
 * ModelBuilder cannot be mounted under the CRA jest transform (importing tldraw breaks
 * jest — see qa_builder_ux.test.ts), so this pins the source contract, as
 * deleteRemovesWholeSelection.test.ts does. Behaviour itself was measured in a browser:
 * 5 layers/4 edges -> delete -> 4/3 -> Cmd+Z -> 5/4, and select-all delete 5 -> 0 ->
 * Cmd+Z -> 5, still 5 after a reload.
 */
import fs from 'fs';
import path from 'path';

const SOURCE = fs.readFileSync(
  path.join(__dirname, '../..', 'components', 'ModelBuilder.tsx'),
  'utf8',
);

const slice = (from: string, len: number) => {
  const start = SOURCE.indexOf(from);
  return start === -1 ? '' : SOURCE.slice(start, start + len);
};

describe('undo restores model edits', () => {
  it('keeps a model-level history stack, not only the tldraw one', () => {
    expect(SOURCE).toContain('historyRef');
    expect(slice('const undoModelEdit = useCallback', 400)).not.toBe('');
    expect(slice('const redoModelEdit = useCallback', 400)).not.toBe('');
  });

  it('snapshots nodes and edges together, so an undone delete comes back wired', () => {
    // The DE2-02 defect: a restored card with no edges. One snapshot must carry both.
    expect(SOURCE).toMatch(/type GraphSnapshot = \{\s*nodes: ModelNode\[\];\s*edges: ModelEdge\[\]/);
    const undo = slice('const undoModelEdit = useCallback', 500);
    expect(undo).toContain('previous.nodes, previous.edges');
  });

  it('tries the model stack before falling back to tldraw', () => {
    // Order matters: tldraw would report success for a canvas move and swallow the edit.
    const dispatch = slice("if (command === 'undo')", 300);
    expect(dispatch).toMatch(/!undoModelEdit\(\)\s*&&\s*editorRef\.current/);
    const redo = slice("if (command === 'redo')", 300);
    expect(redo).toMatch(/!redoModelEdit\(\)\s*&&\s*editorRef\.current/);
  });

  it('binds Cmd/Ctrl+Z and Cmd/Ctrl+Shift+Z to the same path as the toolbar', () => {
    const key = slice("event.key.toLowerCase() === 'z'", 500);
    expect(key).not.toBe('');
    expect(key).toContain('undoModelEdit()');
    expect(key).toContain('redoModelEdit()');
    // Cmd+Z inside a parameter field is text undo, not a model undo.
    expect(SOURCE).toMatch(/!isTyping && \(event\.ctrlKey \|\| event\.metaKey\) && event\.key\.toLowerCase\(\) === 'z'/);
  });

  it('does not record the snapshot it is itself applying', () => {
    // Without this guard an undo re-enters the recorder and the stack never drains.
    const recorder = slice('const h = historyRef.current;', 400);
    expect(recorder).toMatch(/if \(h\.applying\)/);
    expect(slice('const undoModelEdit = useCallback', 500)).toContain('h.applying = true');
  });

  it('bounds the stack so long sessions do not grow without limit', () => {
    expect(SOURCE).toMatch(/MODEL_HISTORY_LIMIT = \d+/);
    expect(SOURCE).toMatch(/h\.past\.length > MODEL_HISTORY_LIMIT/);
  });

  it('subscribes the dispatcher to the callbacks it invokes', () => {
    // A stale closure here would undo to a snapshot from an earlier graph.
    const after = SOURCE.slice(SOURCE.indexOf("if (command === 'undo')"));
    const deps = after.slice(0, after.indexOf(']);') + 3);
    expect(deps).toContain('undoModelEdit');
    expect(deps).toContain('redoModelEdit');
  });
});

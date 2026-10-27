import fs from 'fs';
import path from 'path';

// ModelBuilder mounts tldraw, whose ESM bundle is outside CRA Jest's transform.
// Pin this orchestration at the shipped-source boundary, matching the existing
// builder UX regression suite.
const MODEL_BUILDER = path.resolve(__dirname, '../..', 'components', 'ModelBuilder.tsx');
const source = fs.readFileSync(MODEL_BUILDER, 'utf8');

function sliceBetween(start: string, end: string): string {
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex + start.length);
  expect(startIndex).toBeGreaterThan(-1);
  expect(endIndex).toBeGreaterThan(startIndex);
  return source.slice(startIndex, endIndex);
}

describe('ModelBuilder save identity authority', () => {
  it('autosaves with the live saved identity instead of stale initial props', () => {
    const autosave = sliceBetween('// Debounced autosave to the server,', "detail: {\n          graph:");
    expect(autosave).toContain('const identity = graphIdentityRef.current;');
    expect(autosave).toContain('identity.name,');
    expect(autosave).toContain('identity.description,');
    expect(autosave).not.toContain('initialGraph?.name');
    expect(autosave).not.toContain('initialGraph?.description');
  });

  it('manual save cancels the debounce and adopts its identity before enqueue', () => {
    const handler = sliceBetween(
      'const handleSave = useCallback(async (name: string, description?: string) => {',
      'const generateSuggestedName = useCallback'
    );
    const clearIndex = handler.indexOf('window.clearTimeout(autosaveTimerRef.current);');
    const adoptIndex = handler.indexOf('adoptGraphIdentity(name, description);');
    const enqueueIndex = handler.indexOf('queue.enqueue(request)');
    expect(clearIndex).toBeGreaterThan(-1);
    expect(adoptIndex).toBeGreaterThan(clearIndex);
    expect(enqueueIndex).toBeGreaterThan(adoptIndex);
  });

  it('forces recovered and empty saved graphs through the revision queue', () => {
    expect(source).toContain(
      "if (nodes.length === 0 && !saveQueueRef.current?.authority().graphId) return;"
    );
    expect(source).toContain('initialGraph.recoveredDraft ? null : graphSaveFingerprint(baselineRequest)');
    expect(source).not.toContain('if (saved === null) clearDraft(draftKey);');
  });

  it('journals the queue snapshot and never reapplies an intermediate identity', () => {
    // The write itself lives in writeCrashJournal, shared with the pre-debounce
    // draft; the queue callback must still hand it the queue's own snapshot.
    const callbacks = sliceBetween(
      'saveQueueRef.current = new RevisionedGraphSaveQueue(services.persistGraph, {',
      'const [undoCount,'
    );
    expect(callbacks).toContain('onPendingChanged: ({ graphId, revision, requests })');
    expect(callbacks).toContain('writeCrashJournalRef.current(graphId, revision, requests)');
    expect(callbacks).not.toContain('adoptGraphIdentity(');

    const writer = sliceBetween(
      'const writeCrashJournal = useCallback((',
      'const writeCrashJournalRef'
    );
    expect(writer).toContain('draftStore.replaceJournal(');
    expect(writer).toContain('graphSaveDocument(request, draftKey, revision)');
    expect(writer).not.toContain('adoptGraphIdentity(');
  });

  it('journals a pending edit before the autosave debounce fires', () => {
    // Guide 11 promises a local draft written BEFORE the server call. While the
    // journal was written only on enqueue, reloading inside the 2s debounce
    // silently reverted the edit, and the pill read green "Saved" throughout.
    const draft = sliceBetween(
      '// Journal the pending edit well before the 2s autosave fires',
      '// Debounced autosave to the server,'
    );
    expect(draft).toContain('writeCrashJournalRef.current(graphId, revision, [request])');
    expect(draft).toContain('setSaveState((current) =>');
    // Only for edits that genuinely still need saving, or the pill flickers.
    expect(draft).toContain('if (!queue.needsSave(request)) return;');
  });

  it('never repaints a blocked save state as "Saving…"', () => {
    // Two tabs on one graph: the stale tab's first edit gets a 409 and the pill
    // says so, but the queue then freezes that head and sends nothing more. This
    // effect used to set 'saving' unconditionally, so the NEXT edit replaced the
    // refusal with a "Saving…" that never resolved — the dropped edit went
    // silent. Only the queue may clear 'offline'/'conflict'.
    const draft = sliceBetween(
      '// Journal the pending edit well before the 2s autosave fires',
      '// Debounced autosave to the server,'
    );
    expect(draft).toContain(
      "current === 'conflict' || current === 'offline' ? current : 'saving'"
    );
  });

  it('retires callbacks when the editor session unmounts', () => {
    expect(source).toContain('queue?.activateCallbacks();');
    expect(source).toContain('return () => queue?.retireCallbacks();');
  });

  it('manual save is always an explicit user boundary', () => {
    const handler = sliceBetween(
      'const handleSave = useCallback(async (name: string, description?: string) => {',
      'const saveDialogGraph = useMemo'
    );
    expect(handler).toContain("buildSaveRequest(name, description, 'user')");
    expect(handler).not.toContain('consumeNextSaveAttribution');
  });

  it('replays recovered commit envelopes in their original order', () => {
    const hydration = sliceBetween(
      '// Load the initial graph',
      'useEffect(() => {\n    // Wait until the editor exists'
    );
    expect(hydration).toContain('initialGraph.recoveredCommits?.length');
    expect(hydration).toContain('const requests = pending.map((commit) => ({');
    expect(hydration).toContain('saveQueueRef.current?.enqueue(request)');
  });
});

import {
  createGraphDraftStore,
  localDevelopmentDraftStore,
} from '../../services/localDraft';
import { modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';

const graph = { nodes: [], edges: [] } as any;

function document(id: string, name: string, revision = 0) {
  return modelGraphToDocumentV2(graph, { graphId: id, name, revision });
}

beforeEach(() => {
  localStorage.clear();
});

it('isolates journals and clears by captured owner namespace', () => {
  const ownerA = createGraphDraftStore('user-a');
  const ownerB = createGraphDraftStore('user-b');
  ownerA.replaceJournal('same-id', 2, [{
    document: document('same-id', 'A', 2), author: 'agent', createKey: 'create-a',
  }]);

  expect(ownerA.readRevisioned('same-id')?.document.semantic.name).toBe('A');
  expect(ownerB.read('same-id')).toBeNull();

  ownerB.replaceJournal('same-id', 4, [{
    document: document('same-id', 'B', 4), author: 'user',
  }]);
  ownerB.clear('same-id');

  expect(ownerB.read('same-id')).toBeNull();
  expect(ownerA.readRevisioned('same-id')?.document.semantic.name).toBe('A');
});

it('never assigns an unscoped draft to an authenticated owner', () => {
  const oldDocument = document('g1', 'Unscoped', 3);
  localStorage.setItem('canvas-draft-v2-g1', JSON.stringify({
    document: oldDocument,
    baseRevision: 3,
  }));

  expect(createGraphDraftStore('user-a').read('g1')).toBeNull();
  expect(localDevelopmentDraftStore.read('g1')).toMatchObject({
    kind: 'revisioned',
    draft: { baseRevision: 3 },
  });
});

it('tags a V2 draft with no recorded base revision as unknown-base', () => {
  localStorage.setItem(
    'canvas-draft-v3:local-dev:g1',
    JSON.stringify({ document: document('g1', 'Unknown') })
  );

  expect(localDevelopmentDraftStore.read('g1')).toMatchObject({
    kind: 'legacy-unknown-base',
    graph: { nodes: [], edges: [] },
  });
});

it('preserves raw legacy drafts as unknown-base recovery data', () => {
  localStorage.setItem('canvas-draft-g1', JSON.stringify(graph));

  const result = localDevelopmentDraftStore.read('g1');

  expect(result).toMatchObject({ kind: 'legacy-unknown-base', graph });
  expect(localStorage.getItem('canvas-draft-g1')).not.toBeNull();
});

it('does not erase unknown-base recovery data when a scoped journal is cleared', () => {
  const rawLegacy = JSON.stringify(graph);
  const missingBaseV2 = JSON.stringify({ document: document('g2', 'Unknown') });
  localStorage.setItem('canvas-draft-g1', rawLegacy);
  localStorage.setItem('canvas-draft-v2-g2', missingBaseV2);

  localDevelopmentDraftStore.replaceJournal('g1', 4, [{
    document: document('g1', 'Current', 4), author: 'user',
  }]);
  localDevelopmentDraftStore.replaceJournal('g2', 4, [{
    document: document('g2', 'Current', 4), author: 'user',
  }]);
  localDevelopmentDraftStore.clear('g1');
  localDevelopmentDraftStore.clear('g2');

  expect(localStorage.getItem('canvas-draft-g1')).toBe(rawLegacy);
  expect(localStorage.getItem('canvas-draft-v2-g2')).toBe(missingBaseV2);
});

it('quarantines an unknown-base scoped payload before replacing its journal', () => {
  const sourceKey = 'canvas-draft-v3:local-dev:g1';
  const raw = JSON.stringify({ document: document('g1', 'Unknown') });
  localStorage.setItem(sourceKey, raw);

  localDevelopmentDraftStore.replaceJournal('g1', 6, [{
    document: document('g1', 'Current', 6), author: 'user',
  }]);
  localDevelopmentDraftStore.clear('g1');

  expect(localDevelopmentDraftStore.read('g1')).toMatchObject({
    kind: 'legacy-unknown-base',
    graph: { nodes: [], edges: [] },
  });
  const recoveryKey = Object.keys(localStorage)
    .find((key) => key.startsWith('canvas-recovery-v1:'));
  expect(recoveryKey).toBeDefined();
  expect(localStorage.getItem(recoveryKey!)).toBe(raw);
});

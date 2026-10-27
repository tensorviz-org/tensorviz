import { readVerification, projectStatus } from '../../services/verificationRecord';
import {
  recordVerification,
  readRecordedVerification,
  resolveVerification,
  forgetVerification,
} from '../../services/verificationStore';
import { GraphDocumentV2 } from '../../types/graphDocument';
import { SavedGraph } from '../../types';

/**
 * The redesign puts a model's health on its card and in the projects table. That
 * only works if four states stay distinguishable — and in particular if a pass
 * that predates the current revision is NOT reported as a pass. A green
 * "Verified" pill on a model edited after the last check is the exact "falsely
 * complete" failure this record exists to prevent.
 */

function docWith(metadata: Record<string, unknown>): GraphDocumentV2 {
  return { schemaVersion: 2, semantic: { nodes: [], metadata } } as unknown as GraphDocumentV2;
}

describe('readVerification', () => {
  it('reads a well-formed record out of the document metadata', () => {
    const record = readVerification(
      docWith({ verification: { ok: false, revision: 7, errorCount: 3, firstError: 'boom', firstErrorNodeId: 'n1' } }),
    );
    expect(record).toEqual({
      ok: false,
      revision: 7,
      errorCount: 3,
      firstError: 'boom',
      firstErrorNodeId: 'n1',
      summary: undefined,
    });
  });

  it('returns null for a document that has never been verified', () => {
    expect(readVerification(docWith({}))).toBeNull();
    expect(readVerification(undefined)).toBeNull();
    expect(readVerification(null)).toBeNull();
  });

  it('rejects a malformed blob instead of throwing inside a list render', () => {
    // This parses data an older build or a hand edit could have written, so a bad
    // shape must degrade to "never run" rather than crash /projects.
    expect(readVerification(docWith({ verification: 'yes' }))).toBeNull();
    expect(readVerification(docWith({ verification: { ok: true } }))).toBeNull();
    expect(readVerification(docWith({ verification: { revision: 2 } }))).toBeNull();
  });

  it('defaults a missing error count to one on a failure', () => {
    // A failing record with no count still has at least one error by definition;
    // showing "0 errors" beside a red pill would contradict itself.
    expect(readVerification(docWith({ verification: { ok: false, revision: 1 } }))?.errorCount).toBe(1);
    expect(readVerification(docWith({ verification: { ok: true, revision: 1 } }))?.errorCount).toBe(0);
  });
});

describe('projectStatus', () => {
  it('is never-run with no record', () => {
    expect(projectStatus(null, 3)).toBe('never-run');
  });

  it('is verified when the record passed at the current revision', () => {
    expect(projectStatus({ ok: true, revision: 3, errorCount: 0 }, 3)).toBe('verified');
  });

  it('is error when the record failed at the current revision', () => {
    expect(projectStatus({ ok: false, revision: 3, errorCount: 2 }, 3)).toBe('error');
  });

  it('demotes a pass that predates the current revision to stale', () => {
    // The model has changed since anything checked it. This is the assertion that
    // stops a green pill outliving the graph it described.
    expect(projectStatus({ ok: true, revision: 2, errorCount: 0 }, 5)).toBe('stale');
  });

  it('demotes a stale FAILURE too, rather than reporting a fixed error', () => {
    expect(projectStatus({ ok: false, revision: 2, errorCount: 1 }, 5)).toBe('stale');
  });

  it('trusts the record when the revision is unknown', () => {
    // Fixtures and older rows carry no currentRevision; refusing to report any
    // status there would blank the column for them.
    expect(projectStatus({ ok: true, revision: 2, errorCount: 0 }, undefined)).toBe('verified');
  });
});

describe('verificationStore', () => {
  beforeEach(() => window.localStorage.clear());

  it('round-trips a recorded verification', () => {
    recordVerification('g1', { ok: true, revision: 4, errorCount: 0, summary: { params: 11 } });
    expect(readRecordedVerification('g1')).toEqual({
      ok: true,
      revision: 4,
      errorCount: 0,
      summary: { params: 11 },
    });
  });

  it('returns null for a project it has never seen', () => {
    expect(readRecordedVerification('nope')).toBeNull();
    expect(readRecordedVerification(undefined)).toBeNull();
  });

  it('survives a corrupt store rather than throwing', () => {
    window.localStorage.setItem('pytorchviz.verification.v1', 'not json');
    expect(readRecordedVerification('g1')).toBeNull();
    expect(() => recordVerification('g1', { ok: true, revision: 1, errorCount: 0 })).not.toThrow();
  });

  it('prefers the document record over the local one', () => {
    // Server truth wins, so if the record ever starts riding in the document no
    // call site has to change.
    recordVerification('g1', { ok: false, revision: 9, errorCount: 5 });
    const project = {
      id: 'g1',
      document: docWith({ verification: { ok: true, revision: 9, errorCount: 0 } }),
    } as unknown as SavedGraph;
    expect(resolveVerification(project)?.ok).toBe(true);
  });

  it('falls back to the local record when the document carries none', () => {
    recordVerification('g1', { ok: false, revision: 9, errorCount: 5 });
    const project = { id: 'g1', document: docWith({}) } as unknown as SavedGraph;
    expect(resolveVerification(project)?.errorCount).toBe(5);
  });

  it('forgets a project when it is deleted', () => {
    recordVerification('g1', { ok: true, revision: 1, errorCount: 0 });
    forgetVerification('g1');
    expect(readRecordedVerification('g1')).toBeNull();
  });

  it('evicts the oldest entries past the cap so the blob cannot grow forever', () => {
    for (let i = 0; i < 210; i += 1) {
      recordVerification(`g${i}`, { ok: true, revision: 1, errorCount: 0 });
    }
    const stored = JSON.parse(window.localStorage.getItem('pytorchviz.verification.v1') || '{}');
    expect(Object.keys(stored).length).toBeLessThanOrEqual(200);
    // The most recent write must be the one that survived.
    expect(readRecordedVerification('g209')).not.toBeNull();
  });
});

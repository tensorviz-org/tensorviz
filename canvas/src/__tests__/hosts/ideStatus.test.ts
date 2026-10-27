/**
 * The header in the IDE tells the truth about the run (task.md M4/M8).
 *
 * The bug this replaces: the IDE showed the web product's save pill, which read
 * "Offline — changes queued" when the backend went away. Nothing was queued and
 * nothing was being saved — the canvas is a projection of a file and edits go
 * through the host into that file — so the sentence was false in both halves, and
 * it latched once shown.
 *
 * Worse, the state a user in that position actually needed was invisible: the
 * host writes `traceError` into the projection and nothing read it, so a run that
 * failed part way left stale shapes on screen with no indication.
 */
import { ideStatus } from '../../ide/status';

const BASE = {
  graphId: 'ide:abc#Net',
  revision: 1,
  cls: 'Net',
  file: 'models/blocks.py',
  sourceDigest: 'd',
  traced: false,
  greyed: false,
  document: {},
  spanmap: {},
} as never;

describe('what the IDE header says', () => {
  it('says nothing before there is a projection', () => {
    // Not "Saved", not "Not run": there is no graph, so there is no claim to make.
    expect(ideStatus(null)).toBeNull();
  });

  it('a failed run says the shapes are incomplete, and why', () => {
    const status = ideStatus({
      ...BASE,
      traced: true,
      traceError: { message: 'mat1 and mat2 shapes cannot be multiplied', line: 13 },
    } as never);

    expect(status?.tone).toBe('warn');
    expect(status?.text).toMatch(/Run failed/);
    // The message and the line survive into the detail: a pill that said only
    // "failed" would replace one uninformative sentence with another.
    expect(status?.detail).toContain('mat1 and mat2');
    expect(status?.detail).toContain('line 13');
  });

  it('a run that reached the end says the shapes were measured', () => {
    const status = ideStatus({ ...BASE, traced: true } as never);
    expect(status?.tone).toBe('ok');
    expect(status?.text).toMatch(/measured/);
  });

  it('no run yet says the shapes were read, not measured', () => {
    // The distinction the whole product rests on. "Saved" said nothing about it.
    const status = ideStatus(BASE);
    expect(status?.tone).toBe('muted');
    expect(status?.text).toMatch(/read from the source/);
  });

  it('a buffer that does not parse outranks everything else', () => {
    // The graph on screen is not what the file says, which matters more than
    // where its numbers came from.
    const status = ideStatus({
      ...BASE,
      greyed: true,
      traced: true,
      traceError: { message: 'boom' },
    } as never);
    expect(status?.tone).toBe('warn');
    expect(status?.text).toMatch(/Not parsing/);
  });
});

/**
 * What the header says in the IDE, where "saved" is not a thing that happens.
 *
 * The web product autosaves a graph to the project store, and its pill reports
 * that: saving, saved, offline with changes queued. None of it applies here. The
 * canvas is a projection of a `.py`, edits go through the host into that file,
 * and there is nothing queued anywhere — so a user whose backend had gone away
 * was told "Offline — changes queued", which is false twice over and latches once
 * it appears.
 *
 * What a person actually needs to know in this position is whether the numbers on
 * the cards came from running their model, and that is what this reports. It is a
 * pure function of the projection so it can be tested without a workbench, and so
 * the four states are visible in one place rather than spread across a component.
 */
import { ProjectionPayload } from './protocol.generated';

export interface IdeStatus {
  /** `warn` is for the two states where the graph on screen is not what the file
   *  currently says. `ok` and `muted` both describe a truthful graph; they differ
   *  in whether the shapes were measured or read. */
  tone: 'ok' | 'warn' | 'muted';
  text: string;
  /** The long form, for the title attribute: the error a run failed with, which
   *  is too long for a pill and too important to drop. */
  detail: string;
}

/**
 * The status for a projection, or null when there is none — before the first
 * "Open as Graph", where there is nothing true to say yet.
 */
export function ideStatus(projection: ProjectionPayload | null): IdeStatus | null {
  if (!projection) {
    return null;
  }
  if (projection.greyed) {
    return {
      tone: 'warn',
      text: 'Not parsing — showing the last graph that did',
      detail:
        'The file has a syntax error, so this is the last projection that parsed. ' +
        'It is not what the file says right now.',
    };
  }
  if (projection.traceError) {
    const where = projection.traceError.line ? ` (line ${projection.traceError.line})` : '';
    return {
      tone: 'warn',
      text: 'Run failed — shapes are incomplete',
      detail: `${projection.traceError.message}${where}. Every node after the failure has no shape, because the run never reached it.`,
    };
  }
  if (projection.traced) {
    return {
      tone: 'ok',
      text: 'Traced — shapes measured',
      detail: 'These shapes came from running your model, not from reading the source.',
    };
  }
  return {
    tone: 'muted',
    text: 'Not run — shapes read from the source',
    detail:
      'Save the file to run the model and replace these with measured shapes, ' +
      'or trust the workspace when asked.',
  };
}

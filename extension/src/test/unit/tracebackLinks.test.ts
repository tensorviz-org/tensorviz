import * as assert from 'node:assert/strict';
import * as path from 'node:path';

import type { ApplyTarget } from '../../apply';
import type { TraceFailure } from '../../diagnostics';
import {
  TracebackLinkProvider,
  type TracebackFrame,
  type TracebackHost,
  type TracebackLink,
} from '../../tracebackLinks';
import { Uri } from '../vscodeStub';

const ROOT = path.join(path.sep, 'ws');
const MODEL = path.join(ROOT, 'models', 'resid.py');

/** The spanmap of `Net` in `models/resid.py`, as `/api/ide/parse` returns one.
 *
 * `forward` deliberately encloses `fc2`: the innermost span has to win, or a
 * traceback frame would select the method rather than the layer that raised. */
const SPANMAP: Record<string, unknown> = {
  fc1: { file: 'models/resid.py', start: [30, 8], end: [30, 40], kind: 'decl', digest: 'd1' },
  fc2: { file: 'models/resid.py', start: [41, 8], end: [41, 40], kind: 'decl', digest: 'd2' },
  forward: { file: 'models/resid.py', start: [38, 4], end: [45, 20], kind: 'stmt', digest: 'd3' },
};

/** Every call the provider made on the editor, so a test can assert what did NOT
 *  happen as easily as what did — "no node was selected" is half of this file. */
interface Recorded {
  revealed: TracebackFrame[];
  focused: string[];
  reported: Array<{ file: string; failure: TraceFailure }>;
}

function target(): ApplyTarget {
  return {
    uri: Uri.file(MODEL) as never,
    cls: 'Net',
    rel: 'models/resid.py',
    spanmap: SPANMAP,
    sourceDigest: 'sha256:abc',
  };
}

function makeHost(
  overrides: Partial<TracebackHost> = {},
): { host: TracebackHost; recorded: Recorded } {
  const recorded: Recorded = { revealed: [], focused: [], reported: [] };
  const host: TracebackHost = {
    workspaceRoot: () => ROOT,
    target,
    reveal: (frame) => {
      recorded.revealed.push(frame);
    },
    focusNode: (nodeId) => {
      recorded.focused.push(nodeId);
    },
    reportTrace: (frame, failure) => {
      recorded.reported.push({ file: frame.file, failure });
    },
    ...overrides,
  };
  return { host, recorded };
}

/** Drive a whole printed traceback through the provider the way the terminal does:
 *  one line at a time, in order. The order is not incidental — the exception line
 *  arrives after the frames — so a test that fed the block in one call would be
 *  testing an API VS Code does not have. */
function render(provider: TracebackLinkProvider, block: string, terminal = {}): TracebackLink[] {
  const links: TracebackLink[] = [];
  for (const line of block.split('\n')) {
    links.push(...provider.provideTerminalLinks({ line, terminal } as never));
  }
  return links;
}

/** A real three-frame traceback: the worker's harness, the user's model, torch.
 *  Exactly one of the three is a file the user can edit. */
const THREE_FRAMES = [
  'Traceback (most recent call last):',
  `  File "${path.join(path.sep, 'tmp', 'tv-worker', 'harness.py')}", line 88, in <module>`,
  `  File "${MODEL}", line 41, in forward`,
  `  File "${path.join(path.sep, 'opt', 'py', 'site-packages', 'torch', 'nn', 'modules', 'linear.py')}", line 117, in forward`,
  'RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x8 and 4x4)',
].join('\n');

suite('TracebackLinkProvider', () => {
  test('links the one frame of a three-frame traceback that is the user’s own code', () => {
    const { host } = makeHost();
    const links = render(new TracebackLinkProvider(host), THREE_FRAMES);

    assert.equal(links.length, 1, 'the harness and torch frames must not be linked');
    assert.equal(links[0].frame.file, MODEL);
    assert.equal(links[0].frame.line, 41);
    // The innermost span wins: line 41 is inside `forward` too, and offering the
    // method would send the user to a box that is not the one that raised.
    assert.equal(links[0].nodeId, 'fc2');
    assert.equal(links[0].tooltip, 'Show on graph');
  });

  test('covers the whole "File ..., line N, in fn" so the line number is clickable', () => {
    const { host } = makeHost();
    const frameLine = `  File "${MODEL}", line 41, in forward`;
    const [link] = new TracebackLinkProvider(host).provideTerminalLinks({
      line: frameLine,
      terminal: {},
    } as never);

    assert.equal(
      frameLine.slice(link.startIndex, link.startIndex + link.length),
      `File "${MODEL}", line 41, in forward`,
    );
  });

  test('offers nothing for frames outside the workspace, including a venv inside it', () => {
    const { host } = makeHost();
    const outside = [
      `  File "${path.join(path.sep, 'opt', 'py', 'torch', 'nn', 'functional.py')}", line 117, in linear`,
      // A virtualenv normally lives IN the open folder, so "inside the workspace"
      // alone would link exactly the torch frames this provider is silent about.
      `  File "${path.join(ROOT, '.venv', 'lib', 'python3.11', 'site-packages', 'torch', 'nn', 'functional.py')}", line 21, in relu`,
    ].join('\n');

    assert.deepEqual(render(new TracebackLinkProvider(host), outside), []);
  });

  test('links a workspace frame that no span covers, and selects no node for it', async () => {
    const { host, recorded } = makeHost();
    const provider = new TracebackLinkProvider(host);
    const script = path.join(ROOT, 'scripts', 'train.py');
    const links = render(
      provider,
      [
        'Traceback (most recent call last):',
        `  File "${script}", line 12, in <module>`,
        'ValueError: expected 2 dimensions',
      ].join('\n'),
    );

    assert.equal(links.length, 1, 'a file the user can edit is worth opening even with no node');
    assert.equal(links[0].nodeId, undefined);
    assert.notEqual(links[0].tooltip, 'Show on graph');

    await provider.handleTerminalLink(links[0]);
    assert.deepEqual(
      recorded.revealed.map((frame) => `${frame.file}:${frame.line}`),
      [`${script}:12`],
    );
    assert.deepEqual(recorded.focused, [], 'there is no node to select, so none is selected');
  });

  test('activating a link opens the frame, selects its node and files the exception', async () => {
    const { host, recorded } = makeHost();
    const provider = new TracebackLinkProvider(host);
    const [link] = render(provider, THREE_FRAMES);

    await provider.handleTerminalLink(link);

    assert.deepEqual(recorded.revealed, [{ file: MODEL, line: 41, fn: 'forward' }]);
    assert.deepEqual(recorded.focused, ['fc2']);
    assert.deepEqual(recorded.reported, [
      {
        file: MODEL,
        failure: {
          type: 'RuntimeError',
          message: 'mat1 and mat2 shapes cannot be multiplied (2x8 and 4x4)',
          line: 41,
        },
      },
    ]);
  });

  test('a link reports the exception ITS traceback ended with, not the newest one', async () => {
    const { host, recorded } = makeHost();
    const provider = new TracebackLinkProvider(host);
    const terminal = {};
    const [first] = render(provider, THREE_FRAMES, terminal);
    // The same terminal runs the script again and fails differently. The first
    // link is still on screen and still means what it meant.
    render(
      provider,
      [
        'Traceback (most recent call last):',
        `  File "${MODEL}", line 30, in __init__`,
        'TypeError: Linear() missing 1 required positional argument',
      ].join('\n'),
      terminal,
    );

    await provider.handleTerminalLink(first);
    assert.equal(recorded.reported[0].failure.type, 'RuntimeError');
  });

  test('links every workspace frame, since each is a place the user can fix', async () => {
    const { host } = makeHost();
    const script = path.join(ROOT, 'scripts', 'train.py');
    const links = render(
      new TracebackLinkProvider(host),
      [
        'Traceback (most recent call last):',
        `  File "${script}", line 12, in <module>`,
        `  File "${MODEL}", line 41, in forward`,
        'RuntimeError: boom',
      ].join('\n'),
    );

    assert.deepEqual(
      links.map((link) => `${path.basename(link.frame.file)}:${link.frame.line}`),
      ['train.py:12', 'resid.py:41'],
    );
    // Only the innermost one is inside a span, so only it carries a node — which
    // is what "the traceback resolves to a node" means when the outer frames are
    // the user's too.
    assert.deepEqual(
      links.map((link) => link.nodeId),
      [undefined, 'fc2'],
    );
  });

  test('offers nothing when no folder is open, because nothing is then the user’s', () => {
    const { host } = makeHost({ workspaceRoot: () => undefined });
    assert.deepEqual(render(new TracebackLinkProvider(host), THREE_FRAMES), []);
  });

  test('links the frame but names no node when the canvas is projecting another class', () => {
    const { host } = makeHost({ target: () => undefined });
    const links = render(new TracebackLinkProvider(host), THREE_FRAMES);

    assert.equal(links.length, 1);
    assert.equal(links[0].nodeId, undefined, 'there is no spanmap to resolve line 41 through');
  });
});

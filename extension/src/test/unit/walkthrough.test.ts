import * as assert from 'node:assert/strict';

import {
  CREATE_ENVIRONMENT_COMMAND,
  EnvironmentHost,
  EnvironmentProbe,
  FirstRunHost,
  SELECT_INTERPRETER_COMMAND,
  checkEnvironment,
  qualifiedWalkthroughId,
  showWalkthroughOnce,
} from '../../walkthrough';

interface FirstRunSpy {
  host: FirstRunHost;
  opened: string[];
  remembered: number;
}

function firstRun(options: {
  seen?: boolean;
  editorsOpen?: boolean;
}): FirstRunSpy {
  const opened: string[] = [];
  const spy = { opened, remembered: 0 } as FirstRunSpy;
  spy.host = {
    seen: () => options.seen === true,
    remember: async () => {
      spy.remembered += 1;
    },
    hasOpenEditors: () => options.editorsOpen === true,
    open: async (id) => {
      opened.push(id);
    },
  };
  return spy;
}

suite('showWalkthroughOnce', () => {
  test('opens the walkthrough on the first launch in any VS Code', async () => {
    // The caller decides "empty": it filters the Welcome/Get Started tab, so a bare fresh window
    // — the one a VSIX install activates in — reaches here with hasOpenEditors() false.
    const spy = firstRun({});

    assert.equal(await showWalkthroughOnce(spy.host), 'shown');
    assert.deepEqual(spy.opened, [qualifiedWalkthroughId()]);
    assert.equal(spy.remembered, 1);
  });

  test('a second launch opens nothing', async () => {
    const spy = firstRun({ seen: true });

    assert.equal(await showWalkthroughOnce(spy.host), 'already-seen');
    assert.deepEqual(spy.opened, []);
  });

  test('does not shove itself in front of a window that already has editors', async () => {
    const spy = firstRun({ editorsOpen: true });

    assert.equal(await showWalkthroughOnce(spy.host), 'editors-open');
    assert.deepEqual(spy.opened, []);
    assert.equal(
      spy.remembered,
      1,
      'the flag is still consumed, or this ambushes the user weeks later',
    );
  });

  test('the qualified id is <publisher>.<name>#<walkthrough id>', () => {
    assert.equal(qualifiedWalkthroughId(), 'tensorviz.tensorviz-vscode#tensorviz.firstRun');
  });
});

interface EnvironmentSpy {
  host: EnvironmentHost;
  executed: string[];
  shown: string[];
}

function environment(probe: EnvironmentProbe, answer?: string): EnvironmentSpy {
  const executed: string[] = [];
  const shown: string[] = [];
  const respond = async (message: string, ...actions: string[]): Promise<string | undefined> => {
    shown.push(message);
    return actions.includes(answer ?? '') ? answer : undefined;
  };
  return {
    executed,
    shown,
    host: {
      probe: async () => probe,
      info: respond,
      warn: respond,
      execute: async (command) => {
        executed.push(command);
      },
    },
  };
}

suite('checkEnvironment', () => {
  test('a working interpreter reports its banner and runs nothing', async () => {
    const spy = environment({ ok: true, banner: 'Python 3.11.15 torch 2.7.1' });

    assert.equal(await checkEnvironment(spy.host), 'ok');
    assert.deepEqual(spy.shown, ['Python 3.11.15 torch 2.7.1']);
    assert.deepEqual(spy.executed, []);
  });

  test('a missing interpreter offers ms-python’s creator, not one of ours', async () => {
    const spy = environment({ ok: false, message: 'No interpreter with torch' }, 'Create environment');

    assert.equal(await checkEnvironment(spy.host), 'create');
    assert.deepEqual(spy.executed, [CREATE_ENVIRONMENT_COMMAND]);
  });

  test('"Select interpreter" hands off to ms-python from either outcome', async () => {
    const good = environment({ ok: true, banner: 'ok' }, 'Select interpreter');
    assert.equal(await checkEnvironment(good.host), 'select');
    assert.deepEqual(good.executed, [SELECT_INTERPRETER_COMMAND]);

    const bad = environment({ ok: false, message: 'nope' }, 'Select interpreter');
    assert.equal(await checkEnvironment(bad.host), 'select');
    assert.deepEqual(bad.executed, [SELECT_INTERPRETER_COMMAND]);
  });

  test('a dismissed failure stays a failure', async () => {
    const spy = environment({ ok: false, message: 'No interpreter with torch' });

    assert.equal(await checkEnvironment(spy.host), 'dismissed');
    assert.deepEqual(spy.executed, []);
  });
});

/**
 * The first five minutes (task.md F1).
 *
 * Two things live here, and neither imports `vscode`: the hosts are injected so the decisions
 * — "has this user seen the walkthrough", "is there a Python with torch" — are unit tests
 * rather than a manual pass through a fresh profile.
 *
 *  1. `showWalkthroughOnce`, so a fresh install has a first impression without waiting for the
 *     user to find the walkthrough on the Welcome page. It fires once per install in any VS
 *     Code (stock or the fork) and never opens over the user's own editors: the caller filters
 *     out the Welcome/Get Started surface so a window that only has that still counts as empty.
 *     The flag it writes is what keeps it from ambushing a returning user.
 *  2. `checkEnvironment`, the "use this one / create one" step. It reuses the same interpreter
 *     probe the backend launch uses, so the walkthrough cannot say yes to an environment the
 *     product will later reject.
 */

/** Must match `contributes.walkthroughs[0].id` in package.json. */
export const WALKTHROUGH_ID = 'tensorviz.firstRun';
/** `<publisher>.<name>` from package.json — how VS Code addresses a walkthrough. */
export const EXTENSION_ID = 'tensorviz.tensorviz-vscode';
/** globalState key. Global, not workspace: "get started" is per person, not per repository. */
export const FIRST_RUN_KEY = 'tensorviz.firstRunShown';

export function qualifiedWalkthroughId(): string {
  return `${EXTENSION_ID}#${WALKTHROUGH_ID}`;
}

export interface FirstRunHost {
  /** Whether the walkthrough has been opened before, from globalState. */
  seen(): boolean;
  /** Record that it has, so this never runs twice. */
  remember(): Thenable<void>;
  /** True when the window already has the user's own editors open — a restored session, or a
   *  folder they came back to. The caller excludes the Welcome/Get Started surface, so a bare
   *  fresh window counts as empty. Shoving a walkthrough over someone's code is worse than not
   *  showing it. */
  hasOpenEditors(): boolean;
  open(qualifiedId: string): Thenable<unknown>;
}

export type FirstRunOutcome = 'shown' | 'already-seen' | 'editors-open';

/**
 * Open the walkthrough the first time this person ever activates the extension.
 *
 * `remember()` is called in the `editors-open` case too. The alternative — keep trying until a
 * window happens to be empty — turns a one-time greeting into something that ambushes the user
 * weeks later, when they have long since worked out what tensorViz is. The command
 * `tensorViz: Get Started` is how it is reached after that.
 */
export async function showWalkthroughOnce(host: FirstRunHost): Promise<FirstRunOutcome> {
  if (host.seen()) {
    return 'already-seen';
  }
  await host.remember();
  if (host.hasOpenEditors()) {
    return 'editors-open';
  }
  await host.open(qualifiedWalkthroughId());
  return 'shown';
}

// ---------------------------------------------------------------------------
// "Pick or create the torch environment"
// ---------------------------------------------------------------------------

/** ms-python's own picker. Ours would be a second, competing source of truth. */
export const SELECT_INTERPRETER_COMMAND = 'python.setInterpreter';
/** ms-python's environment creator (venv/conda), which is the "create one" answer. */
export const CREATE_ENVIRONMENT_COMMAND = 'python.createEnvironment';

const SELECT_LABEL = 'Select interpreter';
const CREATE_LABEL = 'Create environment';

/** What the probe found, narrowed to what this step needs from `BackendInterpreterResult`. */
export type EnvironmentProbe = { ok: true; banner: string } | { ok: false; message: string };

export interface EnvironmentHost {
  probe(): Promise<EnvironmentProbe>;
  /** `showInformationMessage(message, ...actions)`; resolves to the chosen action or undefined. */
  info(message: string, ...actions: string[]): Thenable<string | undefined>;
  /** `showWarningMessage(message, ...actions)`. */
  warn(message: string, ...actions: string[]): Thenable<string | undefined>;
  execute(command: string): Thenable<unknown>;
}

export type EnvironmentOutcome = 'ok' | 'select' | 'create' | 'dismissed';

/**
 * Answer "is there a Python here that can run tensorViz", and offer the two ways out.
 *
 * A working environment still shows a message: this is a walkthrough step the user pressed a
 * button on, and silence would read as nothing having happened. The offer to switch stays on
 * that path too — the first interpreter that passes is not necessarily the one they meant.
 */
export async function checkEnvironment(host: EnvironmentHost): Promise<EnvironmentOutcome> {
  const found = await host.probe();
  const pick = found.ok
    ? await host.info(found.banner, SELECT_LABEL)
    : await host.warn(found.message, SELECT_LABEL, CREATE_LABEL);

  if (pick === SELECT_LABEL) {
    await host.execute(SELECT_INTERPRETER_COMMAND);
    return 'select';
  }
  if (pick === CREATE_LABEL) {
    await host.execute(CREATE_ENVIRONMENT_COMMAND);
    return 'create';
  }
  return found.ok ? 'ok' : 'dismissed';
}

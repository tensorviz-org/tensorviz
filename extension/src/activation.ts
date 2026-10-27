/**
 * When the backend should start, decided from what is on screen.
 *
 * task.md P0 deletes the "Start tensorViz" row from the Models view, on the grounds that there
 * is nothing for a user to start: opening a folder with a model in it is the instruction. That
 * left a hole. The class list comes from the backend's AST scan
 * (`/api/ide/workspace/classes`), `workspaceClasses` deliberately throws rather than starting a
 * python process from a CodeLens refresh, and `CodeLensIndex.refresh` deliberately swallows
 * that — so with the row gone and nothing else starting the stack, a fresh window sat on
 * "Indexing…" forever and the affordance that used to rescue it had been removed.
 *
 * So something must decide, locally and cheaply, that this folder is worth starting a backend
 * for. That decision is here, as a pure function over document text, for two reasons: it is the
 * kind of rule that is wrong in interesting ways (a class inside a string, a comment, a
 * subclass of a subclass) and belongs in tests rather than in an event handler; and the
 * alternative — starting the backend on activation for every folder — spends a python process
 * and an `import torch` on people who opened a repository with no model in it.
 *
 * This is a TRIGGER, never the answer. The backend's AST scan remains the authority on which
 * classes exist; all this has to get right is "is it worth looking". A false positive costs one
 * backend start that finds nothing. A false negative is the bug above, so the rule is
 * deliberately generous.
 */

/** `class Net(nn.Module):`, `class Net(Module):`, `class Net(torch.nn.Module):`, and the same
 *  with several bases — `class Net(Base, nn.Module):`. Anchored at a line start (indented
 *  classes are ordinary inside `if TYPE_CHECKING:` or a factory) so a mention of the phrase
 *  inside a sentence does not count. */
const MODULE_CLASS = /^[ \t]*class[ \t]+(\w+)[ \t]*\(([^)]*)\)[ \t]*:/gm;

/** A base list that means "this is a torch module". Bare `Module` counts because
 *  `from torch.nn import Module` is ordinary, and this is a trigger, not a verdict. */
const TORCH_BASE = /(^|[\s,])(?:torch\.)?(?:nn\.)?Module(?=[\s,\]]|$)/;

/** Lines whose content is a comment, and nothing else. Removing them is enough to stop
 *  `# class Old(nn.Module):` from starting a backend; a class inside a triple-quoted string
 *  still would, which costs one wasted start and is not worth a Python parser here. */
const FULL_LINE_COMMENT = /^[ \t]*#.*$/gm;

/**
 * Does this Python source look like it defines an `nn.Module` subclass?
 *
 * Used by the direct-base auto-open shortcut. Backend startup must also allow imported and
 * indirect bases; only source discovery can classify them. Never decides graph contents.
 */
export function declaresModule(text: string): boolean {
  return moduleClasses(text).length > 0;
}

/**
 * The `nn.Module` subclasses this source declares, by name, in source order.
 *
 * The same generous rule as `declaresModule`, and the reason the fork can open the canvas the
 * instant a model file opens: auto-open used to ask the backend to scan the whole workspace
 * first — measured at 2.2 to 5.6 s on this repository — to learn a class name that is on the
 * screen in front of the user. A base list is not a verdict; the projection is what decides
 * whether the class is really a module, and it says so on the canvas.
 */
export function moduleClasses(text: string): string[] {
  if (!text || !text.includes('class')) {
    return [];
  }
  const source = text.replace(FULL_LINE_COMMENT, '');
  const names: string[] = [];
  MODULE_CLASS.lastIndex = 0;
  for (let match = MODULE_CLASS.exec(source); match; match = MODULE_CLASS.exec(source)) {
    if (TORCH_BASE.test(match[2])) {
      names.push(match[1]);
    }
  }
  return names;
}

/** What `shouldStartBackend` needs to know about the world. Every field is a fact the caller
 *  already has, so the decision can be tested without an editor. */
export interface StartTrigger {
  /** The opened document's language id. */
  readonly languageId: string;
  /** Its uri scheme: `file`, or something virtual we should ignore. */
  readonly scheme: string;
  /** Its text. Read once by the caller; this function does not go and fetch it. */
  readonly text: string;
  /** Is a backend already running for this window? */
  readonly backendRunning: boolean;
  /** Is the workspace trusted? An untrusted window must not start a python process, which is
   *  the whole point of adopting Workspace Trust in D20. */
  readonly trusted: boolean;
  /** Has a start already been attempted and failed? One failed start per window is a
   *  diagnostic; one per keystroke-triggered save is a loop. */
  readonly startFailed: boolean;
}

/**
 * Should opening or saving this document start the backend?
 *
 * Deliberately conservative about everything except the class rule: a python file, on disk, in
 * a trusted window, with no backend yet and no earlier failure.
 */
export function shouldStartBackend(trigger: StartTrigger): boolean {
  if (trigger.backendRunning || trigger.startFailed) return false;
  if (!trigger.trusted) return false;
  if (trigger.languageId !== 'python' || trigger.scheme !== 'file') return false;
  // Imported aliases and indirect bases need the Python index before we know whether they
  // are modules. Do not duplicate its library catalog here: every nonempty base list merits
  // a source scan, but only discovered models get a CodeLens. Use a fresh regex so the
  // direct-base shortcut's global lastIndex cannot affect startup after a reload.
  const source = trigger.text.replace(FULL_LINE_COMMENT, '');
  return Array.from(source.matchAll(new RegExp(MODULE_CLASS)))
    .some((match) => match[2].trim().length > 0)
    || (/^\s*(?:from\s+jax(?:[.\s])|import\s+jax(?:[.\s,]|$))/m.test(source)
      && /^\s*def\s+\w+\s*\(/m.test(source));
}


/** What decides whether opening a model file should also draw it. */
export interface AutoOpenTrigger {
  /** Is this the fork? The fork opens a graph nobody asked for, because it is the product
   *  whose whole proposition is the graph beside the code. Another editor does not, by
   *  DEFAULT — doing it unasked inside a window someone opened for something else would be
   *  taking it over. `enabledExplicitly` is how they ask. */
  readonly isFork: boolean;
  /** Is a graph already on screen? A second one is not help. */
  readonly graphShowing: boolean;
  /** `tensorviz.autoOpenCanvas`. A setting, not a memento: closing the canvas is not "never
   *  again" — the canvas is the product, and a person who closed it to read a traceback has not
   *  asked to stop seeing their models. Turning it off is a preference, and preferences are
   *  configured where the user can find and reverse them. */
  readonly enabled: boolean;
  /** Whether that setting was WRITTEN by the user rather than left at its default.
   *
   *  The fork gate used to be a veto, and a defensible one — until the canvas became the
   *  project rather than a class, and the first thing anyone does in a normal editor is open
   *  a model file and see nothing happen, with no hint that a setting governs it. The rule
   *  that replaces it keeps the manners and drops the dead end: the fork opens by default,
   *  and any other editor opens when the user has said so in their own settings. Default
   *  `true` is not saying so — nobody chose it. */
  readonly enabledExplicitly?: boolean;
  /** The classes the backend found in this file, in source order. */
  readonly classes: readonly string[];
  /** The file's source, so a class that is only a COMPONENT can be told from the model.
   *  Optional: without it the rule degrades to "the first class", which is what this
   *  used to do unconditionally. */
  readonly source?: string;
}

/** Classes this file builds INSIDE another of its own classes.
 *
 * `self.encoder_stack = EncoderStack(...)` makes `EncoderStack` a component of whatever
 * declares it. Text-level on purpose: the caller reads the buffer rather than waiting on
 * the backend's AST scan, which cost 2 to 6 seconds between opening a model and seeing it.
 */
function componentClasses(source: string, classes: readonly string[]): Set<string> {
  const used = new Set<string>();
  // Everything from `if __name__ == "__main__":` on is EXAMPLE USAGE, not the model's
  // structure — and it is where the generated file writes `model = BertBase()`. Counting
  // that made the model look like a component of itself, every class got excluded, and the
  // rule fell straight back to "the first class", which is the bug it exists to fix. The
  // first version of this shipped with tests that had no `__main__` block, so they passed.
  const guard = source.search(/^if\s+__name__\s*==/m);
  const bodies = (guard === -1 ? source : source.slice(0, guard)).replace(FULL_LINE_COMMENT, '');
  for (const name of classes) {
    // `Name(` anywhere except its own `class Name` declaration line.
    const call = new RegExp(`(?<!class\\s)\\b${name}\\s*\\(`);
    if (call.test(bodies)) {
      used.add(name);
    }
  }
  return used;
}

/**
 * Should opening this file draw its top-level class beside the code, and which one?
 *
 * The MODEL, which is usually the first class and is not always. The rule used to be
 * "`classes[0]`, because it is the one the user is looking at and picking cleverly would
 * be a rule nobody could predict" — sound for a file somebody wrote by hand, and wrong for
 * the files this product generates: `code_generation` emits helper modules FIRST and the
 * model LAST, so every modular model it wrote opened on a helper.
 *
 * That was not a cosmetic mis-pick. `EncoderStack(module_type=..., num_layers=12, ...)`
 * cannot be constructed without arguments, so pressing Validate on a generated BERT
 * answered "Failed at build", and projecting it standalone reported its output as
 * disconnected — 38 findings about a class the user never asked to see.
 *
 * So: the first class that nothing else in the file BUILDS. A class another class
 * instantiates is a component of it; the one no one instantiates is the model. That is
 * still a rule someone can predict, and for a single-class file it is the same answer as
 * before. Ambiguous files (several unreferenced classes) keep the old behaviour: the first.
 */
export function autoOpenClass(trigger: AutoOpenTrigger): string | undefined {
  if (!trigger.enabled || trigger.graphShowing) {
    return undefined;
  }
  if (!trigger.isFork && !trigger.enabledExplicitly) {
    return undefined;
  }
  if (trigger.source && trigger.classes.length > 1) {
    const components = componentClasses(trigger.source, trigger.classes);
    const root = trigger.classes.find((name) => !components.has(name));
    if (root) {
      return root;
    }
  }
  return trigger.classes[0];
}

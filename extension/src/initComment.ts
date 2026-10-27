/**
 * `# tensorviz: init=Net(hidden=8, depth=2)` — the line that says how to BUILD a model.
 *
 * task.md P0 / §6.1. A class whose `__init__` needs an argument cannot be constructed at
 * all: the worker exits 3 and the user sees a warning toast about a log. That is the most
 * common first-trace failure on a real repository, because real models take a config, and
 * it is the one this declaration answers.
 *
 * This is the THIRD reader of the same line. `backend/utils/code_parser.py` reads it for
 * the tier that draws the graph, `backend/services/ingest/service.py` reads it for the
 * worker that builds the object, and this file reads it for the host that assembles the
 * worker's argv. All three refuse the same things for the same reasons, and
 * `tests/graph/test_ide_init_comment.py` plus `src/test/unit/initComment.test.ts` say so.
 *
 * Deliberately hand-rolled rather than a Python-expression parser: the accepted grammar is
 * one call, keyword arguments only, JSON-shaped literal values. Anything richer would have
 * to be evaluated, and evaluating it means running the user's code to decide how to run the
 * user's code.
 */

/** The pattern, character for character the same as both Python copies. */
export const INIT_COMMENT = /#\s*tensorviz:\s*init\s*=\s*([^\n]+)/g;

export interface InitDeclaration {
  /** The class the declaration names; empty when the line was too malformed to say. */
  readonly cls: string;
  /** Constructor keyword arguments, ready for `--overrides`. Empty when `error` is set. */
  readonly overrides: Record<string, unknown>;
  /** Why the declaration will not be used, or undefined when it will be.
   *
   * A refused declaration is reported, never dropped: a user who writes a
   * `# tensorviz: init=` that silently does nothing has no way to discover that. */
  readonly error?: string;
  /** 0-based line the comment is on, so a diagnostic can point at it. */
  readonly line: number;
}

/**
 * `cfg=ModelConfig(width=8)` — an argument that is itself built by calling a name.
 *
 * Real models take a config object, which no literal can spell. The accepted grammar stays
 * one level deep: a bare name, called with literal keyword arguments, resolved by the worker
 * against the module it has just imported. A dotted path, a nested call, or a bare name are
 * each refused with their own reason, because each is a different mistake.
 *
 * The brand key is what keeps a user's own `{"name": …, "kwargs": …}` dict from being read
 * as one of these.
 */
export interface ConfigCall {
  readonly __tensorvizConfigCall: true;
  readonly name: string;
  readonly kwargs: Record<string, unknown>;
}

export function configCall(name: string, kwargs: Record<string, unknown>): ConfigCall {
  return { __tensorvizConfigCall: true, name, kwargs };
}

export function isConfigCall(value: unknown): value is ConfigCall {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as ConfigCall).__tensorvizConfigCall === true
  );
}

/** A literal value, or `undefined` when the text is not one. */
function readLiteral(text: string): { value: unknown } | undefined {
  const trimmed = text.trim();
  if (trimmed === 'True' || trimmed === 'true') {
    return { value: true };
  }
  if (trimmed === 'False' || trimmed === 'false') {
    return { value: false };
  }
  if (trimmed === 'None' || trimmed === 'null') {
    return { value: null };
  }
  // Python's single-quoted strings are not JSON, and they are what people write.
  const singleQuoted = /^'((?:[^'\\]|\\.)*)'$/.exec(trimmed);
  if (singleQuoted) {
    return { value: singleQuoted[1].replace(/\\'/g, "'") };
  }
  // Tuples have no JSON spelling; a list is what the worker receives either way.
  const asJson = trimmed.startsWith('(') && trimmed.endsWith(')')
    ? `[${trimmed.slice(1, -1).replace(/,\s*$/, '')}]`
    : trimmed;
  try {
    return { value: JSON.parse(asJson) as unknown };
  } catch {
    return undefined;
  }
}

/** Split on top-level commas only, so `dims=[1, 2]` stays one argument. */
function splitArguments(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let quote: string | undefined;
  let start = 0;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quote) {
      if (ch === '\\') {
        i += 1;
      } else if (ch === quote) {
        quote = undefined;
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
    } else if (ch === '(' || ch === '[' || ch === '{') {
      depth += 1;
    } else if (ch === ')' || ch === ']' || ch === '}') {
      depth -= 1;
    } else if (ch === ',' && depth === 0) {
      parts.push(text.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(text.slice(start));
  return parts.map((part) => part.trim()).filter((part) => part.length > 0);
}

/**
 * Read `ModelConfig(width=8)` as the thing to build for `keyword`, or say why not.
 *
 * Each refusal names its own mistake, because "that is not a literal" sends a user who
 * wrote a dotted path looking for the wrong problem.
 */
function readConfigCall(
  keyword: string,
  text: string,
): { value: ConfigCall } | { error: string } {
  const trimmed = text.trim();
  const call = /^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)$/s.exec(trimmed);
  if (!call) {
    return {
      error:
        `\`${keyword}\` must be a literal value (a number, string, bool, list, tuple or dict) ` +
        'or a call that builds one, e.g. `cfg=ModelConfig()`. tensorViz will not evaluate an ' +
        'arbitrary expression to build your model.',
    };
  }
  if (call[1].includes('.')) {
    return {
      error:
        `\`${keyword}\` must name a class to build, e.g. \`${keyword}=ModelConfig()\`. Import ` +
        'the name into this file; tensorViz will not resolve a dotted path.',
    };
  }
  const kwargs: Record<string, unknown> = {};
  for (const argument of splitArguments(call[2])) {
    const inner = /^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)(.*)$/s.exec(argument);
    if (!inner) {
      return {
        error:
          `Name the arguments of \`${call[1]}\`: write \`${keyword}=${call[1]}(width=8)\`, not ` +
          'positional values.',
      };
    }
    const literal = readLiteral(inner[2]);
    if (!literal) {
      // One level only: building the argument of the argument would be an evaluator.
      return {
        error:
          `\`${inner[1]}\` must be a literal value (a number, string, bool, list, tuple or ` +
          `dict). tensorViz builds one ${call[1]}(...), not an expression.`,
      };
    }
    kwargs[inner[1]] = literal.value;
  }
  return { value: configCall(call[1], kwargs) };
}

function readCall(value: string, line: number): InitDeclaration {
  const text = value.trim();
  const call = /^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$/s.exec(text);
  if (!call) {
    return {
      cls: /^[A-Za-z_][A-Za-z0-9_]*$/.test(text) ? text : '',
      overrides: {},
      line,
      error:
        'An init declaration must be a call on the class name, e.g. `# tensorviz: init=Net(hidden=8)`.',
    };
  }
  const cls = call[1];
  const overrides: Record<string, unknown> = {};
  for (const argument of splitArguments(call[2])) {
    const keyword = /^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)(.*)$/s.exec(argument);
    if (!keyword) {
      return {
        cls,
        overrides: {},
        line,
        error: `Name the arguments: write \`# tensorviz: init=${cls}(hidden=8)\`, not positional values.`,
      };
    }
    const literal = readLiteral(keyword[2]);
    if (literal) {
      overrides[keyword[1]] = literal.value;
      continue;
    }
    const built = readConfigCall(keyword[1], keyword[2]);
    if ('error' in built) {
      return { cls, overrides: {}, line, error: built.error };
    }
    overrides[keyword[1]] = built.value;
  }
  return { cls, overrides, line };
}

/**
 * The declaration `source` carries for `cls`, or undefined when it carries none.
 *
 * A file holding three models carries three lines, so a declaration naming another class is
 * not this class's — reading the wrong one would build the wrong model, with arguments that
 * may well be valid for it, so the failure would be silent and the graph simply wrong. A
 * line too malformed to name any class is returned regardless of the filter, because the
 * user needs to be told about it.
 */
export function declaredInit(source: string, cls?: string): InitDeclaration | undefined {
  const pattern = new RegExp(INIT_COMMENT.source, 'g');
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(source)) !== null) {
    const line = source.slice(0, match.index).split('\n').length - 1;
    const found = readCall(match[1], line);
    if (cls !== undefined && found.cls && found.cls !== cls) {
      continue;
    }
    return found;
  }
  return undefined;
}

/** The `--overrides` argument value for a declaration, or undefined when there is nothing
 *  to pass. A refused declaration passes nothing: the construction then fails the way it
 *  would have without the line, and the diagnostic explains why the line was not used. */
export function overridesArgument(declaration: InitDeclaration | undefined): string | undefined {
  if (!declaration || declaration.error || Object.keys(declaration.overrides).length === 0) {
    return undefined;
  }
  // A config call cannot travel as `--overrides`: it names something only the worker can
  // build, in the module it has just imported. Passing the OTHER arguments alone would be
  // worse than passing none — `ingest_source` takes `dict(ref.overrides) or
  // _init_from_comment(...)`, so a partial dict wins and the config is never built at all.
  // Passing nothing makes the worker read the declaration itself, which is the whole line.
  if (Object.values(declaration.overrides).some(isConfigCall)) {
    return undefined;
  }
  return JSON.stringify(declaration.overrides);
}

/** The comment skeleton the Quick Fix inserts above a class that needs arguments.
 *
 * An entry that already carries a value — `hidden=128`, from `missingArguments` below —
 * is written as it stands; a bare name gets the blank the user has to fill. Both spellings
 * appear in the same line when the repository answered some of the arguments and not others.
 */
export function initSkeleton(cls: string, argumentNames: readonly string[]): string {
  const filled = argumentNames.length > 0 ? argumentNames : ['arg'];
  return `# tensorviz: init=${cls}(${filled
    .map((name) => (name.includes('=') ? name : `${name}=`))
    .join(', ')})`;
}

/** The `name=value` pairs a skeleton line proposes, ignoring the blanks.
 *
 * What the Quick Fix needs to disclose: inserting `hidden=128` into somebody's source
 * without saying where 128 came from is the silent auto-pick the backend went out of its
 * way not to make.
 */
export function proposedArguments(skeleton: string): string[] {
  return [...initArguments(skeleton)].map(([name, value]) => `${name}=${value}`);
}

/** `name -> value`, verbatim, for every argument of a `# tensorviz: init=` line that has one.
 *
 * Verbatim rather than parsed-and-re-rendered: `True`, `(3, 3)` and `'relu'` are Python
 * spellings, and a round trip through this file's JSON-shaped reader turns the first into
 * `true` and the second into a list — neither of which is what the backend proposed.
 *
 * The closing `)` is found by requiring the delimiter that follows it, so a nested config
 * call (`Net(cfg=ModelConfig())`) ends where it really ends.
 */
function initArguments(text: string): Map<string, string> {
  const found = /#\s*tensorviz:\s*init\s*=\s*[A-Za-z_][A-Za-z0-9_]*\((.*?)\)(?:`|\n|$)/s.exec(text);
  const values = new Map<string, string>();
  for (const argument of found ? splitArguments(found[1]) : []) {
    const keyword = /^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)(.*)$/s.exec(argument);
    if (keyword && keyword[2].trim().length > 0) {
      values.set(keyword[1], keyword[2].trim());
    }
  }
  return values;
}

/** The word `missing argument \`cfg\`` out of the worker's own construction failure.
 *
 * The worker reports what Python said — `__init__() missing 1 required positional argument:
 * 'cfg'` — and the argument names in it are the whole value of the diagnostic: a message
 * that says "could not construct Transformer" and stops sends the user to read a traceback.
 *
 * A name comes back as `hidden=128` when the SAME failure carries the line the backend wants
 * written and that line has a value for it: `services/ingest/service.py` reads every
 * all-literal instantiation of the class in the workspace, and the answer is usually there.
 * The value is attached only to the argument it was proposed for, so the config-object case —
 * where the missing name belongs to the config class and the advice names the model —
 * attaches nothing. A repository that contradicts itself gets a blank skeleton from the
 * backend on purpose; scraping a value out of the sentence that lists the candidates would
 * undo the one decision it declined to make.
 */
export function missingArguments(detail: string, reported: readonly string[] = []): string[] {
  const names = new Set(reported.filter(name => /^[A-Za-z_][A-Za-z0-9_]*$/.test(name)));
  // The quoted list only, not the rest of the message. `detail` is the worker's whole
  // failure — Python's sentence AND the advice after it — so a greedy tail also swallows
  // any quoted VALUE the advice proposes, and `act='relu'` would report an argument named
  // `relu` that no constructor has.
  const required =
    /missing \d+ required (?:positional|keyword-only) arguments?: ((?:'[^']*'(?:, | and )?)+)/
      .exec(detail);
  if (required) {
    for (const quoted of required[1].matchAll(/'([^']+)'/g)) {
      names.add(quoted[1]);
    }
  }
  // The other spelling: `__init__() takes 2 positional arguments but 1 was given` names none,
  // so nothing is invented for it — the skeleton falls back to a placeholder.
  const proposed = initArguments(detail);
  return [...names].map((name) => {
    const value = proposed.get(name);
    return value === undefined ? name : `${name}=${value}`;
  });
}

/** 0-based line of `class <cls>` in `source`, or 0 when it is not found.
 *
 * The declaration and the diagnostic both have to land somewhere the user is looking, and
 * the class line is where they will look for "this class could not be built". A regex is
 * enough: the host already knows the class name, so there is nothing to disambiguate. */
export function classLine(source: string, cls: string): number {
  const lines = source.split('\n');
  const pattern = new RegExp(`^\\s*class\\s+${cls}\\b`);
  const found = lines.findIndex((line) => pattern.test(line));
  return found < 0 ? 0 : found;
}

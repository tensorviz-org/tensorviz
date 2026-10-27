/* GENERATED from packages/contracts/src/initializerArguments.ts; npm run protocol:sync. */
/** Relational checks shared by live retention and the portable reader.
 * Structural types come from Python. Values describe initializer entry only. */
import type { components } from '../types/semanticContracts.generated';

type Module = components['schemas']['ModuleInstance'];
const identifier = /^(?:_|\p{XID_Start})\p{XID_Continue}*$/u;

export function initializerArgumentsValid(module: Module): boolean {
  const entry = module.initializerArguments;
  if (entry == null) return true;
  if (!module.constructorClassVerified) return false;
  const positional = entry.positional ?? [], keywordOnly = entry.keywordOnly ?? [], only = entry.positionalOnly ?? 0;
  const names = [...positional, ...keywordOnly].map(item => item.name);
  if (names.length > 256 || new Set(names).size !== names.length || names.some(name => !identifier.test(name))
    || only > positional.length || Object.keys(entry.varKwargs ?? {}).some(name => names.slice(only).includes(name))) return false;
  if (Object.keys(entry.varKwargs ?? {}).some(name => /[\uD800-\uDFFF]/u.test(name))) return false;
  if ((entry.status === 'observed') !== (entry.reason == null)) return false;
  if (entry.status === 'unavailable' && (names.length || only || entry.varArgs != null || entry.varKwargs != null)) return false;
  const values = [...positional, ...keywordOnly].map(item => item.value).concat(entry.varArgs ?? [], Object.values(entry.varKwargs ?? {}));
  return values.every(value => value.kind === 'opaque' ? value.typeName === null
    : value.kind !== 'float' || !Object.is(value.value, -0));
}

/** The source proof must already exclude argument expansion at the call. */
export function initializerSourceArgument(entry: Module['initializerArguments'], path: Array<string | number>) {
  if (!entry || entry.status !== 'observed' || path.length !== 2) return undefined;
  const [group, slot] = path;
  let value;
  if (group === 'args' && typeof slot === 'number' && Number.isInteger(slot) && slot >= 0) {
    const positional = entry.positional ?? [];
    value = slot < positional.length ? positional[slot].value : entry.varArgs?.[slot - positional.length];
  } else if (group === 'kwargs' && typeof slot === 'string') {
    value = [...(entry.positional ?? []).slice(entry.positionalOnly ?? 0), ...(entry.keywordOnly ?? [])]
      .find(parameter => parameter.name === slot)?.value;
    if (value === undefined && Object.hasOwn(entry.varKwargs ?? {}, slot)) value = entry.varKwargs![slot];
  }
  return value?.kind === 'opaque' ? undefined : value;
}

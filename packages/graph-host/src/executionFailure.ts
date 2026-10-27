import type { GraphFailure, ValidationRequirement } from '@tensorviz/contracts/protocol';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';

type Diagnostic = components['schemas']['Diagnostic'];

/** A failing forward is evidence about the model, not permission to replace inputs. */
export function executionFailure(diagnostic?: Diagnostic): GraphFailure {
  if (!diagnostic) return { kind: 'capture' };
  const { code, phase } = diagnostic;
  if (code === 'missing_dependency') {
    const missingModules = diagnostic.names.filter(name => /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name)).slice(0, 32);
    return { kind: 'environment', ...(missingModules.length ? { missingModules } : {}) };
  }
  if (code === 'unresolved_import' || phase === 'import' && code !== 'invalid_source') return { kind: 'environment', importFailure: true };
  if (['unavailable_environment', 'unavailable_device'].includes(code) || phase === 'environment') return { kind: 'environment' };
  if (code === 'invalid_source' || code === 'unsaved' || phase === 'parse' || phase === 'index') return { kind: 'source' };
  if (phase === 'forward') return { kind: 'execution' };
  if (phase === 'binding' || phase === 'configuration' || phase === 'materialization'
    || code === 'invalid_factory' || code === 'missing_file') {
    const binding = diagnostic.field[0];
    const names = diagnostic.names.filter(name => /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name)).slice(0, 64);
    return { kind: 'setup', ...(names.length && ['init', 'inputs', 'factory'].includes(String(binding))
      ? { fields: names.map(name => `${binding}.${name}`) } : {}) };
  }
  if (phase === 'construction') return { kind: 'source' };
  return { kind: 'capture' };
}

export function profileFailure(requirements: readonly ValidationRequirement[]): GraphFailure {
  if (requirements.some(entry => entry.code === 'environment')) return { kind: 'environment' };
  if (requirements.some(entry => entry.code === 'unsaved' || entry.code === 'invalid_source')) return { kind: 'source' };
  return { kind: 'setup' };
}

import type { components } from '@tensorviz/contracts/semanticContracts.generated';
type Diagnostic = components['schemas']['Diagnostic'];

/** Fixed messages and bounded structural facts; never render Python exception prose. */
export function executionDiagnosticDetail(diagnostic: Diagnostic | undefined, cls: string, layer?: string | null): string | undefined {
  if (!diagnostic) return undefined;
  const failingLayer = layer && /^[A-Za-z_][\w.]{0,127}$/.test(layer) ? layer : undefined;
  const names = (diagnostic.names ?? []).filter(name => /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name)).slice(0, 8);
  const named = names.length ? ` (${names.join(', ').slice(0, 180)})` : '';
  if (diagnostic.phase === 'binding') {
    const inputs = diagnostic.field?.[0] === 'inputs' || diagnostic.remedy === 'provide_inputs';
    const subject = inputs ? 'inputs' : diagnostic.field?.[0] === 'factory' ? 'factory arguments' : 'constructor arguments';
    const problem = diagnostic.code === 'missing_argument' ? 'Missing arguments'
      : diagnostic.code === 'unexpected_argument' ? 'Unexpected arguments'
        : diagnostic.code === 'invalid_binding' ? 'Conflicting or excess positional arguments' : 'Unknown Python signature';
    return `${cls} needs explicit ${subject}. ${problem}${named}. `
      + 'Declare the required args and kwargs in a tensorviz.toml model profile, or provide explicit MCP inputs before validating again.';
  }
  if (diagnostic.code === 'missing_dependency') {
    return `${cls} could not import a required dependency${named}. `
      + "Install the missing dependency in TensorViz's selected Python environment, then retry.";
  }
  if (diagnostic.code === 'unresolved_import') return `${cls} could not resolve an import${named}. Check the project's required library versions and import paths in TensorViz's selected Python environment, then retry.`;
  if (diagnostic.phase === 'import') return `${cls} failed while loading its model or package initialization code${named}. Check library-version compatibility and import-time model code in TensorViz's selected Python environment, then retry.`;
  if (diagnostic.code === 'unavailable_device') return `${cls} cannot use the configured device in the selected Python environment. Check the profile device and environment.`;
  if (diagnostic.code === 'invalid_factory') return `${cls} requires a factory that returns the exact selected model class. Check the factory in tensorviz.toml.`;
  if (diagnostic.code === 'missing_file') return `${cls} cannot find a required setup file. Check the configured factory, fixture and checkpoint paths in the workspace.`;
  if (diagnostic.phase === 'materialization') return `${cls} could not prepare the configured inputs, state or device. Check the explicit profile values and tensorViz Output in VS Code.`;
  if (diagnostic.phase === 'forward') {
    const where = failingLayer ? ` at ${failingLayer}` : '';
    const expected = diagnostic.expectedShape?.map(size => size ?? '?').join(', ');
    const actual = diagnostic.actualShape?.join(', ');
    if (diagnostic.code === 'invalid_shape' && expected && actual) return `${cls} reached an incompatible tensor shape${where}: expected [${expected}], received [${actual}]. Inspect the failing operation and its inputs.`;
    if (diagnostic.code === 'invalid_dtype') return `${cls} reached an incompatible tensor dtype${where}. Inspect the failing operation and its inputs.`;
    if (diagnostic.code === 'assertion_failed') return `${cls} failed an assertion during the forward pass${where}. Check the explicit inputs and the failure in tensorViz Output in VS Code.`;
    return `${cls} failed during the forward pass${where}. Inspect the captured failure and tensorViz Output in VS Code.`;
  }
  return `${cls} could not complete its configured setup. Check the model profile and tensorViz Output in VS Code.`;
}

import type { GraphOpeningState } from './protocol';

export interface GraphMcpHelpContext {
  opening: GraphOpeningState;
  interpreter?: string;
  environmentSource?: string;
}

/** A copyable investigation request; producing it never calls or messages an agent. */
export function graphMcpHelpRequest({ opening, interpreter, environmentSource }: GraphMcpHelpContext): string {
  const { target } = opening;
  const selected = opening.environment?.candidates.find(candidate => candidate.id === opening.environment?.selectedId);
  const facts = {
    openingId: opening.id,
    target: { workspaceFolder: target.workspaceFolder, file: target.file, cls: target.cls,
      sourceDigest: target.sourceDigest, documentVersion: target.documentVersion, backendRevision: target.backendRevision },
    phase: opening.phase,
    failure: opening.failure,
    detail: opening.detail.slice(0, 4000),
    hasCapture: opening.hasCapture,
    ...(opening.environment ? { environmentRecovery: { reason: opening.environment.reason,
      missingModules: opening.environment.missingModules, requirements: opening.environment.requirements,
      compatibilityPackages: opening.environment.compatibilityPackages } } : {}),
    ...(selected ? { selectedEnvironment: { interpreter: selected.interpreter, source: selected.source,
      python: selected.python, torch: selected.torch, fingerprint: selected.fingerprint } } : {}),
    ...(interpreter ? { reportedEnvironment: { interpreter, source: environmentSource } } : {}),
  };
  return 'Investigate this TensorViz opening using MCP directly. This is the snapshot when help was opened; verify the current model, source and environment before acting.\n\n'
    + JSON.stringify(facts, null, 2)
    + '\n\nUse get_execution_environment for this model file and inspect the opening/validation evidence. Distinguish missing libraries, incompatible packages and model/tool bugs. Reuse the saved tensorviz.toml recipe and the selected capture Python. Request native approval for any dependency installation or source change; preserve existing Torch/CUDA. Do not retry unchanged failures or claim retained/source-only data is a successful new capture. Continue until the requested capture succeeds, or report a concrete blocker with evidence and the next step. Explain the cause and what was verified.';
}

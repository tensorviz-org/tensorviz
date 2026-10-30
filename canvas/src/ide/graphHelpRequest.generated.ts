/* GENERATED from packages/contracts/src/graphHelpRequest.ts; npm run protocol:sync. */
import type { GraphOpeningState } from './protocol.generated';

export interface GraphMcpHelpContext {
  opening: GraphOpeningState;
  interpreter?: string;
  environmentSource?: string;
}

/** A copyable investigation request; producing it never calls or messages an agent. */
export function graphMcpHelpRequest({ opening, interpreter, environmentSource }: GraphMcpHelpContext): string {
  const { target } = opening;
  const setup = opening.phase === 'needs_setup' && (!opening.environment || opening.environment.reason === 'ready');
  const selected = opening.environment?.candidates.find(candidate => candidate.id === opening.environment?.selectedId);
  const facts = {
    openingId: opening.id,
    target: { workspaceFolder: target.workspaceFolder, file: target.file, cls: target.cls,
      sourceDigest: target.sourceDigest, documentVersion: target.documentVersion, backendRevision: target.backendRevision },
    phase: opening.phase,
    failure: opening.failure,
    detail: opening.detail.slice(0, 4000),
    hasCapture: opening.hasCapture,
    ...(setup && opening.draft ? { draft: opening.draft } : {}),
    ...(opening.environment ? { environmentRecovery: { reason: opening.environment.reason,
      missingModules: opening.environment.missingModules, requirements: opening.environment.requirements,
      compatibilityPackages: opening.environment.compatibilityPackages } } : {}),
    ...(selected ? { selectedEnvironment: { interpreter: selected.interpreter, source: selected.source,
      python: selected.python, torch: selected.torch, fingerprint: selected.fingerprint } } : {}),
    ...(interpreter ? { reportedEnvironment: { interpreter, source: environmentSource } } : {}),
  };
  if (setup) return 'Continue this TensorViz input setup using MCP directly. This pending draft contains values entered or selected in Canvas; it has not been saved or validated by asking for help.\n\n'
    + JSON.stringify(facts, null, 2)
    + '\n\nCall get_graph_setup first and verify the openingId, model and sourceDigest. For this same opening, use suggest_graph_inputs to read source constraints while preserving the current draft, profileName, constructor settings, shapes, dtypes and generators. Do not start a new open_graph request merely to read this setup. The current host draft takes precedence if the user edited it after this snapshot. Fill missing values, explain any unresolved choices, then call configure_graph_setup with the current openingId and sourceDigest and poll get_graph_setup. If the opening changed, reconcile with the user before reusing this draft. Continue until capture succeeds or report a concrete blocker; asking for help does not itself save or execute.';
  return 'Investigate this TensorViz opening using MCP directly. This is the snapshot when help was opened; verify the current model, source and environment before acting.\n\n'
    + JSON.stringify(facts, null, 2)
    + '\n\nUse get_execution_environment for this model file and inspect the opening/validation evidence. Distinguish missing libraries, incompatible packages and model/tool bugs. Reuse the saved tensorviz.toml recipe and the selected capture Python. Request native approval for any dependency installation or source change; preserve existing Torch/CUDA. Do not retry unchanged failures or claim retained/source-only data is a successful new capture. Continue until the requested capture succeeds, or report a concrete blocker with evidence and the next step. Explain the cause and what was verified.';
}

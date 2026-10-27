import { ApiService, UnsupportedModule } from './apiService';
import { graphDocumentV2ToModelGraph } from './graphDocumentAdapter';
import { parseInputShape } from '../utils/inputShape';
import {
  describeMissingParameters,
  findMissingRequiredParameters,
} from '../utils/requiredParameters';
import { ModelGraph, SavedGraph } from '../types';
import * as hostServices from '../ide/hostServices';

/**
 * Export a saved project as a single runnable model.py, from the list pages.
 *
 * The builder already exports, but only for the graph currently open on the
 * canvas: to get the code for any other model you had to open it first. This is
 * the same pipeline reached from a project row.
 *
 * The order of the gates is the point. /api/models/validate is a structural
 * check — it never reads input_shape and the endpoint drops its warnings, so a
 * graph with no Input node validates clean (backend/utils/model_validation.py,
 * server_v2.py:445). Codegen then needs exactly that Input node for its shape. So
 * the client preflight runs first and refuses on its own terms; validate is the
 * second gate, not the only one.
 */

export type ExportCodeResult =
  | {
      ok: true;
      fileName: string;
      /**
       * Modules whose bodies are not in the graph. The file still downloads —
       * it carries the backend's own `# INCOMPLETE EXPORT` banner — but the
       * caller has to say so rather than report a clean export.
       */
      unsupported: UnsupportedModule[];
    }
  | { ok: false; message: string };

/**
 * The class the file declares, and the file's own name. The backend sanitises
 * `class_name` again on its side (server_v2.py:491); this copy exists so the
 * downloaded filename matches the class the user will read inside it.
 */
export function pythonClassName(name: string): string {
  return (
    name
      .replace(/[^a-zA-Z0-9]/g, '')
      .replace(/^[0-9]/, 'Model$&') || 'GeneratedModel'
  );
}

/**
 * The graph the backend is given.
 *
 * `document` is the authoritative saved representation and `graph` its derived
 * canvas projection, so the document wins where there is one — the same
 * precedence graphMetrics uses. Module wrapper nodes and reconciler-synthesised
 * edges are display structure, and sending them is a stated correctness bug, not
 * an inefficiency: they are what ModelBuilder's buildVerificationPayload strips
 * before every verify and export. This duplicates that stripping because the
 * builder's copy closes over canvas refs and cannot be called from here.
 */
export function exportPayload(project: Pick<SavedGraph, 'graph' | 'document'>): ModelGraph {
  const source: ModelGraph = project.document
    ? graphDocumentV2ToModelGraph(project.document)
    : project.graph;
  const nodes = (source?.nodes || []).filter((node) => node.type !== 'moduleNode');
  const kept = new Set(nodes.map((node) => node.id));
  const edges = (source?.edges || []).filter(
    (edge) =>
      !(edge as { data?: { synthetic?: boolean } }).data?.synthetic &&
      kept.has(edge.source) &&
      kept.has(edge.target)
  );
  return { nodes, edges };
}

/** The anchor-and-object-URL dance moved into hostServices, which keeps it
 *  verbatim for the browser and posts the bytes to the extension host in the
 *  IDE, where `a[download]` does nothing at all. */
function triggerDownload(payload: Blob, fileName: string): void {
  hostServices.saveFile(fileName, payload);
}

/** The first two validation errors, so a failure names what broke without
 *  becoming a wall of text in a toast. */
function summariseErrors(errors: Array<{ message: string }>): string {
  const messages = errors.map((error) => error.message).filter(Boolean);
  if (messages.length === 0) return 'the graph did not pass validation.';
  const head = messages.slice(0, 2).join('; ');
  return messages.length > 2 ? `${head} (+${messages.length - 2} more)` : head;
}

type ExportPlan = { graph: ModelGraph; inputShape: number[]; className: string };

/**
 * Everything both formats must clear before anything is fetched. Sharing it is
 * the point: a project that refuses to export as a file must refuse to export
 * as an archive for the same stated reason, rather than the two formats
 * disagreeing about whether the graph is sound.
 */
async function planExport(
  project: Pick<SavedGraph, 'name' | 'graph' | 'document'>
): Promise<{ ok: true; plan: ExportPlan } | { ok: false; message: string }> {
  const className = pythonClassName(project.name);
  const graph = exportPayload(project);

  if (graph.nodes.length === 0) {
    return { ok: false, message: 'This project has no layers to export.' };
  }

  // Preflight, in the builder's words — see CodeExportDialog. Exporting a graph
  // whose layers are missing required parameters emits `nn.Linear()`, which
  // raises TypeError on the first line that uses it.
  const inputNode = graph.nodes.find((node) => node.data?.layerType?.id === 'input');
  if (!inputNode || !inputNode.data.parameters?.shape) {
    return {
      ok: false,
      message: 'No Input node with a shape — open the project and add one before exporting.',
    };
  }
  const incomplete = findMissingRequiredParameters(graph.nodes as any);
  if (incomplete.length) {
    return {
      ok: false,
      message:
        `${incomplete.length} layer${incomplete.length > 1 ? 's are' : ' is'} missing required `
        + `parameters: ${describeMissingParameters(incomplete)}.`,
    };
  }
  const inputShape = parseInputShape(inputNode.data.parameters.shape);

  let validation;
  try {
    validation = await ApiService.validateModel(graph, inputShape);
  } catch (error) {
    // Fail loud, and fail differently: a backend that never answered is not a
    // model that failed validation, and telling the user their graph is broken
    // when the server is down sends them to fix the wrong thing.
    return {
      ok: false,
      message: `Couldn't validate — ${error instanceof Error ? error.message : 'the backend did not respond'}.`,
    };
  }
  if (!validation.isValid) {
    return { ok: false, message: `Validation failed — ${summariseErrors(validation.errors)}` };
  }

  // Structural validation is not enough, and this is the gate that was missing.
  // /api/models/export-package refuses a model that does not run, but it does so
  // in the BACKEND — so "Export as zip" was safe while "Export as Python", one
  // menu row below it, called only generate-code and handed the user a file that
  // raised `mat1 and mat2 shapes cannot be multiplied` on first use, under a
  // green "downloaded" banner. Two adjacent menu items with opposite guarantees.
  //
  // The gate belongs here, in the shared preflight, rather than on the Python
  // route alone: that is what stops the two exports diverging again the next time
  // one of them grows a step. A model that cannot complete a forward pass is not
  // exportable in either shape.
  let verification;
  try {
    verification = await ApiService.verifyModel(graph, inputShape);
  } catch (error) {
    return {
      ok: false,
      message: `Couldn't verify — ${error instanceof Error ? error.message : 'the backend did not respond'}.`,
    };
  }
  if (!verification.ok) {
    const first = verification.errors?.[0]?.message;
    return {
      ok: false,
      message: first
        ? `This model does not run, so there is no code to give you: ${first}`
        : `This model does not run, so there is no code to give you (failed at the ${verification.stage} stage).`,
    };
  }

  return { ok: true, plan: { graph, inputShape, className } };
}

export async function exportProjectAsPython(
  project: Pick<SavedGraph, 'name' | 'graph' | 'document'>
): Promise<ExportCodeResult> {
  const prepared = await planExport(project);
  if (!prepared.ok) return prepared;
  const { graph, inputShape, className } = prepared.plan;

  try {
    const response = await ApiService.generateCode({
      graph,
      input_shape: inputShape,
      class_name: className,
    });
    if (!response.code) {
      return { ok: false, message: 'The backend returned no code for this graph.' };
    }
    const fileName = `${className}.py`;
    triggerDownload(new Blob([response.code], { type: 'text/x-python' }), fileName);
    return { ok: true, fileName, unsupported: response.unsupported || [] };
  } catch (error) {
    return {
      ok: false,
      message: `Export failed — ${error instanceof Error ? error.message : 'code generation did not complete'}.`,
    };
  }
}

/**
 * The runnable package: model.py, __init__.py, sample_run.py, README.md and
 * requirements.txt, zipped by the backend.
 *
 * Two requests, deliberately. /api/models/export-package returns the archive
 * and nothing else — it never calls unreproducible_modules, so the only trace
 * of a partial reconstruction is the "# INCOMPLETE EXPORT" banner buried inside
 * model.py. generate-code is asked the same question purely for its
 * `unsupported` list, so a zip warns about unreproduced modules exactly as the
 * single file does instead of downloading a partial model in silence. They run
 * together and both must succeed: the two endpoints share one generator, so
 * codegen failing means the model.py inside the archive is not to be trusted
 * either.
 *
 * Note export-package is authenticated where generate-code is not, so this is
 * the one export that can fail with a 401 after validation passed.
 */
export async function exportProjectAsZip(
  project: Pick<SavedGraph, 'name' | 'graph' | 'document'>
): Promise<ExportCodeResult> {
  const prepared = await planExport(project);
  if (!prepared.ok) return prepared;
  const { graph, inputShape, className } = prepared.plan;

  try {
    const [archive, generated] = await Promise.all([
      ApiService.exportPackage(graph, inputShape, className),
      ApiService.generateCode({ graph, input_shape: inputShape, class_name: className }),
    ]);
    const fileName = `${className}.zip`;
    triggerDownload(archive, fileName);
    return { ok: true, fileName, unsupported: generated.unsupported || [] };
  } catch (error) {
    return {
      ok: false,
      message: `Export failed — ${error instanceof Error ? error.message : 'the package was not built'}.`,
    };
  }
}

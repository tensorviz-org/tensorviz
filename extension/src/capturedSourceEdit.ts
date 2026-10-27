/** Host-only bounded planning. Credentials, saved source and the original capture
 * never cross to the browser/MCP; only the exact reviewed artifact can be applied. */
import Ajv from 'ajv';
import { isDeepStrictEqual } from 'node:util';
import schema from '@tensorviz/contracts/semanticSchema.generated';
import type { SourceEditCapability, CapturedSourceEdit } from '@tensorviz/contracts/protocol';
import { ACTIVATION_NAMES } from '@tensorviz/contracts/protocol';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { SourceEdit } from './apply';

const ajv = new Ajv({ strict: false });
const activationClasses = { relu: 'ReLU', gelu: 'GELU', silu: 'SiLU', sigmoid: 'Sigmoid', tanh: 'Tanh' } as const;
ajv.addSchema(schema, 'semantic');
export const isSourceEditCapability = ajv.compile<SourceEditCapability>({ anyOf: [
  { $ref: 'semantic#/definitions/LiteralEditCapability' }, { $ref: 'semantic#/definitions/ActivationEditCapability' },
  { $ref: 'semantic#/definitions/ConstructorLiteralCapability' },
  { $ref: 'semantic#/definitions/ConfigurationLiteralCapability' },
  { $ref: 'semantic#/definitions/ReconnectArgumentCapability' },
  { $ref: 'semantic#/definitions/InsertUnaryCapability' },
  { $ref: 'semantic#/definitions/DeleteUnaryCapability' },
] });
export class CapturedEditError extends Error {}
export interface CapturedSourcePlan {
  file: string; expected_digest: string; source_digest: string; edits: SourceEdit[];
  capability: SourceEditCapability; validation: 'static_only';
}

export async function prepareCapturedSourceEdit(capture: CaptureSnapshot, sources: Record<string, string>,
  intent: CapturedSourceEdit, capability: SourceEditCapability, options: {
    origin: string; token: string; signal: AbortSignal; fetchImpl?: typeof fetch;
  }): Promise<CapturedSourcePlan> {
  if (intent.kind !== capability.command) throw new CapturedEditError('The requested edit does not match its capability.');
  if (intent.kind === 'replace_activation' && capability.command === 'replace_activation' &&
    !(capability.compatibleActivations ?? ACTIVATION_NAMES).includes(intent.value)) {
    throw new CapturedEditError('This activation would not preserve the authored options.');
  }
  const replacement = intent.kind === 'reconnect_argument' && capability.command === 'reconnect_argument'
    ? capability.choices.find(choice => isDeepStrictEqual(choice.value, intent.value)) : undefined;
  if (intent.kind === 'reconnect_argument' && !replacement) {
    throw new CapturedEditError('This tensor is not a proven available value for the selected operand.');
  }
  if (intent.kind === 'delete_unary_op' && capability.command === 'delete_unary_op' &&
    !isDeepStrictEqual(intent.value, capability.input.value)) {
    throw new CapturedEditError('Deletion must bypass the activation to its proven original input.');
  }
  const body = JSON.stringify({ capture, sources, intent });
  if (Buffer.byteLength(body) > 24 * 1024 * 1024) throw new CapturedEditError('This capture and source exceed the source-proposal preparation limit.');
  const controller = new AbortController();
  let rejectAbort!: (error: Error) => void;
  const cancelled = new Promise<never>((_resolve, reject) => { rejectAbort = reject; });
  const abort = () => { controller.abort(); rejectAbort(new CapturedEditError('Source proposal preparation was cancelled or timed out.')); };
  options.signal.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(abort, 10_000);
  const request = async () => {
    controller.signal.throwIfAborted();
    const response = await (options.fetchImpl ?? fetch)(`${options.origin}/api/ide/capture/edit`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': options.token },
      redirect: 'error', body, signal: controller.signal,
    });
    if (!response.ok) {
      await response.body?.cancel();
      throw new CapturedEditError(response.status === 413 ? 'This source proposal exceeds the preparation limit.'
        : 'The saved source no longer supports this exact edit capability. Refresh the graph and inspect its available edits.');
    }
    const reader = response.body?.getReader();
    if (!reader) throw new CapturedEditError('The backend returned no source proposal.');
    const chunks: Uint8Array[] = []; let size = 0;
    try {
      for (;;) {
        const next = await reader.read();
        if (next.done) break;
        size += next.value.byteLength;
        if (size > 64 * 1024) throw new CapturedEditError('The source proposal exceeds the response limit.');
        chunks.push(next.value);
      }
    } finally { await reader.cancel(); }
    const plan = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))) as CapturedSourcePlan;
    const anchor = capability.binding.anchors[0];
    if (!plan || plan.validation !== 'static_only' || plan.file !== anchor.file || plan.expected_digest !== anchor.digest ||
      typeof plan.source_digest !== 'string' || !/^[0-9a-f]{64}$/.test(plan.source_digest) ||
      !isDeepStrictEqual(plan.capability, capability) || !Array.isArray(plan.edits)) throw new CapturedEditError('The backend returned a mismatched source proposal.');
    if (isDeepStrictEqual(intent.value, intent.expected)) {
      if (plan.edits.length !== 0 || plan.source_digest !== anchor.digest) throw new CapturedEditError('A no-op source proposal unexpectedly changed source.');
    } else {
      const edit = plan.edits[0];
      const importPlan = capability.command === 'replace_activation' || capability.command === 'insert_unary_op' ? capability.importPlan : null;
      if (plan.edits.length !== (importPlan ? 2 : 1) || !edit || !isDeepStrictEqual(edit.range,
        { start: [anchor.startLine, anchor.startColumn], end: [anchor.endLine, anchor.endColumn] }) ||
        typeof edit.newText !== 'string' || edit.newText.length > 4096) throw new CapturedEditError('The proposal changed a range outside its exact source binding.');
      if (intent.kind === 'set_literal' || intent.kind === 'set_constructor_literal' || intent.kind === 'set_configuration_literal') {
        const exact = intent.value.kind === 'int' ? edit.newText === intent.value.value :
          /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?(?![\s\S])/.test(edit.newText) &&
            /[.e]/.test(edit.newText) &&
            Object.is(Number(edit.newText), intent.value.value === '-0' ? -0 : intent.value.value);
        if (!exact) throw new CapturedEditError('The proposal did not contain the requested numeric literal.');
      } else if (intent.kind === 'reconnect_argument') {
        if (edit.newText !== replacement!.name) throw new CapturedEditError('The proposal did not contain the requested tensor name.');
      } else if (intent.kind === 'insert_unary_op') {
        if (capability.command !== 'insert_unary_op' || edit.newText !== `${capability.namespace}.${intent.value}(${capability.expectedName})`) {
          throw new CapturedEditError('The proposal did not wrap the exact operand in the requested activation.');
        }
      } else if (intent.kind === 'delete_unary_op') {
        if (capability.command !== 'delete_unary_op' || edit.newText !== capability.input.name) {
          throw new CapturedEditError('The proposal did not bypass the activation to its original input.');
        }
      } else if (capability.command !== 'replace_activation' || edit.newText !==
        `${capability.namespace}.${capability.recipe === 'module_activation_v1' ? activationClasses[intent.value] : intent.value}`) {
        throw new CapturedEditError('The proposal did not contain the requested activation.');
      }
      if (importPlan && (capability.command === 'replace_activation' || capability.command === 'insert_unary_op')) {
        const module = capability.recipe === 'module_activation_v1' ? 'torch.nn' : 'torch.nn.functional';
        if (importPlan.module !== module || !/^_tensorviz_[A-Za-z0-9_]+$/.test(capability.namespace) ||
          !Number.isSafeInteger(importPlan.line) || importPlan.line < 1 || importPlan.line > anchor.startLine ||
          !['\n', '\r\n'].includes(importPlan.lineEnding) || !isDeepStrictEqual(plan.edits[1], {
            range: { start: [importPlan.line, 0], end: [importPlan.line, 0] },
            newText: `import ${module} as ${capability.namespace}${importPlan.lineEnding}`,
          })) throw new CapturedEditError('The proposal changed its exact required Torch import.');
      }
    }
    controller.signal.throwIfAborted();
    return plan;
  };
  try {
    if (options.signal.aborted) abort();
    return await Promise.race([request(), cancelled]);
  } catch (error) {
    if (error instanceof CapturedEditError) throw error;
    throw new CapturedEditError('The backend could not prepare a valid captured source proposal. Refresh and try again.');
  } finally { clearTimeout(timer); options.signal.removeEventListener('abort', abort); }
}

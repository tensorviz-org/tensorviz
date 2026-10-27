/** Python owns the schema and full relational validation. This host boundary
 * checks structure, execution identity and bounded completeness before retention. */
import Ajv from 'ajv';
import { isDeepStrictEqual } from 'node:util';
import schema from '@tensorviz/contracts/semanticSchema.generated';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
import type { ResolvedExecutionPlan, CapturePreparation } from './executionPlanCore';
import { initializerArgumentsValid } from '@tensorviz/contracts/initializerArguments';

export type CaptureSnapshot = components['schemas']['CaptureSnapshot'];
type Argument = components['schemas']['Operation']['output'];
type Arguments = components['schemas']['Arguments'];
const ajv = new Ajv({ strict: false });
ajv.addSchema(schema, 'semantic');
const structure = ajv.compile<CaptureSnapshot>({ $ref: 'semantic#/definitions/CaptureSnapshot' });

export function semanticCapture(value: unknown, attemptId: string, setup: ResolvedExecutionPlan | CapturePreparation | undefined,
  setupDigest: string | null): CaptureSnapshot | undefined {
  try {
    if (!setup || !setupDigest || !structure(value)) return undefined;
    const capture = value;
    if (capture.backend === 'jaxpr') {
      if (!capture.jaxVersion || capture.torchVersion !== null || capture.coverage.scope !== 'traced_program'
        || !('execution' in setup) || setup.execution.framework !== 'jax') return undefined;
    } else if (!capture.torchVersion || capture.jaxVersion != null || capture.coverage.scope !== 'executed_path'
      || ('execution' in setup && setup.execution.framework === 'jax')) return undefined;
    const bindings = new Map(capture.sourceBindings.map(binding => [binding.id, binding]));
    if (capture.modules.some(module => {
      if (!initializerArgumentsValid(module)) return true;
      if (!module.constructorClassVerified) return false;
      const refs = module.constructorBindingIds ?? [], binding = bindings.get(refs[0]);
      return refs.length !== 1 || binding?.kind !== 'constructor' || binding.quality !== 'exact';
    })) return undefined;
    if ('policy' in setup) {
      if (capture.planDigest !== null || capture.preparationDigest !== setupDigest || !capture.automatic
        || capture.automatic.candidateIndex > capture.automatic.candidatesTried) return undefined;
    } else if (capture.planDigest !== setupDigest || capture.preparationDigest !== null || capture.automatic !== null) return undefined;
    if (capture.attemptId !== attemptId || setup.id !== attemptId
      || !isDeepStrictEqual(capture.source, setup.source) || !isDeepStrictEqual(capture.limits, setup.captureLimits)
      || Buffer.byteLength(JSON.stringify(capture)) > capture.limits.bytes
      || capture.operations.length > capture.limits.operations || capture.invocations.length > capture.limits.invocations
      || capture.values.length > capture.limits.values) return undefined;
    const calls = new Map(capture.invocations.map(call => [call.id, call]));
    const ops = new Map(capture.operations.map(op => [op.id, op]));
    const values = new Map(capture.values.map(fact => [fact.id, fact]));
    if (calls.size !== capture.invocations.length || ops.size !== capture.operations.length
      || values.size !== capture.values.length) return undefined;
    const root = capture.rootInvocationId === null ? undefined : calls.get(capture.rootInvocationId);
    if ((capture.rootInvocationId !== null && !root) || (root && (root.parentId !== null
      || !isDeepStrictEqual(root.arguments, capture.inputs) || !isDeepStrictEqual(root.output, capture.output)))) return undefined;
    let opaque = false;
    function tree(argument: Argument, before: number, depth = 0, count = { nodes: 0 }): boolean {
      if (argument === null) return true;
      if (depth > 32 || ++count.nodes > 16384) return false;
      if (argument.kind === 'opaque') opaque = true;
      if (argument.kind === 'float' && Object.is(argument.value, -0)) return false;
      if (argument.kind === 'value') {
        const fact = values.get(argument.id), producer = fact?.producer && ops.get(fact.producer.operationId);
        return !!fact && (!fact.producer || !!producer && producer.sequence < before);
      }
      if (argument.kind === 'list' || argument.kind === 'tuple') return argument.items.every(child => tree(child, before, depth + 1, count));
      if (argument.kind === 'dict') return argument.entries.every(entry => tree(entry.value, before, depth + 1, count));
      return true;
    }
    function args(arguments_: Arguments, before: number): boolean {
      const count = { nodes: 0 };
      return [...arguments_.args, ...Object.values(arguments_.kwargs)].every(argument => tree(argument, before, 0, count));
    }
    if (!args(capture.inputs, -1) || !tree(capture.output, Infinity)) return undefined;
    for (const call of capture.invocations) {
      if (!args(call.arguments, call.sequence) || !tree(call.output, call.endSequence ?? Infinity)
        || (call.parentId !== null && !calls.has(call.parentId))
        || (call.parentId === null && call.id !== capture.rootInvocationId)
        || (call.status === 'completed') !== (call.output !== null)) return undefined;
    }
    for (const op of capture.operations) {
      if (!args(op.arguments, op.sequence) || !tree(op.output, op.sequence + 1)
        || (op.invocationId !== null && !calls.has(op.invocationId))
        || (op.status === 'completed') !== (op.output !== null)) return undefined;
      for (const effect of op.effects) {
        if (!tree({ kind: 'value', id: effect.beforeId }, op.sequence) || !tree(effect.after, op.sequence + 1)) return undefined;
      }
    }
    const coverage = capture.coverage;
    if (coverage.operationsSeen < ops.size || coverage.invocationsSeen < calls.size) return undefined;
    if (capture.status === 'complete' && (!root || root.status !== 'completed' || opaque || coverage.omissions.length
      || coverage.operationsSeen !== ops.size || coverage.invocationsSeen !== calls.size
      || capture.operations.some(op => op.status === 'interrupted' || op.status === 'failed' && op.effects.length > 0)
      || capture.invocations.some(call => call.status === 'interrupted'))) return undefined;
    return capture;
  } catch { return undefined; }
}

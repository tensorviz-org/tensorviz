/** Verify private worker verdict provenance and its canonical witnesses.
 * Python owns the source AST, lexical flow and full activation/alias proof; this is not a
 * second edit planner or a cross-run operation matcher. */
import { isDeepStrictEqual } from 'node:util';
import type { CandidateRequest, CandidateExecution } from './executionPlanCore';
import type { CaptureSnapshot } from './semanticCapture';
import type { CandidatePostconditions } from '@tensorviz/contracts/protocol';
import { initializerSourceArgument } from '@tensorviz/contracts/initializerArguments';

type Operation = CaptureSnapshot['operations'][number];
type Invocation = CaptureSnapshot['invocations'][number];
const classes = { relu: 'ReLU', gelu: 'GELU', silu: 'SiLU', sigmoid: 'Sigmoid', tanh: 'Tanh' } as const;
type Activation = keyof typeof classes;
const input = (call: Invocation) => call.arguments.args.length === 1 && !Object.keys(call.arguments.kwargs).length
  ? call.arguments.args[0] : !call.arguments.args.length && Object.keys(call.arguments.kwargs).join() === 'input'
    ? call.arguments.kwargs.input : undefined;

function activation(op: Operation, inplace: boolean): Activation | undefined {
  const name = (Object.keys(classes) as Activation[]).find(name => op.target === `aten.${name}${inplace ? '_' : ''}.default`);
  if (!name || op.status !== 'completed' || op.arguments.args.length !== 1 || op.arguments.args[0].kind !== 'value'
    || op.output?.kind !== 'value' || (inplace && name !== 'relu' && name !== 'silu')) return undefined;
  const kwargs = op.arguments.kwargs;
  if (Object.keys(kwargs).length && (name !== 'gelu' || !isDeepStrictEqual(kwargs, { approximate: { kind: 'str', value: 'none' } }))) return undefined;
  if (!inplace && op.effects.length) return undefined;
  if (inplace) {
    const writes = op.effects.filter(effect => effect.kind === 'write');
    if (writes.length !== 1 || writes[0].beforeId !== op.arguments.args[0].id || !isDeepStrictEqual(writes[0].after, op.output)
      || op.effects.some(effect => !isDeepStrictEqual(effect.argumentPath, ['args', 0]))) return undefined;
  }
  return name;
}

export function candidatePostconditionsMatch(evidence: CandidatePostconditions, request: CandidateRequest,
  execution: CandidateExecution, capture?: CaptureSnapshot): boolean {
  if (evidence.executionId !== request.id) return false;
  return sourcePostconditionsMatch(evidence, request, execution.afterDigest, capture);
}

/** Shared scalar/call witnesses for a copied candidate or a later live capture.
 * Callers separately establish source/dependency provenance and editor freshness. */
export function sourcePostconditionsMatch(evidence: CandidatePostconditions,
  request: Pick<CandidateRequest, 'intent' | 'capture' | 'reviewedFile'>,
  afterDigest: string, capture?: CaptureSnapshot): boolean {
  const intent = request.intent;
  const recipe = intent.kind === 'set_literal' ? 'binary_scalar_v1' : intent.kind === 'replace_activation'
    ? intent.target.kind === 'operation' ? 'functional_activation_v1' : 'module_activation_v1'
      : intent.kind === 'reconnect_argument' ? 'straight_line_tensor_v1'
        : intent.kind === 'insert_unary_op' ? 'straight_line_unary_insert_v1'
          : intent.kind === 'delete_unary_op' ? 'straight_line_unary_delete_v1'
            : intent.kind === 'set_constructor_literal' ? 'constructor_scalar_v1'
              : intent.kind === 'set_configuration_literal' ? 'configuration_scalar_v1' : null;
  if (evidence.recipe !== recipe
    || capture !== undefined && evidence.executionId !== capture.attemptId
    || evidence.captureId !== null && evidence.captureId !== capture?.id) return false;
  if (evidence.status === 'unavailable') return true;
  if (!capture || capture.status !== 'complete' || capture.backend !== 'runtime'
    || (intent.kind !== 'set_literal' && intent.kind !== 'replace_activation'
      && intent.kind !== 'reconnect_argument' && intent.kind !== 'insert_unary_op' && intent.kind !== 'delete_unary_op'
      && intent.kind !== 'set_constructor_literal' && intent.kind !== 'set_configuration_literal')) return false;
  const moduleRecipe = recipe === 'module_activation_v1', deletionRecipe = recipe === 'straight_line_unary_delete_v1';
  const constructorRecipe = recipe === 'constructor_scalar_v1', configurationRecipe = recipe === 'configuration_scalar_v1';
  const instanceRecipe = moduleRecipe || constructorRecipe || configurationRecipe;
  const binding = capture.sourceBindings.find(item => item.id === evidence.bindingId);
  if (!binding || binding.kind !== (deletionRecipe ? 'definition' : instanceRecipe ? 'constructor' : 'call') || binding.quality !== 'exact'
    || binding.anchors.length !== 1 || binding.anchors[0].file !== request.reviewedFile
    || binding.anchors[0].digest !== afterDigest) return false;
  const baseCall = instanceRecipe ? request.capture.invocations.find(item => item.id === intent.target.semanticId) : undefined;
  const baseModule = baseCall ? request.capture.modules.find(item => item.id === baseCall.moduleId) : undefined;
  const baseOperations = moduleRecipe ? request.capture.operations.filter(item => item.invocationId === baseCall?.id)
    : request.capture.operations.filter(item => item.id === intent.target.semanticId);
  if (!constructorRecipe && !configurationRecipe && baseOperations.length !== 1) return false;
  const original = baseOperations[0], ids = instanceRecipe ? baseModule?.constructorBindingIds : original.sourceBindingIds;
  const originalBinding = ids?.length === 1 ? request.capture.sourceBindings.find(item => item.id === ids[0]) : undefined;
  if (!originalBinding || originalBinding.quality !== 'exact' || originalBinding.kind !== (deletionRecipe ? 'call' : binding.kind)) return false;
  const before = originalBinding.anchors[0], after = binding.anchors[0], shift = after.startLine - before.startLine;
  const allowsImport = recipe === 'functional_activation_v1' || moduleRecipe || recipe === 'straight_line_unary_insert_v1';
  // These recipes keep the expression's start column and scope. Activation
  // replacement may insert one reviewed import; Python verifies its exact text.
  if (after.file !== before.file || after.scope !== before.scope) return false;
  if (deletionRecipe) {
    // The name replacing the old call has no dispatcher event. Its exact
    // containing forward definition and completed calls hold the flow evidence.
    if (after.startLine > before.startLine || after.startLine === before.startLine && after.startColumn > before.startColumn
      || after.endLine < before.startLine || after.endLine === before.startLine && after.endColumn <= before.startColumn) return false;
  } else if (configurationRecipe) {
    // The definition may precede the constructor on the same line. The call's
    // tokens stay identical; Python proves its exact translated AST range.
    if (shift !== 0 || after.endLine !== before.endLine || after.tokenDigest !== before.tokenDigest) return false;
  } else if (after.startColumn !== before.startColumn || after.endLine - before.endLine !== shift
    || (shift !== 0 && (!allowsImport || shift !== 1))) return false;
  const location = ({ tokenDigest: _token, ...anchor }: typeof after) => anchor;
  if (capture.sourceBindings.filter(item => item.kind === binding.kind
    && item.anchors.some(anchor => isDeepStrictEqual(location(anchor), location(after)))).length !== 1) return false;
  if (intent.kind === 'set_configuration_literal') {
    const uses = evidence.configurationUses ?? [];
    if (!uses.some(use => use.bindingId === binding.id && isDeepStrictEqual(use.sourceArgumentPath, intent.sourceArgumentPath))) return false;
    const instances = new Set<string>();
    let matches = true;
    for (const use of uses) {
      const witness = capture.sourceBindings.find(item => item.id === use.bindingId);
      if (!witness || witness.kind !== 'constructor' || witness.quality !== 'exact' || witness.anchors.length !== 1) return false;
      const anchor = witness.anchors[0];
      if (anchor.file !== request.reviewedFile || anchor.digest !== afterDigest
        || capture.sourceBindings.filter(item => item.kind === 'constructor'
          && item.anchors.some(other => isDeepStrictEqual(location(other), location(anchor)))).length !== 1) return false;
      const modules = capture.modules.filter(module => module.constructorBindingIds?.includes(witness.id));
      if (!modules.length || modules.some(module => module.constructorBindingIds?.length !== 1 || !module.constructorClassVerified)) return false;
      for (const module of modules) {
        const value = initializerSourceArgument(module.initializerArguments, use.sourceArgumentPath);
        if (value === undefined) return false;
        instances.add(module.id);
        matches = matches && isDeepStrictEqual(value, intent.value);
      }
    }
    return instances.size === evidence.observedInitializers && (evidence.status === 'verified') === matches;
  }
  if (intent.kind === 'set_constructor_literal') {
    const modules = capture.modules.filter(module => module.constructorBindingIds?.includes(binding.id));
    if (!modules.length || modules.length !== evidence.observedInitializers
      || modules.some(module => module.constructorBindingIds?.length !== 1 || !module.constructorClassVerified)) return false;
    const values = modules.map(module => initializerSourceArgument(module.initializerArguments, intent.sourceArgumentPath));
    if (values.some(value => value === undefined)) return false;
    return (evidence.status === 'verified') === values.every(value => isDeepStrictEqual(value, intent.value));
  }
  if (deletionRecipe) {
    const definitions = capture.definitions.filter(item => item.sourceBindingIds.includes(binding.id));
    if (!definitions.length || definitions.some(item => item.sourceBindingIds.length !== 1)) return false;
    const ids = new Set(definitions.map(item => item.id));
    const modules = new Set(capture.modules.filter(item => ids.has(item.definitionId)).map(item => item.id));
    const calls = capture.invocations.filter(item => modules.has(item.moduleId));
    return calls.length === evidence.observedOperations && calls.every(call => call.status === 'completed');
  }
  const operations = capture.operations.filter(item => item.sourceBindingIds.includes(binding.id));
  if (intent.kind === 'reconnect_argument' || intent.kind === 'insert_unary_op') {
    // ValueRefs are execution-local. The Python source-flow proof resolves the
    // reviewed name and inserted call in each candidate invocation; old IDs cannot check its
    // verdict. Validate provenance and the retained operand occurrences here.
    return operations.length === evidence.observedOperations && operations.every(op => op.status === 'completed'
      && op.invocationId !== null && !op.effects.length && op.sourceBindingIds.length === 1
      && op.target === original.target && isDeepStrictEqual(op.arguments.kwargs, original.arguments.kwargs)
      && op.arguments.args.length === 2 && op.arguments.args[intent.argumentPath[1] as number]?.kind === 'value'
      && op.output?.kind === 'value');
  }
  let matches: boolean;
  if (intent.kind === 'set_literal') {
    if (operations.length !== evidence.observedOperations || operations.some(item => item.status !== 'completed'
      || item.effects.length || item.sourceBindingIds.length !== 1 || item.arguments.args.length !== 2
      || item.target !== original.target || !isDeepStrictEqual(item.arguments.kwargs, original.arguments.kwargs)
      || item.arguments.args[0].kind !== 'value' || !['int', 'float'].includes(item.arguments.args[1].kind))) return false;
    matches = operations.every(item => isDeepStrictEqual(item.arguments.args[1], intent.value));
  } else {
    const inplace = original.target === `aten.${intent.expected}_.default`;
    const names: Activation[] = [];
    if (!moduleRecipe) {
      for (const op of operations) {
        const name = activation(op, inplace);
        if (!name || op.sourceBindingIds.length !== 1) return false;
        names.push(name);
      }
    } else {
      const definitions = new Map(capture.definitions.map(item => [item.id, item]));
      const byModule = new Map<string, Invocation[]>(), byCall = new Map<string, Operation[]>();
      const parents = new Set(capture.invocations.map(call => call.parentId));
      for (const call of capture.invocations) {
        const group = byModule.get(call.moduleId) ?? []; group.push(call); byModule.set(call.moduleId, group);
      }
      for (const op of capture.operations) if (op.invocationId !== null) {
        const group = byCall.get(op.invocationId) ?? []; group.push(op); byCall.set(op.invocationId, group);
      }
      for (const module of capture.modules) {
        if (!module.constructorBindingIds?.includes(binding.id)) continue;
        if (module.constructorBindingIds.length !== 1) return false;
        const qualified = definitions.get(module.definitionId)?.qualifiedName;
        const name = (Object.keys(classes) as Activation[]).find(name => qualified === `torch.nn.modules.activation.${classes[name]}`);
        if (!name) return false;
        for (const call of byModule.get(module.id) ?? []) {
          const ops = byCall.get(call.id) ?? [], arg = input(call);
          if (call.status !== 'completed' || parents.has(call.id) || ops.length !== 1 || arg?.kind !== 'value'
            || activation(ops[0], inplace) !== name || !isDeepStrictEqual(ops[0].arguments.args[0], arg)
            || !isDeepStrictEqual(ops[0].output, call.output)) return false;
          names.push(name);
        }
      }
    }
    if (names.length !== evidence.observedOperations) return false;
    matches = names.every(name => name === intent.value);
  }
  return (evidence.status === 'verified') === matches;
}

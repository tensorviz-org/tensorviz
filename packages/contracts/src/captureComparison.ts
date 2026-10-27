/** Compare exact retained semantic facts under conservative role correspondence.
 * Consumers must not turn unknown correspondence into additions, removals or proof
 * of equality. This compares observed computation, not all possible executions. */
import type { components } from './semanticContracts.generated';
import { argumentStructure, captureJson, correspondCaptures, mapCaptureArgument,
  type CaptureCorrespondence, type CaptureEntityKind } from './captureCorrespondence';

type Capture = components['schemas']['CaptureSnapshot'];
type Argument = NonNullable<Capture['output']>;
type RecordKind = CaptureEntityKind | 'capture';
export interface CaptureFieldChange {
  field: string;
  before: { present: boolean; value?: unknown };
  after: { present: boolean; value?: unknown };
}
export interface CaptureRecordChange { kind: RecordKind; beforeId: string; afterId: string; fields: CaptureFieldChange[] }
export interface CaptureControl { field: string; status: 'same' | 'different' | 'unknown'; before: unknown; after: unknown }
export interface CapturedValueUse {
  kind: 'argument' | 'effect' | 'result';
  entity: { kind: 'operation' | 'output'; id: string };
  path: Array<string | number>; valueId: string;
}
export interface CapturedUseChange { before: CapturedValueUse | null; after: CapturedValueUse | null }
export interface SemanticCaptureComparison {
  correspondence: CaptureCorrespondence;
  changes: CaptureRecordChange[];
  observations: CaptureRecordChange[];
  unresolved: CaptureRecordChange[];
  controls: CaptureControl[];
  valueUses: { changes: CapturedUseChange[]; unresolved: CapturedUseChange[] };
  /** false means a known difference; null means equivalence cannot be established. */
  equivalent: boolean | null;
  coverageComplete: boolean;
}
interface Fact { normalized: unknown; value: unknown; available: boolean }
const plain = (value: unknown): Fact => ({ normalized: value, value, available: true });
const fields = (record: unknown): Record<string, unknown> => record as Record<string, unknown>;
const pick = (record: unknown, keys: string[]) => Object.fromEntries(keys.map(key => [key, fields(record)[key]]));
const sorted = <T>(items: T[]): T[] => [...items].sort((a, b) => { const left=captureJson(a),right=captureJson(b);return left<right?-1:left>right?1:0; });

export function compareCaptures(before: Capture, after: Capture): SemanticCaptureComparison {
  const correspondence = correspondCaptures(before, after);
  const makeMap = (side: 'beforeId' | 'afterId') => {
    const map = { module:new Map<string,string>(),invocation:new Map<string,string>(),operation:new Map<string,string>(),value:new Map<string,string>() };
    correspondence.pairs.forEach((pair, i) => map[pair.kind].set(pair[side], `${pair.kind}:${i}`)); return map;
  };
  const prepare = (capture: Capture, map: ReturnType<typeof makeMap>) => {
    const definitions = new Map(capture.definitions.map(definition => [definition.id, definition.qualifiedName]));
    const diagnostics = new Map(capture.diagnostics.map(diagnostic => [diagnostic.id, diagnostic]));
    let available = true;
    const ref = (kind: CaptureEntityKind, id: string | null): unknown => {
      if (id === null) return null;
      const role=map[kind].get(id);if (role===undefined) available=false;
      return role ?? {unresolved:kind};
    };
    const literal = (item: Argument): unknown => { if(item.kind==='opaque')available=false;return item; };
    const tree = (item: Argument | null) => {
      if(item===null)available=false; // No captured return is not a Python None literal.
      return mapCaptureArgument(item, id => ({kind:'value',role:ref('value',id)}), literal);
    };
    const args = (value: Capture['inputs']) => argumentStructure(value,id=>({kind:'value',role:ref('value',id)}),literal);
    const fact = (value: unknown, normalize: () => unknown): Fact => {
      available=true;const normalized=normalize();return {normalized,value,available};
    };
    const diagnostic = (id: string | null): unknown => {
      if (id===null) return null;
      // Source binding IDs belong to the capture; comparison uses typed error facts.
      const {id: ignored,sourceBindingId,...facts}=diagnostics.get(id)!;return facts;
    };
    const registration = () => sorted(capture.modules.map(module=>({paths:[...module.paths].sort(),definition:definitions.get(module.definitionId)})));
    const sharing = (field: 'objectId' | 'storageId') => {
      const groups=new Map<string,string[]>();let missing=false;
      for(const value of capture.values) {
        const key=value[field];if(key===null){missing=true;continue;}
        const group=groups.get(key);if(group)group.push(value.id);else groups.set(key,[value.id]);
      }
      const value=Array.from(groups.values());
      return fact(value,()=>{if(missing)available=false;return sorted(value.map(ids=>sorted(ids.map(id=>ref('value',id)))));});
    };
    const stateRegistration = () => sorted(capture.values.filter(value=>value.origin==='state')
      .map(value=>({names:sorted(value.state),logicalVersion:value.logicalVersion})));
    const captureFacts: Record<string,Fact> = {
      inputs:fact(capture.inputs,()=>args(capture.inputs)),output:fact(capture.output,()=>tree(capture.output)),
      moduleRegistrations:{...plain(registration()),available:capture.status==='complete'},
      observedStateRegistrations:{...plain(stateRegistration()),available:capture.status==='complete'},
      recordedCounts:{...plain({operations:capture.operations.length,invocations:capture.invocations.length,values:capture.values.length}),
        available:capture.status==='complete'},
      objectSharing:sharing('objectId'),storageSharing:sharing('storageId'),
    };
    const semantics = new Map<string,Record<string,Fact>>(), observations = new Map<string,Record<string,Fact>>();
    const record = (kind: RecordKind,id:string,semantic:Record<string,Fact>,observed:Record<string,Fact>) => {
      semantics.set(`${kind}:${id}`,semantic);observations.set(`${kind}:${id}`,observed);
    };
    record('capture',capture.id,captureFacts,{
      status:plain(capture.status), coverage:fact(capture.coverage,()=>({scope:capture.coverage.scope,
        operationsSeen:capture.coverage.operationsSeen,invocationsSeen:capture.coverage.invocationsSeen,
        omissions:sorted(capture.coverage.omissions),modules:sorted(capture.coverage.modules.map(module=>({...module,
          invocationIds:sorted(module.invocationIds.map(id=>ref('invocation',id)))})))})),
    });
    for(const module of capture.modules.filter(item=>map.module.has(item.id)))record('module',module.id,{definition:plain(definitions.get(module.definitionId)),paths:plain([...module.paths].sort())},
      module.initializerArguments == null ? {} : {initializerArguments:plain(module.initializerArguments)});
    for(const call of capture.invocations.filter(item=>map.invocation.has(item.id)))record('invocation',call.id,{
      module:fact(call.moduleId,()=>ref('module',call.moduleId)),parent:fact(call.parentId,()=>ref('invocation',call.parentId)),
      arguments:fact(call.arguments,()=>args(call.arguments)),output:fact(call.output,()=>tree(call.output)),
    },Object.fromEntries(Object.entries({...pick(call,['status','training','gradEnabled','parameterCount','parameterCountUnavailable']),
      diagnostic:diagnostic(call.diagnosticId)}).map(([key,value])=>[key,plain(value)])));
    for(const op of capture.operations.filter(item=>map.operation.has(item.id)))record('operation',op.id,{
      target:plain(op.target),schemaSignature:plain(op.schemaSignature),owner:fact(op.invocationId,()=>ref('invocation',op.invocationId)),
      arguments:fact(op.arguments,()=>args(op.arguments)),output:fact(op.output,()=>tree(op.output)),
      effects:fact(op.effects,()=>op.effects.map(effect=>({kind:effect.kind,argumentPath:effect.argumentPath,
        before:ref('value',effect.beforeId),after:tree(effect.after)}))),
    },{status:plain(op.status),diagnostic:plain(diagnostic(op.diagnosticId))});
    for(const value of capture.values.filter(item=>map.value.has(item.id)))record('value',value.id,{
      origin:plain(value.origin),state:plain(sorted(value.state)),logicalVersion:plain(value.logicalVersion),
      producer:fact(value.producer,()=>value.producer===null?null:{...value.producer,operationId:ref('operation',value.producer.operationId)}),
      aliasOf:fact(value.aliasOf,()=>ref('value',value.aliasOf)),previousVersion:fact(value.previousVersion,()=>ref('value',value.previousVersion)),
    },Object.fromEntries(Object.entries(pick(value,['shape','dtype','device','requiresGrad','stride','storageOffset','logicalBytes','version']))
      .map(([key,value])=>[key,plain(value)])));
    return {semantics,observations};
  };
  const left=prepare(before,makeMap('beforeId')),right=prepare(after,makeMap('afterId'));
  const changes:CaptureRecordChange[]=[],observations:CaptureRecordChange[]=[],unresolved:CaptureRecordChange[]=[];
  const present = (value: unknown) => value===undefined?{present:false}:{present:true,value};
  const compare = (kind:RecordKind,beforeId:string,afterId:string,a:Map<string,Record<string,Fact>>,b:Map<string,Record<string,Fact>>,into:CaptureRecordChange[]) => {
    const x=a.get(`${kind}:${beforeId}`)!,y=b.get(`${kind}:${afterId}`)!;
    const known:CaptureFieldChange[]=[],unknown:CaptureFieldChange[]=[];
    for(const field of Array.from(new Set([...Object.keys(x),...Object.keys(y)]))) {
      const l=x[field]??plain(undefined),r=y[field]??plain(undefined);
      const entry={field,before:present(l.value),after:present(r.value)};
      if(!l.available||!r.available)unknown.push(entry);
      else if(captureJson(l.normalized)!==captureJson(r.normalized))known.push(entry);
    }
    if(known.length)into.push({kind,beforeId,afterId,fields:known});
    if(unknown.length)unresolved.push({kind,beforeId,afterId,fields:unknown});
  };
  for(const pair of [{kind:'capture' as const,beforeId:before.id,afterId:after.id},...correspondence.pairs]) {
    compare(pair.kind,pair.beforeId,pair.afterId,left.semantics,right.semantics,changes);
    compare(pair.kind,pair.beforeId,pair.afterId,left.observations,right.observations,observations);
  }
  const controls = (capture:Capture):Record<string,unknown> => ({
    inputs:capture.inputDigest,weights:capture.stateDigest,environment:{digest:capture.source.environmentDigest,torch:capture.torchVersion,
      ...(capture.jaxVersion ? {jax:capture.jaxVersion} : {})},
    source:{className:capture.source.className,files:sorted(capture.source.files.map(file=>({role:file.role,digest:file.digest}))),
      modelDigest:capture.source.files.find(file=>file.path===capture.source.modelFile)!.digest,configDigest:capture.source.configDigest,
      dependencyScope:capture.source.dependencyScope},
    backend:{name:capture.backend,version:capture.backendVersion,normalizations:capture.normalizations},
    automaticSelection:capture.automatic,
  });
  const ca=controls(before),cb=controls(after);
  const controlChanges:CaptureControl[]=Object.keys(ca).map(field=>({field,
    status:(field==='inputs'||field==='weights')&&(ca[field]===null||cb[field]===null)?'unknown':captureJson(ca[field])===captureJson(cb[field])?'same':'different',before:ca[field],after:cb[field]}));
  const missing=Object.values(correspondence.unmatched.before).some(ids=>ids.length)||Object.values(correspondence.unmatched.after).some(ids=>ids.length);
  const coverageComplete=before.status==='complete'&&after.status==='complete';
  const valueUses = compareCapturedValueUses(before, after, correspondence);
  return {correspondence,changes,observations,unresolved,controls:controlChanges,coverageComplete,valueUses,
    equivalent:changes.length||valueUses.changes.length?false:missing||unresolved.length||!coverageComplete?null:true};
}

/** Computational connections are operand uses, including repeated operands and
 * non-returned writes. Invocation boundaries describe ownership, not extra ops. */
export function capturedValueUses(capture: Capture): CapturedValueUse[] {
  const uses: CapturedValueUse[] = [];
  const walk = (argument: Argument | null, entity: CapturedValueUse['entity'], path: Array<string | number>, kind: CapturedValueUse['kind']) => {
    if (argument === null) return;
    if (argument.kind === 'value') uses.push({ kind, entity, path, valueId: argument.id });
    else if (argument.kind === 'list' || argument.kind === 'tuple') argument.items.forEach((child, index) => walk(child, entity, [...path, 'items', index], kind));
    else if (argument.kind === 'dict') argument.entries.forEach((entry, index) => walk(entry.value, entity, [...path, 'entries', index, 'value'], kind));
  };
  for (const operation of capture.operations) {
    const entity = { kind: 'operation' as const, id: operation.id };
    operation.arguments.args.forEach((argument, index) => walk(argument, entity, ['arguments', 'args', index], 'argument'));
    Object.entries(operation.arguments.kwargs).forEach(([name, argument]) => walk(argument, entity, ['arguments', 'kwargs', name], 'argument'));
    operation.effects.forEach((effect, index) => uses.push({ kind: 'effect', entity, path: ['effects', index, 'beforeId'], valueId: effect.beforeId }));
  }
  walk(capture.output, { kind: 'output', id: capture.id }, ['output'], 'result');
  return uses;
}

function compareCapturedValueUses(before: Capture, after: Capture, correspondence: CaptureCorrespondence): SemanticCaptureComparison['valueUses'] {
  const changes: CapturedUseChange[] = [], unresolved: CapturedUseChange[] = [];
  const prepare = (capture: Capture, side: 'beforeId' | 'afterId') => {
    const operations = new Map<string, number>(), values = new Map<string, number>();
    correspondence.pairs.forEach((pair, index) => {
      if (pair.kind === 'operation') operations.set(pair[side], index);
      if (pair.kind === 'value') values.set(pair[side], index);
    });
    const slots = new Map<string, { use: CapturedValueUse; value: number | undefined }>();
    for (const use of capturedValueUses(capture)) {
      const target = use.entity.kind === 'output' ? 'output-boundary' : operations.get(use.entity.id);
      if (target === undefined) { unresolved.push(side === 'beforeId' ? { before: use, after: null } : { before: null, after: use }); continue; }
      slots.set(captureJson([target, use.kind, use.path]), { use, value: values.get(use.valueId) });
    }
    return slots;
  };
  const a = prepare(before, 'beforeId'), b = prepare(after, 'afterId');
  for (const key of Array.from(new Set([...Array.from(a.keys()), ...Array.from(b.keys())]))) {
    const left = a.get(key), right = b.get(key);
    const entry = { before: left?.use ?? null, after: right?.use ?? null };
    if ((left && left.value === undefined) || (right && right.value === undefined)
      || ((!left || !right) && (before.status !== 'complete' || after.status !== 'complete'))) unresolved.push(entry);
    else if (left?.value !== right?.value) changes.push(entry);
  }
  return { changes, unresolved };
}

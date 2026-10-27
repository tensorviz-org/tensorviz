import * as assert from 'node:assert/strict';
import { argumentStructure, captureJson, correspondCaptures } from '@tensorviz/contracts/captureCorrespondence';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
type CaptureSnapshot = components['schemas']['CaptureSnapshot'];
import * as fs from 'node:fs';
import * as path from 'node:path';
const portable = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../tests/fixtures/contracts/portable-capture.json'), 'utf8'));

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
const fixture = () => clone(portable.capture) as CaptureSnapshot;
/** Change every scoped identity, including references, without changing any role. */
function renumber(capture: CaptureSnapshot): CaptureSnapshot {
  const ids = new Set([capture.id, capture.attemptId, capture.source.id,
    ...[capture.definitions, capture.modules, capture.invocations, capture.operations, capture.values, capture.sourceBindings]
      .flatMap(items => items.map(item => item.id)), ...capture.values.flatMap(value => [value.objectId, value.storageId].filter((id): id is string => id !== null))]);
  const rewrite = (value: unknown): unknown => Array.isArray(value) ? value.map(rewrite)
    : value !== null && typeof value === 'object' ? Object.fromEntries(Object.entries(value).map(([key, item]) => [key, rewrite(item)]))
      : typeof value === 'string' && ids.has(value) ? `candidate:${value}` : value;
  return rewrite(capture) as CaptureSnapshot;
}
function bind(capture: CaptureSnapshot, operation: number, token = 'b'.repeat(64)) {
  const id = `source:${operation}`;
  capture.sourceBindings.push({ id, snapshotId: capture.source.id, kind: 'call', quality: 'exact', anchors: [{
    file: capture.source.modelFile, digest: capture.source.files[0].digest, startLine: 10 + operation, startColumn: 0,
    endLine: 10 + operation, endColumn: 5, scope: 'Model.forward', tokenDigest: token }] });
  capture.operations[operation].sourceBindingIds = [id];
}
function nested(capture: CaptureSnapshot, paths = ['layer']) {
  capture.definitions.push({id:'def:layer', qualifiedName:'torch.nn.Linear', sourceBindingIds:[]});
  capture.modules.push({id:'module:layer',definitionId:'def:layer',paths});
  const root = capture.invocations[0]; root.endSequence = 5;
  capture.invocations.push({...clone(root),id:'call:layer',moduleId:'module:layer',parentId:root.id,sequence:1,endSequence:4});
  capture.operations.forEach((op,i) => {op.invocationId='call:layer';op.sequence=i+2;});
  capture.coverage.invocationsSeen=2;
  return capture;
}
suite('canonical capture correspondence', () => {
  test('matches h+1 then a+a by recorded roles after all scoped ids and ordinals change', () => {
    const before=fixture(), after=renumber(before);
    after.operations.forEach(op=>{op.sequence*=2;}); after.invocations[0].endSequence=6;
    const result=correspondCaptures(before,after);
    assert.equal(result.pairs.length,7); assert.deepEqual(result.unmatched.before,{module:[],invocation:[],operation:[],value:[]});
    assert.equal(result.pairs.find(pair=>pair.beforeId==='add2')?.afterId,'candidate:add2');
    assert.equal(result.pairs.find(pair=>pair.beforeId==='add2')?.basis,'dataflow_role');
  });
  test('does not use record order or operation numbering to pair computations', () => {
    const before=fixture(), after=renumber(before);
    after.operations.reverse(); after.values.reverse();
    assert.equal(correspondCaptures(before,after).pairs.find(pair=>pair.beforeId==='add1')?.afterId,'candidate:add1');
  });
  test('literal values can change in the same distinct dataflow role', () => {
    const before=fixture(), after=renumber(before);
    after.operations[0].arguments.args[1]={kind:'int',value:'2'};
    assert.equal(correspondCaptures(before,after).pairs.filter(pair=>pair.kind==='operation').length,2);
    assert.notEqual(captureJson(argumentStructure(before.operations[0].arguments,id=>id)),
      captureJson(argumentStructure(after.operations[0].arguments,id=>id)));
  });
  test('identical repeated operations stay ambiguous even when their ids are unchanged', () => {
    const before=fixture(); before.operations[1].arguments=clone(before.operations[0].arguments);
    const result=correspondCaptures(before,clone(before));
    assert.equal(result.pairs.filter(pair=>pair.kind==='operation').length,0);
    assert.deepEqual(result.uncertain.find(group=>group.kind==='operation'),{kind:'operation',beforeIds:['add1','add2'],afterIds:['add1','add2']});
  });
  test('counts already source-paired duplicates when considering remaining candidates', () => {
    const before=fixture(); before.operations[1].arguments=clone(before.operations[0].arguments); bind(before,0);
    const after=renumber(before); const result=correspondCaptures(before,after);
    assert.deepEqual(result.pairs.filter(pair=>pair.kind==='operation').map(pair=>pair.beforeId),['add1']);
  });
  test('exact source anchors pair changed operator and operand roles without treating line numbers as identity', () => {
    const before=fixture();bind(before,0);const after=renumber(before);
    after.operations[0].target='aten.mul.Tensor';after.operations[0].arguments.args.reverse();
    after.sourceBindings[0].anchors[0].startLine=20;after.sourceBindings[0].anchors[0].endLine=20;
    const result=correspondCaptures(before,after);
    assert.equal(result.pairs.find(pair=>pair.beforeId==='add1')?.basis,'source_call');
  });
  test('candidate source bindings and duplicate token anchors cannot resolve repeated expressions', () => {
    const before=fixture();before.operations[1].arguments=clone(before.operations[0].arguments);bind(before,0);bind(before,1);
    assert.equal(correspondCaptures(before,renumber(before)).pairs.filter(pair=>pair.kind==='operation').length,0);
    before.sourceBindings[0].quality='candidate';before.sourceBindings[1].quality='candidate';
    assert.equal(correspondCaptures(before,renumber(before)).pairs.filter(pair=>pair.kind==='operation').length,0);
  });
  test('matches named module calls by their ownership, never the displayed label or class alone', () => {
    const before=nested(fixture()),after=renumber(before);
    assert.equal(correspondCaptures(before,after).pairs.find(pair=>pair.beforeId==='call:layer')?.basis,'unique_module_call');
    after.modules[1].paths=['renamed'];
    assert.equal(correspondCaptures(before,after).pairs.filter(pair=>pair.kind==='invocation').length,1);
  });
  test('numeric ModuleList paths remain ambiguous even if both runs reuse the same numeric positions', () => {
    const before=nested(fixture(),['layers.0']),after=renumber(before);
    const result=correspondCaptures(before,after);
    assert.ok(result.unmatched.before.module.includes('module:layer'));
    assert.equal(result.pairs.filter(pair=>pair.kind==='operation').length,0);
  });
  test('does not conflate one shared module with separate instances', () => {
    const before=nested(fixture(),['left','right']),after=renumber(before);
    after.modules[1].paths=['left'];after.modules.push({id:'separate',definitionId:after.modules[1].definitionId,paths:['right']});
    assert.ok(correspondCaptures(before,after).unmatched.before.module.includes('module:layer'));
  });
  test('root inputs preserve sharing and nested operand positions, not just equal shapes', () => {
    const before=fixture();before.inputs.args.push({kind:'value',id:'h'});before.invocations[0].arguments=clone(before.inputs);
    const after=renumber(before);const next=clone(after.values[0]);next.id='separate-input';next.objectId='separate-object';next.storageId='separate-storage';after.values.push(next);
    after.inputs.args[1]={kind:'value',id:next.id};after.invocations[0].arguments=clone(after.inputs);
    assert.ok(correspondCaptures(before,after).unmatched.before.value.includes('h'));
  });
  test('keeps kwargs insertion order and typed literal/container distinctions', () => {
    const left={args:[],kwargs:{first:{kind:'int' as const,value:'1'},second:{kind:'bool' as const,value:true}}};
    const right={args:[],kwargs:{second:left.kwargs.second,first:left.kwargs.first}};
    assert.notEqual(captureJson(argumentStructure(left,id=>id)),captureJson(argumentStructure(right,id=>id)));
    assert.notEqual(captureJson({kind:'int',value:'1'}),captureJson({kind:'bool',value:true}));
  });
  test('opaque and external input facts do not become a confident dataflow correspondence', () => {
    const before=fixture();before.operations[0].arguments.args[1]={kind:'opaque',reason:'unsupported_type',typeName:'Custom'};
    assert.equal(correspondCaptures(before,renumber(before)).pairs.filter(pair=>pair.kind==='operation').length,0);
  });
  test('matches effect-produced values by the exact producer port, without comparing runtime ids', () => {
    const before=fixture();
    before.operations[0].effects=[{kind:'write',argumentPath:['args',0],beforeId:'h',after:{kind:'value',id:'a'}}];
    before.operations[0].output={kind:'none'};
    Object.assign(before.values[1],{producer:{operationId:'add1',kind:'effect',path:[0,'after']},
      objectId:before.values[0].objectId,storageId:before.values[0].storageId,previousVersion:'h',logicalVersion:1,version:1});
    const after=renumber(before);const result=correspondCaptures(before,after);
    assert.equal(result.pairs.find(pair=>pair.beforeId==='a')?.basis,'producer_port');
    after.operations[0].effects.unshift({...clone(after.operations[0].effects[0]),after:{kind:'opaque',reason:'value_limit',typeName:null}});
    after.values[1].producer!.path=[1,'after'];
    assert.ok(correspondCaptures(before,after).unmatched.before.value.includes('a'));
  });
  test('handles a long dependent chain without ordinal matching and leaves both inputs untouched', () => {
    const before=fixture();before.operations=[];before.values=before.values.slice(0,1);
    for(let i=0;i<4000;i++) {
      before.operations.push({...clone(fixture().operations[0]),id:`op${i}`,sequence:i+1,
        arguments:{args:[{kind:'value',id:i?`value${i-1}`:'h'},{kind:'int',value:'1'}],kwargs:{}},output:{kind:'value',id:`value${i}`}});
      before.values.push({...clone(fixture().values[1]),id:`value${i}`,objectId:`object${i}`,storageId:`storage${i}`,producer:{operationId:`op${i}`,kind:'output',path:[]}});
    }
    before.invocations[0].endSequence=4001;before.output={kind:'value',id:'value3999'};before.invocations[0].output=before.output;
    before.coverage.operationsSeen=4000;before.limits.operations=5000;before.limits.values=5000;before.limits.bytes=16000000;
    const after=renumber(before), original=JSON.stringify([before,after]);
    assert.equal(correspondCaptures(before,after).pairs.filter(pair=>pair.kind==='operation').length,4000);
    assert.equal(JSON.stringify([before,after]),original);
  });
});

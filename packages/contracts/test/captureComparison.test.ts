import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { compareCaptures } from '@tensorviz/contracts/captureComparison';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
type CaptureSnapshot = components['schemas']['CaptureSnapshot'];
const initializerFixture = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/initializer-arguments.json'), 'utf8'));
const fixture = ():CaptureSnapshot => JSON.parse(readFileSync(path.join(__dirname,'../../../tests/fixtures/contracts/portable-capture.json'),'utf8')).capture;
const clone=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
const ids = new Set(['id','beforeId','attemptId','snapshotId','moduleId','parentId','rootInvocationId','invocationId','definitionId','operationId','diagnosticId','objectId','storageId','aliasOf','previousVersion','sourceBindingIds','constructorBindingIds','invocationIds']);
function renamed(value:unknown,key=''):any {
  if(Array.isArray(value))return value.map(item=>renamed(item,key));
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,renamed(item,key)]));
  return typeof value==='string'&&ids.has(key)?`new:${value}`:value;
}
const fields=(result:ReturnType<typeof compareCaptures>)=>result.changes.flatMap(record=>record.fields.map(field=>`${record.kind}:${record.beforeId}:${field.field}`));
suite('canonical captured computation differences',()=>{
  test('initializer entry differences are observations; legacy absence is explicit',()=>{
    const before=fixture(),after=renamed(before);
    before.modules[0].initializerArguments=clone(initializerFixture.entry) as any;
    after.modules[0].initializerArguments=clone(initializerFixture.entry);
    after.modules[0].initializerArguments.positional[0].value.value='8';
    const result=compareCaptures(before,after);
    assert.equal(result.equivalent,true);assert.deepEqual(result.changes,[]);
    const observation=result.observations.find(record=>record.kind==='module')!;
    assert.equal(observation.fields[0].field,'initializerArguments');
    assert.equal((observation.fields[0].before.value as any).positional[0].value.value,'4');
    assert.equal((observation.fields[0].after.value as any).positional[0].value.value,'8');
    delete after.modules[0].initializerArguments;
    const legacy=compareCaptures(before,after);
    assert.equal(legacy.equivalent,true);
    assert.deepEqual(legacy.observations.find(record=>record.kind==='module')!.fields[0].after,{present:false});
  });
  test('runtime identity changes alone do not change captured computation or observations',()=>{
    const before=fixture(),after=renamed(before);after.operations[0].sequence=1;after.operations[1].sequence=3;after.invocations[0].endSequence=4;
    const result=compareCaptures(before,after);
    assert.equal(result.equivalent,true);assert.deepEqual(result.changes,[]);assert.deepEqual(result.observations,[]);assert.deepEqual(result.unresolved,[]);
  });
  test('detects exact literal changes in corresponding operations',()=>{
    const before=fixture(),after=renamed(before);after.operations[0].arguments.args[1].value='2';
    const result=compareCaptures(before,after);
    assert.equal(result.equivalent,false);assert.ok(fields(result).includes('operation:add1:arguments'));
    const diff=result.changes.find(record=>record.beforeId==='add1')!.fields[0];
    assert.equal((diff.before.value as any).args[1].value,'1');assert.equal((diff.after.value as any).args[1].value,'2');
  });
  test('preserves kwargs order, mapping order and literal types even with equal numeric interpretations',()=>{
    const before=fixture();before.inputs.kwargs={a:{kind:'int',value:'1'},b:{kind:'bool',value:true}};
    before.invocations[0].arguments=clone(before.inputs);
    const after=renamed(before);after.inputs.kwargs={b:after.inputs.kwargs.b,a:after.inputs.kwargs.a};after.invocations[0].arguments=clone(after.inputs);
    assert.equal(compareCaptures(before,after).equivalent,false);
    after.inputs.kwargs.a={kind:'bool',value:true};
    assert.ok(fields(compareCaptures(before,after)).includes('capture:capture:0:inputs'));
  });
  test('ordered mapping entries remain distinct from the same keys in a different order',()=>{
    const before=fixture();before.inputs.args.push({kind:'dict',entries:[
      {key:{kind:'str',value:'first'},value:{kind:'int',value:'1'}},
      {key:{kind:'str',value:'second'},value:{kind:'int',value:'2'}}]});
    before.invocations[0].arguments=clone(before.inputs);
    const after=renamed(before);after.inputs.args[1].entries.reverse();after.invocations[0].arguments=clone(after.inputs);
    assert.ok(fields(compareCaptures(before,after)).includes('capture:capture:0:inputs'));
  });
  test('retains unresolved operands rather than reporting ID churn as rewiring',()=>{
    const before=fixture(),after=renamed(before);
    after.operations[1].arguments.args[1]={kind:'value',id:'new:h'};
    const result=compareCaptures(before,after);
    assert.equal(result.equivalent,null);assert.deepEqual(result.changes,[]);
    assert.ok(result.unresolved.some(record=>record.fields.some(field=>field.field==='output')));
    assert.ok(result.correspondence.unmatched.before.operation.includes('add2'));
  });
  test('ambiguous repeated operations cannot certify equivalence even in the same imported file',()=>{
    const before=fixture();before.operations[1].arguments=clone(before.operations[0].arguments);
    const result=compareCaptures(before,clone(before));
    assert.equal(result.equivalent,null);assert.deepEqual(result.changes,[]);assert.ok(result.correspondence.uncertain.length);
  });
  test('different complete operation counts prove captured computation differs without inventing occurrence matches',()=>{
    const before=fixture(),after=renamed(before),next=clone(after.operations[1]);
    next.id='third';next.sequence=3;next.output={kind:'value',id:'third-value'};after.operations.push(next);
    after.values.push({...clone(after.values[2]),id:'third-value',objectId:'third-object',storageId:'third-storage',
      producer:{operationId:'third',kind:'output',path:[]}});
    after.invocations[0].endSequence=4;after.invocations[0].output=next.output;after.output=next.output;after.coverage.operationsSeen=3;
    const result=compareCaptures(before,after);assert.equal(result.equivalent,false);
    assert.ok(fields(result).includes('capture:capture:0:recordedCounts'));
    assert.ok(result.correspondence.unmatched.before.operation.includes('add2'));
  });
  test('shape, dtype, device and counter observations stay separate from computation',()=>{
    const before=fixture(),after=renamed(before);after.values[0].shape=[4,2];after.values[0].dtype='float64';after.values[0].device='cuda:0';after.values[0].version=4;
    const result=compareCaptures(before,after);
    assert.equal(result.equivalent,true);assert.deepEqual(result.changes,[]);
    assert.deepEqual(result.observations.find(record=>record.beforeId==='h')!.fields.map(field=>field.field),['shape','dtype','device','version']);
  });
  test('weight and input fingerprints are execution controls, not architecture edits',()=>{
    const before=fixture(),after=renamed(before);after.inputDigest='b'.repeat(64);after.stateDigest='c'.repeat(64);
    const result=compareCaptures(before,after);assert.equal(result.equivalent,true);
    assert.equal(result.controls.find(control=>control.field==='weights')!.status,'different');
    after.stateDigest=null;assert.equal(compareCaptures(before,after).controls.find(control=>control.field==='weights')!.status,'unknown');
  });
  test('training and grad modes remain per-call execution observations',()=>{
    const before=fixture(),after=renamed(before);after.invocations[0].training=true;after.invocations[0].gradEnabled=true;
    const result=compareCaptures(before,after);assert.equal(result.equivalent,true);
    assert.deepEqual(result.observations[0].fields.map(field=>field.field),['training','gradEnabled']);
  });
  test('detects instance sharing changes without equating class identity with object identity',()=>{
    const before=fixture();before.modules.push({id:'shared',definitionId:before.definitions[0].id,paths:['left','right']});
    const after=renamed(before);after.modules[1].paths=['left'];after.modules.push({id:'split',definitionId:after.definitions[0].id,paths:['right']});
    const result=compareCaptures(before,after);assert.equal(result.equivalent,false);assert.ok(fields(result).includes('capture:capture:0:moduleRegistrations'));
  });
  test('compares storage sharing and explicit aliases independently of raw object/storage IDs',()=>{
    const before=fixture(),after=renamed(before);after.values[2].storageId=after.values[1].storageId;after.values[2].aliasOf=after.values[1].id;
    const result=compareCaptures(before,after);assert.equal(result.equivalent,false);
    assert.ok(fields(result).includes('capture:capture:0:storageSharing'));assert.ok(fields(result).includes('value:b:aliasOf'));
    assert.ok(!fields(result).includes('capture:capture:0:objectSharing'));
  });
  test('logical value versions and predecessor relationships remain semantic facts',()=>{
    const before=fixture(),after=renamed(before);after.values[2].logicalVersion=1;after.values[2].previousVersion='new:a';
    after.values[2].objectId=after.values[1].objectId;after.values[2].storageId=after.values[1].storageId;after.values[2].version=1;
    const result=compareCaptures(before,after);assert.equal(result.equivalent,false);
    assert.ok(fields(result).includes('value:b:logicalVersion'));assert.ok(fields(result).includes('value:b:previousVersion'));
  });
  test('writable argument paths and nonreturned effects participate in comparison',()=>{
    const before=fixture();before.operations[0].effects=[{kind:'write',argumentPath:['args',0],beforeId:'h',after:{kind:'value',id:'a'}}];
    const after=renamed(before);after.operations[0].effects[0].kind='alias_update';
    const result=compareCaptures(before,after);assert.equal(result.equivalent,false);assert.ok(fields(result).includes('operation:add1:effects'));
    before.operations[0].arguments.args[1]={kind:'value',id:'h'};
    const repeated=renamed(before);repeated.operations[0].effects[0].argumentPath=['args',1];
    assert.ok(fields(compareCaptures(before,repeated)).includes('operation:add1:effects'));
  });
  test('failed or partial capture never certifies equality or treats absent return as Python None',()=>{
    const before=fixture(),after=renamed(before);after.status='partial';after.output=null;after.invocations[0].output=null;
    after.invocations[0].status='interrupted';after.invocations[0].endSequence=null;after.coverage.omissions=['cancelled'];
    const result=compareCaptures(before,after);assert.equal(result.equivalent,null);assert.equal(result.coverageComplete,false);
    assert.deepEqual(result.changes,[]);assert.ok(result.observations.some(record=>record.fields.some(field=>field.field==='status')));
    assert.ok(result.unresolved.some(record=>record.fields.some(field=>field.field==='output')));
  });
  test('incomplete registration evidence does not certify an added or removed instance',()=>{
    const before=fixture(),after=renamed(before);after.status='partial';after.modules.push({id:'extra',definitionId:after.definitions[0].id,paths:['extra']});
    const result=compareCaptures(before,after);assert.equal(result.equivalent,null);assert.deepEqual(result.changes,[]);
  });
  test('unknown storage identity cannot become an assertion that tensors do not alias',()=>{
    const before=fixture(),after=renamed(before);after.values[0].storageId=null;
    const result=compareCaptures(before,after);assert.equal(result.equivalent,null);assert.deepEqual(result.changes,[]);
    assert.ok(result.unresolved.some(record=>record.fields.some(field=>field.field==='storageSharing')));
  });
  test('comparison leaves both retained inputs unchanged',()=>{
    const before=fixture(),after=renamed(before);after.operations[0].arguments.args[1].value='2';const original=JSON.stringify([before,after]);
    compareCaptures(before,after);assert.equal(JSON.stringify([before,after]),original);
  });
});

import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { compareRetainedRuns, type ComparableRun } from '@tensorviz/graph-host/mcpRunComparison';
import { inspectCapture } from '@tensorviz/graph-host/captureInspection';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';

const clone=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
const idKeys = new Set(['id','beforeId','captureId','sourceId','semanticId','valueId','attemptId','snapshotId','moduleId','parentId','rootInvocationId','invocationId','definitionId','operationId','diagnosticId','objectId','storageId','aliasOf','previousVersion','sourceBindingIds','constructorBindingIds','invocationIds','nodeId','edgeId','sourceNodeId','targetNodeId','sourcePort','targetPort','sourcePortId','targetPortId','displaySourcePortId','displayTargetPortId']);
function rename(value:unknown,key=''):any {
  if(Array.isArray(value))return value.map(item=>rename(item,key));
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,rename(item,key)]));
  return typeof value==='string'&&idKeys.has(key)?`next:${value}`:value;
}
function fixture():ComparableRun {
  const {capture}=canonicalFixture();
  const document=JSON.parse(readFileSync(path.resolve(__dirname,'../../../canvas/src/test/fixtures/canvas/capture-document.json'),'utf8'));
  return {runId:'before',status:'passed',evidence:{capture,execution:{operations:[{operationId:'wrong-legacy'}]}},graph:{capture:{id:capture.id},
    nodes:document.semantic.nodes.map((node:any)=>({nodeId:node.node_id,captureRef:node.metadata.captureRef,typedPorts:node.metadata.capturePorts})),
    edges:document.semantic.edges.map((edge:any)=>({edgeId:edge.edge_id,sourceNodeId:edge.source_node_id,targetNodeId:edge.target_node_id,
      sourcePort:edge.source_port,targetPort:edge.target_port,captureEdge:edge.metadata.captureEdge}))}};
}
const cap=(run:ComparableRun)=>run.evidence!.capture as CaptureSnapshot;
const compare=(a:ComparableRun,b:ComparableRun,section='summary')=>compareRetainedRuns(a,b,section,0,50,1e-6,1e-5) as any;
function anchor(capture:CaptureSnapshot) {
  capture.sourceBindings.push({id:'call-anchor',kind:'call',quality:'exact',snapshotId:capture.source.id,anchors:[{
    file:capture.source.modelFile,digest:capture.source.files[0].digest,scope:'Model.forward',tokenDigest:'b'.repeat(64),
    startLine:10,startColumn:0,endLine:10,endColumn:5}]});
  capture.operations[1].sourceBindingIds=['call-anchor'];
}
function rewireProjection(run:ComparableRun,kind:string,id:string,portPath:unknown[],valueId:string) {
  const nodes=run.graph.nodes as any[],edges=run.graph.edges as any[];
  const node=nodes.find(node=>node.captureRef.kind===kind&&node.captureRef.semanticId===id);
  const port=node.typedPorts.find((port:any)=>port.paths.some((path:unknown)=>JSON.stringify(path)===JSON.stringify(portPath)));
  const input=nodes.find(node=>node.captureRef.kind==='input');
  const edge=edges.find(edge=>edge.targetNodeId===node.nodeId&&edge.targetPort===port.id);
  port.valueId=valueId;edge.captureEdge.valueId=valueId;edge.sourceNodeId=input.nodeId;edge.sourcePort=input.typedPorts[0].id;
  edge.captureEdge.sourcePortId=input.typedPorts[0].id;
  return edge;
}
suite('canonical MCP value-use comparison',()=>{
  test('ignores capture/document/port ID churn and maps real graph records by capture references',()=>{
    const before=fixture(),after=rename(before);after.runId='after';
    const result=compare(before,after);assert.equal(result.format,'semantic-v1');assert.equal(result.equivalentCapturedComputation,true);
    assert.equal(result.connectionChangeCount,0);assert.equal(result.unresolvedConnectionCount,0);
    const pairs=compare(before,after,'correspondence').pairs.items;
    const operation=pairs.find((pair:any)=>pair.kind==='operation'&&pair.beforeId==='add2');
    assert.ok(operation.baselineNodeId);assert.equal(operation.candidateNodeId,`next:${operation.baselineNodeId}`);
  });
  test('reconnecting one occurrence of a+a yields one changed value-use with its actual graph edge',()=>{
    const before=fixture();anchor(cap(before));const after=rename(before);
    cap(after).operations[1].arguments.args[1]={kind:'value',id:'next:h'};
    const edge=rewireProjection(after,'operation','next:add2',['arguments','args',1],'next:h');
    const result=compare(before,after,'connections');
    assert.equal(result.count,1);assert.equal(result.unresolved.count,0);
    assert.deepEqual(result.items[0].baseline.path,['arguments','args',1]);assert.equal(result.items[0].baseline.valueId,'a');
    assert.equal(result.items[0].candidate.valueId,'next:h');assert.equal(result.items[0].candidate.edgeId,edge.edgeId);
    assert.equal(result.equivalentCapturedComputation,false);
  });
  test('an output connection supplies a valid inspect_run boundary-relative path',()=>{
    const before=fixture(),after=rename(before);cap(after).output={kind:'value',id:'next:h'};cap(after).invocations[0].output=cap(after).output;
    rewireProjection(after,'output',cap(after).id,['output'],'next:h');
    const result=compare(before,after,'connections');const output=result.items.find((item:any)=>item.candidate.kind==='result');
    assert.deepEqual(output.candidate.inspect.path,[]);
    const page=inspectCapture(cap(after),{captureId:cap(after).id,sourceId:cap(after).source.id,
      entity:output.candidate.inspect.entity,path:output.candidate.inspect.path,offset:0,limit:50});
    assert.equal(page.items.find(item=>item.key==='id')!.value,'next:h');
  });
  test('missing or foreign projection references cannot invent a graph node to focus',()=>{
    const before=fixture(),after=rename(before);for(const node of after.graph.nodes as any[])node.captureRef.sourceId='foreign';
    const result=compare(before,after,'correspondence');
    assert.ok(result.pairs.items.every((pair:any)=>pair.candidateNodeId===null));assert.equal(result.equivalentCapturedComputation,true);
  });
  test('partial or ambiguous operand correspondence cannot become a removed connection',()=>{
    const before=fixture(),after=rename(before);cap(after).operations[1].arguments.args[1]={kind:'value',id:'next:h'};
    const result=compare(before,after,'connections');assert.equal(result.count,0);assert.ok(result.unresolved.count>0);
    assert.equal(result.equivalentCapturedComputation,null);
  });
  test('legacy/canonical and rejected/canonical pairs refuse semantic comparison',()=>{
    const before=fixture(),after=clone(before);after.evidence!.capture=null;after.evidence!.captureUnavailable='invalid_capture';
    assert.equal(compare(before,after).format,'unavailable');
    delete after.graph.capture;after.evidence!.captureUnavailable='not_captured';
    assert.equal(compare(before,after).semanticComparisonAvailable,false);
    assert.equal(compare(before,after).nodeChangeCount,undefined);
  });
});

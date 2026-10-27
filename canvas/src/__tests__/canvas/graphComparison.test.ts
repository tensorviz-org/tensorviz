import { compareGraphFiles } from '../../services/graphComparison';
import { createGraphFile } from '../../services/graphFile';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';
const file = () => createGraphFile(nestedGraph(), { name: 'Net', status: 'static' });
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));
it('reports public parameter values including absent, null, arrays and objects distinctly', () => {
  const before = file(); const after = clone(before);
  const node = after.document.semantic.nodes.find(n => n.node_id === 'fc')!;
  node.params.out_features = 100; delete node.params.in_features;
  node.params.bias = null; node.params.shape = [2, 4]; node.params.config = {enabled:true};
  const result = compareGraphFiles(before, after);
  expect(result.changedLayers).toHaveLength(1);
  expect(result.changedLayers[0].changes).toEqual(expect.arrayContaining([
    {field:'parameters.out_features',before:{present:true,value:4},after:{present:true,value:100}},
    {field:'parameters.in_features',before:{present:true,value:8},after:{present:false}},
    {field:'parameters.bias',before:{present:false},after:{present:true,value:null}},
    {field:'parameters.shape',before:{present:false},after:{present:true,value:[2,4]}},
  ]));
  expect(result.architectureChanged).toBe(true);
});
it('compares operations, kinds, parent membership, ports and repeat counts', () => {
  const before = file(); const after = clone(before);
  const node = after.document.semantic.nodes.find(n => n.node_id === 'fc')!;
  node.op='Conv2d';node.kind='op';node.metadata.parentId='block';node.metadata.repeatCount=3;node.in_ports=['other'];
  expect(compareGraphFiles(before,after).changedLayers[0].changes.map(c=>c.field)).toEqual(['inputPorts','kind','operation','parent','repeatCount']);
});
it('detects added/removed layers and compares endpoint-port connections as multisets', () => {
  const before=file();const after=clone(before);
  after.document.semantic.nodes=after.document.semantic.nodes.filter(n=>n.node_id!=='output');
  after.document.semantic.nodes.push({...clone(after.document.semantic.nodes.find(n=>n.node_id==='fc')!),node_id:'dropout'});
  after.document.semantic.edges=[{...after.document.semantic.edges[0],edge_id:'renamed'}, {...after.document.semantic.edges[0],edge_id:'duplicate'}];
  const result=compareGraphFiles(before,after);
  expect(result.addedLayers.map(n=>n.node_id)).toEqual(['dropout']); expect(result.removedLayers.map(n=>n.node_id)).toEqual(['output']);
  expect(result.addedConnections[0]).toMatchObject({count:1,beforeCount:1,afterCount:2,target:{source:'input',target:'fc'}});
  expect(result.removedConnections[0]).toMatchObject({count:1,beforeCount:1,afterCount:0,target:{source:'fc',target:'output'}});
});
it('never reports layout, edge ids/order, annotation edits or snapshot time as architecture changes', () => {
  const before=file();const after=clone(before);
  after.exportedAt='2026-09-12T15:00:00Z';after.snapshotId='different';
  after.document.view.node_positions.fc={x:900,y:12};
  after.document.semantic.nodes.reverse();after.document.semantic.edges.reverse().forEach((e,i)=>e.edge_id=`new-${i}`);
  after.document.semantic.nodes[0].metadata.repeatHidden=true;
  after.collaboration.walkthrough.title='Changed annotation';
  const result=compareGraphFiles(before,after);
  expect(result.architectureChanged).toBe(false);expect(result.recordedShapes).toEqual([]);
  expect(result.changedLayers).toEqual([]);expect(result.addedConnections).toEqual([]);expect(result.removedConnections).toEqual([]);
});
it('reports recorded shapes and tracing status separately from architecture', () => {
  const before=file();const after=clone(before);after.status='traced';
  after.document.semantic.nodes.find(n=>n.node_id==='fc')!.metadata.outputShape=[1,4];
  const result=compareGraphFiles(before,after);
  expect(result.architectureChanged).toBe(false);expect(result.snapshotEvidenceChanged).toBe(true);
  expect(result.recordedShapes[0]).toMatchObject({nodeId:'fc',changes:[{field:'outputShape',before:{present:false},after:{present:true,value:[1,4]}}]});
});
it('flags a possible rename without pairing identities or hiding added/removed layers', () => {
  const before=file();const after=clone(before);after.document.semantic.nodes.find(n=>n.node_id==='fc')!.node_id='classifier';
  const result=compareGraphFiles(before,after);
  expect(result.uncertainMatches).toEqual([{beforeIds:['fc'],afterIds:['classifier']}]);
  expect(result.addedLayers.map(n=>n.node_id)).toEqual(['classifier']);expect(result.removedLayers.map(n=>n.node_id)).toEqual(['fc']);
  expect(result.changedLayers).toEqual([]);
});
it('groups ambiguous rename candidates rather than guessing a one-to-one correspondence', () => {
  const before=file();const after=clone(before);const source=after.document.semantic.nodes.find(n=>n.node_id==='fc')!;
  after.document.semantic.nodes=after.document.semantic.nodes.filter(n=>n.node_id!=='fc');
  after.document.semantic.nodes.push({...source,node_id:'b'}, {...source,node_id:'a'});
  expect(compareGraphFiles(before,after).uncertainMatches).toEqual([{beforeIds:['fc'],afterIds:['a','b']}]);
});
it('reports differing model names as an identity warning without fabricating a structural change', () => {
  const before=file();const after=clone(before);after.document.semantic.name='OtherNet';
  expect(compareGraphFiles(before,after)).toMatchObject({modelNamesDiffer:true,architectureChanged:false});
});
it('does not mutate either input and produces stable ordering for large graphs', () => {
  const before=file(); const source=before.document.semantic.nodes.find(n=>n.node_id==='fc')!;
  before.document.semantic.nodes=Array.from({length:5000},(_,i)=>({...clone(source),node_id:`layer-${String(i).padStart(4,'0')}`}));
  before.document.semantic.edges=[];
  const after=clone(before);after.document.semantic.nodes.reverse();after.document.semantic.nodes[0].params.out_features=100;
  const original=JSON.stringify([before,after]);const result=compareGraphFiles(before,after);
  expect(result.changedLayers.map(n=>n.nodeId)).toEqual(['layer-4999']);expect(JSON.stringify([before,after])).toBe(original);
});

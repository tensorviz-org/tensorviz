import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import GraphFileViewer from '../../viewer/GraphFileViewer';
import { createGraphFile, GraphFile } from '../../services/graphFile';
import { downloadGraphFile } from '../../ide/graphFileExport';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';
import { capturedFile, renameCaptureFile, operationNode, bindOperation, replaceOperand } from '../../test/fixtures/capturedGraphFileFixture';
import { edgeTarget } from '../../services/graphCollaboration';
const mockFocus = jest.fn();
const mockMount = jest.fn();
jest.mock('../../ide/graphFileExport', () => ({ downloadGraphFile: jest.fn() }));
jest.mock('../../viewer/GraphFileCanvas', () => {
  const React = require('react');
  return { __esModule:true, default:React.forwardRef(({file,initialView,initialTarget,onTargetChange}:any,ref:any)=>{
    const [pan,setPan]=React.useState((initialView?.camera.x??0)/10);
    const value=file.document.semantic.nodes.find((n:any)=>n.node_id==='fc')?.params.out_features;
    const [selection,setSelection]=React.useState(initialTarget ?? (file.capture
      ? {kind:'node',nodeId:file.document.semantic.nodes.find((n:any)=>n.metadata.captureRef.kind==='operation').node_id}
      : {kind:'node',nodeId:'fc'}));
    React.useEffect(()=>{mockMount({value,initialView,initialTarget});},[]);
    React.useImperativeHandle(ref,()=>({captureView:()=>({camera:{x:pan*10,y:20,z:1},nodePositions:file.document.view.node_positions,expandedModuleIds:[],selectedNodeIds:selection.kind==='node'?[selection.nodeId]:[],...(selection.kind==='edge'?{selectedEdge:selection}:{})}),
      focusTarget:(target:any)=>{setSelection(target);mockFocus({value,target});}}));
    return <><div data-testid="canvas">Canvas architecture {value ?? file.capture?.capture.id}</div><button onClick={()=>setPan(pan+1)}>Pan {pan}</button><button onClick={()=>onTargetChange({kind:'node',nodeId:'fc'})}>Select fc</button></>;
  })};
});
const snapshot=()=>createGraphFile(nestedGraph(),{name:'Net',status:'static'});
const clone=<T,>(value:T):T=>JSON.parse(JSON.stringify(value));
const choose=(name:string,file:GraphFile|string)=>fireEvent.change(screen.getByLabelText(name),{target:{files:[new File([typeof file==='string'?file:JSON.stringify(file)],'model.tensorviz.json')]}});
async function open(after= snapshot()) {render(<GraphFileViewer/>);choose('Graph file',after);await screen.findByTestId('canvas');}
async function compare(before:GraphFile) {fireEvent.click(screen.getByRole('button',{name:'Compare',exact:true}));choose('Comparison Before file',before);await screen.findByRole('region',{name:'Comparison summary'});}
beforeEach(()=>{mockFocus.mockReset();mockMount.mockReset();(downloadGraphFile as jest.Mock).mockReset();});
it('reports concrete values and navigates added/removed layers on the correct snapshot',async()=>{
  const before=snapshot(); const graph=nestedGraph(); graph.nodes.find(n=>n.id==='fc')!.data.parameters.out_features=100;
  graph.nodes=graph.nodes.filter(n=>n.id!=='output');graph.edges=graph.edges.filter(e=>e.target!=='output');
  graph.nodes.push({...clone(graph.nodes.find(n=>n.id==='fc')!),id:'dropout'});
  const after=createGraphFile(graph,{name:'Net',status:'static'});after.collaboration.walkthrough.title='After guide';
  await open(after);fireEvent.click(screen.getByRole('button',{name:'Pan 0'}));await compare(before);
  const summary=screen.getByRole('region',{name:'Comparison summary'});
  expect(summary).toHaveTextContent('1 added layers · 1 removed · 1 changed');
  const fields=within(screen.getByRole('article',{name:'Changed layer fc'}));
  expect(fields.getByText('parameters.out_features')).toBeInTheDocument();expect(fields.getByText('100')).toBeInTheDocument();expect(fields.getByText('4')).toBeInTheDocument();
  fireEvent.click(within(screen.getByRole('region',{name:'Added layers'})).getByRole('button',{name:'Show after'}));
  expect(mockFocus).toHaveBeenCalledWith({value:100,target:{kind:'node',nodeId:'dropout'}});
  fireEvent.click(within(screen.getByRole('region',{name:'Removed layers'})).getByRole('button',{name:'Show before'}));
  expect(screen.getByTestId('canvas')).toHaveTextContent('architecture 4');
  expect(mockMount.mock.calls.at(-1)[0].initialTarget).toEqual({kind:'node',nodeId:'output'});
  expect(screen.getByRole('button',{name:'Update architecture'})).toBeDisabled();
  fireEvent.click(screen.getByRole('button',{name:'Download After graph file'}));
  expect((downloadGraphFile as jest.Mock).mock.calls[0][0].collaboration.walkthrough.title).toBe('After guide');
  fireEvent.click(screen.getByRole('button',{name:'Close comparison'}));
  expect(screen.getByTestId('canvas')).toHaveTextContent('architecture 100');expect(screen.getByRole('button',{name:'Pan 1'})).toBeInTheDocument();
});
it('retains the last comparison after an invalid import and clears it when the main file changes',async()=>{
  const before=snapshot();const after=clone(before);after.document.semantic.nodes.find(n=>n.node_id==='fc')!.params.out_features=100;
  await open(after);await compare(before);choose('Comparison Before file','invalid');
  expect(await screen.findByRole('alert')).toHaveTextContent('valid JSON');
  expect(screen.getByText('Architecture changes found')).toBeInTheDocument();
  choose('Graph file',snapshot());await waitFor(()=>expect(screen.queryByRole('complementary',{name:'Architecture comparison'})).toBeNull());
  fireEvent.click(screen.getByRole('button',{name:'Compare',exact:true}));
  expect(screen.getByRole('button',{name:'Choose Before file'})).toBeInTheDocument();expect(screen.queryByRole('region',{name:'Comparison summary'})).toBeNull();
});
it('keeps an unposted discussion draft through comparison and restores After before authoring',async()=>{
  const after=snapshot();after.document.semantic.nodes.find(n=>n.node_id==='fc')!.params.out_features=100;
  await open(after);fireEvent.click(screen.getByRole('button',{name:'Select fc'}));fireEvent.click(screen.getByRole('button',{name:'Discussions',exact:true}));
  fireEvent.click(screen.getByRole('button',{name:'Start discussion'}));fireEvent.change(screen.getByLabelText('Message'),{target:{value:'Keep this current-layer question.'}});
  await compare(snapshot());fireEvent.click(screen.getByRole('button',{name:'View before'}));
  expect(screen.getByTestId('canvas')).toHaveTextContent('architecture 4');
  fireEvent.click(screen.getByRole('button',{name:'Discussions',exact:true}));
  expect(screen.getByTestId('canvas')).toHaveTextContent('architecture 100');expect(screen.getByLabelText('Message')).toHaveValue('Keep this current-layer question.');
  fireEvent.click(screen.getByRole('button',{name:'Download graph file'}));expect(screen.getByRole('alert')).toHaveTextContent('draft message');expect(downloadGraphFile).not.toHaveBeenCalled();
});
it('clearly reports layout-only equality and keeps record evidence separate',async()=>{
  const before=snapshot();const after=clone(before);after.document.view.node_positions.fc={x:1234,y:0};after.status='traced';after.document.semantic.nodes.find(n=>n.node_id==='fc')!.metadata.outputShape=[2,4];
  await open(after);await compare(before);
  expect(screen.getByText('No architecture changes')).toBeInTheDocument();
  expect(screen.getByRole('region',{name:'Recorded evidence'})).toHaveTextContent('Not recorded');
});
it('makes uncertain rename candidates visible without removing them from the added/removed counts',async()=>{
  const before=snapshot();const graph=nestedGraph();graph.nodes.find(n=>n.id==='fc')!.id='classifier';
  graph.edges.forEach(e=>{if(e.source==='fc')e.source='classifier';if(e.target==='fc')e.target='classifier';});
  await open(createGraphFile(graph,{name:'Net',status:'static'}));await compare(before);
  expect(screen.getByRole('region',{name:'Uncertain matches'})).toHaveTextContent('classifier');
  expect(screen.getByRole('region',{name:'Comparison summary'})).toHaveTextContent('1 added layers · 1 removed');
});
it('does not claim an ambiguous duplicate connection can be uniquely highlighted',async()=>{
  const before=snapshot();const after=clone(before);after.document.semantic.edges.push({...after.document.semantic.edges[0],edge_id:'duplicate'});
  await open(after);await compare(before);
  const section=within(screen.getByRole('region',{name:'Added connections'}));
  expect(section.getByRole('button',{name:'Show after'})).toBeDisabled();expect(section.getByText(/no unique edge/)).toBeInTheDocument();
});

it('follows renamed captured cards in both directions while restoring After when comparison closes',async()=>{
  const before=capturedFile(),after=renameCaptureFile(before);
  await open(after);fireEvent.click(screen.getByRole('button',{name:'Pan 0'}));await compare(before);
  fireEvent.click(screen.getByRole('button',{name:'View before'}));
  expect(mockMount.mock.calls.at(-1)[0].initialTarget).toEqual({kind:'node',nodeId:operationNode(before,'add1').node_id});
  fireEvent.click(screen.getByRole('button',{name:'View after'}));
  expect(mockMount.mock.calls.at(-1)[0].initialTarget).toEqual({kind:'node',nodeId:operationNode(after,'after:add1').node_id});
  fireEvent.click(screen.getByRole('button',{name:'View before'}));
  fireEvent.click(screen.getByRole('button',{name:'Close comparison'}));
  expect(mockMount.mock.calls.at(-1)[0].initialTarget).toBeUndefined();
  expect(screen.getByRole('button',{name:'Pan 1'})).toBeInTheDocument();
});

it('keeps a changed operand selected when toggling canonical Before and After',async()=>{
  const before=capturedFile();bindOperation(before,1);const after=renameCaptureFile(before);replaceOperand(after,1,1,'after:h');
  await open(after);await compare(before);
  fireEvent.click(within(screen.getByRole('region',{name:'Changed tensor uses'})).getByRole('button',{name:'Show before'}));
  expect(mockMount.mock.calls.at(-1)[0].initialTarget).toEqual(edgeTarget(before.document.semantic.edges[2]));
  fireEvent.click(screen.getByRole('button',{name:'View after'}));
  expect(mockMount.mock.calls.at(-1)[0].initialTarget).toEqual(edgeTarget(after.document.semantic.edges[2]));
});

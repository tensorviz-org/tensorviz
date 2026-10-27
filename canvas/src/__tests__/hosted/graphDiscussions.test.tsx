import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import GraphFileViewer from '../../viewer/GraphFileViewer';
import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { downloadGraphFile } from '../../ide/graphFileExport';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';
import { anchorStatus, edgeTarget } from '../../services/graphCollaboration';

const mockFocus = jest.fn();
jest.mock('../../ide/graphFileExport', () => ({ downloadGraphFile: jest.fn() }));
jest.mock('../../viewer/GraphFileCanvas', () => {
  const React = require('react');
  return { __esModule: true, default: React.forwardRef(({ file, onTargetChange }: any, ref: any) => {
    React.useImperativeHandle(ref, () => ({ focusTarget: mockFocus, captureView: () => ({ camera: { x: 0, y: 0, z: 1 }, nodePositions: file.document.view.node_positions, expandedModuleIds: ['block','inner'], selectedNodeIds: ['fc'] }), restoreView: () => undefined }));
    return <><button onClick={() => onTargetChange({kind:'node',nodeId:'fc'})}>Select fc</button>
      <button onClick={() => onTargetChange({kind:'edge',source:'input',target:'fc',sourcePort:'y',targetPort:'x'})}>Select connection</button></>;
  }) };
});
const file = () => createGraphFile(nestedGraph(), { name: 'Net', status: 'static' });
function choose(f: unknown, label = 'Graph file') {
  fireEvent.change(screen.getByLabelText(label), { target: { files: [new File([typeof f === 'string' ? f : JSON.stringify(f)], 'Net.tensorviz.json')] } });
}
async function open() { render(<GraphFileViewer />); choose(file()); await screen.findByRole('heading', { name: 'Net' }); fireEvent.click(screen.getByRole('button', { name: 'Discussions', exact: true })); }
function start(target = 'Select fc') { fireEvent.click(screen.getByRole('button', { name: target })); fireEvent.click(screen.getByRole('button', { name: 'Start discussion' })); }
function post(body: string) { fireEvent.change(screen.getByLabelText('Message'), { target: { value: body } }); fireEvent.click(screen.getByRole('button', { name: 'Add message' })); }
beforeEach(() => { mockFocus.mockReset(); (downloadGraphFile as jest.Mock).mockReset(); });
it('round trips an authored layer discussion, reply and resolution in the graph file', async () => {
  await open(); start(); fireEvent.change(screen.getByLabelText('Your display name'), { target: { value: 'Sam' } }); post('Why normalize here?');
  const thread = screen.getByRole('article', { name: 'Discussion on fc' });
  fireEvent.click(within(thread).getByRole('button', { name: 'Reply' })); post('It stabilizes the activation distribution.');
  fireEvent.click(within(thread).getByRole('button', { name: 'Resolve discussion' })); expect(within(thread).getByText('Resolved')).toBeInTheDocument();
  fireEvent.click(within(thread).getByRole('button', { name: 'Reopen discussion' }));
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const saved = parseGraphFile(JSON.stringify((downloadGraphFile as jest.Mock).mock.calls[0][0]));
  expect(saved.collaboration.threads[0].messages.map(m => [m.author,m.body])).toEqual([['Sam','Why normalize here?'],['Sam','It stabilizes the activation distribution.']]);
  expect(saved.collaboration.threads[0].resolved).toBe(false);
  choose(saved); await waitFor(() => expect(screen.queryByRole('article')).toBeNull());
  fireEvent.click(screen.getByRole('button', { name: 'Discussions', exact: true }));
  expect(screen.getByText('Why normalize here?')).toBeInTheDocument();
});
it('anchors to connection endpoints rather than a sequential exported edge id', async () => {
  await open(); start('Select connection'); post('Is this residual path intentional?');
  const thread = screen.getByRole('article'); fireEvent.click(within(thread).getByRole('button', { name: 'Show current target' }));
  expect(mockFocus).toHaveBeenCalledWith(edgeTarget(file().document.semantic.edges[0]));
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const saved = (downloadGraphFile as jest.Mock).mock.calls[0][0];
  expect(saved.collaboration.threads[0].anchor.target).toEqual({kind:'edge',source:'input',target:'fc',sourcePort:'y',targetPort:'x'});
});
it('preserves original discussion context through changed and missing architecture replacements', async () => {
  await open(); start(); post('Keep this classifier?');
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const original = (downloadGraphFile as jest.Mock).mock.calls[0][0];
  const next = file(); next.document.semantic.nodes.find(n => n.node_id === 'fc')!.params.out_features = 100;
  choose(next, 'Replacement architecture file'); expect(await screen.findByText('Context changed')).toBeInTheDocument();
  expect(screen.getByText('Keep this classifier?')).toBeInTheDocument();
  const missing = file(); missing.document.semantic.nodes = missing.document.semantic.nodes.filter(n => n.node_id !== 'fc');
  missing.document.semantic.edges = []; delete missing.document.view.node_positions.fc;
  choose(missing, 'Replacement architecture file'); expect(await screen.findByText('Target missing')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Show current target' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const saved = (downloadGraphFile as jest.Mock).mock.calls.at(-1)[0];
  expect(saved.collaboration.threads[0].anchor).toEqual(original.collaboration.threads[0].anchor);
  expect(anchorStatus(saved, saved.collaboration.threads[0].anchor)).toBe('missing');
});
it('keeps drafts when switching panels and prevents a download from silently excluding them', async () => {
  await open(); start(); fireEvent.change(screen.getByLabelText('Message'), { target: { value: 'Unsent thought' } });
  fireEvent.click(screen.getByRole('button', { name: 'Walkthrough', exact: true }));
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('draft message'); expect(downloadGraphFile).not.toHaveBeenCalled();
  const event = new Event('beforeunload', {cancelable:true}); window.dispatchEvent(event); expect(event.defaultPrevented).toBe(true);
  choose(file()); await screen.findByRole('alertdialog'); fireEvent.click(screen.getByRole('button', {name:'Keep editing'}));
  fireEvent.click(screen.getByRole('button', { name: 'Discussions', exact: true })); expect(screen.getByLabelText('Message')).toHaveValue('Unsent thought');
  fireEvent.click(screen.getByRole('button', { name: 'Discard draft' })); expect(screen.getByText('No unsaved notes')).toBeInTheDocument();
});
it('refuses malformed or annotated architecture replacements without discarding local discussions', async () => {
  await open(); start(); post('Keep my context'); fireEvent.click(screen.getByRole('button', {name:'Download graph file'}));
  const annotated = (downloadGraphFile as jest.Mock).mock.calls[0][0];
  choose('broken', 'Replacement architecture file'); expect(await screen.findByRole('alert')).toHaveTextContent('valid JSON');
  choose(annotated, 'Replacement architecture file'); expect(await screen.findByRole('alert')).toHaveTextContent('contains notes');
  expect(screen.getByText('Keep my context')).toBeInTheDocument();
});

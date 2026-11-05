import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import LocalBrowserModelMenu, { LocalModelCatalog, LocalModelOption } from '../../ide/LocalBrowserModelMenu';

const target = { workspaceFolder: '/ws', file: 'layers/audio_vae_v2.py', cls: 'AudioVAEV2', documentVersion: 1,
  sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const model = (update: Partial<LocalModelOption> = {}): LocalModelOption => ({
  id: 'audio', cls: 'AudioVAEV2', file: target.file, evidence: 'source', opened: true, ...update,
});
const catalog = (models = [model(), model({ id: 'encoder', cls: 'Encoder', file: 'layers/encoder.py', opened: false })]): LocalModelCatalog => ({
  status: 'ready', activeId: 'audio', models,
});

describe('local model picker', () => {
  it('preserves manual utilities without claiming discovery in a host without an index', () => {
    render(<LocalBrowserModelMenu target={target}><button>Change environment</button></LocalBrowserModelMenu>);
    fireEvent.click(screen.getByRole('button', { name: 'Model and file actions' }));
    expect(screen.queryByText('No models found in this workspace.')).not.toBeInTheDocument();
    expect(screen.queryByText('Finding models…')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('File and setup'));
    expect(screen.getByRole('button', { name: 'Change environment' })).toBeVisible();
  });

  it('lists opened and project models with their actual evidence, retaining file utilities as secondary details', () => {
    render(<LocalBrowserModelMenu target={target} catalog={catalog([model({ evidence: 'captured' }), model({ id: 'encoder', cls: 'Encoder', file: 'layers/encoder.py', opened: false })])} onSelect={jest.fn()}>
      <button>Change environment</button>
    </LocalBrowserModelMenu>);
    fireEvent.click(screen.getByRole('button', { name: 'Switch model' }));
    const picker = screen.getByRole('dialog', { name: 'Switch model' });
    expect(within(picker).getByRole('region', { name: 'Opened models' })).toHaveTextContent('Captured');
    expect(within(picker).getByRole('region', { name: 'Project models' })).toHaveTextContent('From code');
    expect(picker.querySelector('[aria-current="true"]')).toHaveTextContent('AudioVAEV2');
    expect(screen.getByText('File and setup')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Change environment', hidden: true })).not.toBeVisible();
    fireEvent.click(screen.getByText('File and setup'));
    expect(screen.getByRole('button', { name: 'Change environment' })).toBeVisible();
  });

  it('selects the opaque host identity, closes and returns focus without reopening the current model', () => {
    const select = jest.fn();
    render(<LocalBrowserModelMenu target={target} catalog={catalog()} onSelect={select} />);
    const trigger = screen.getByRole('button', { name: 'Switch model' });
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole('button', { name: /Encoder/ }));
    expect(select).toHaveBeenCalledWith('encoder');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole('button', { name: /AudioVAEV2.*From code/ }));
    expect(select).toHaveBeenCalledTimes(1);
  });

  it('filters long lists by class or full source path and preserves complete model names', () => {
    const longName = 'UncommonMultiStageEncoderWithACompleteTechnicalClassName';
    const models = Array.from({ length: 10 }, (_, i) => model({ id: `model-${i}`, cls: `Model${i}`, file: `layers/stage_${i}.py`, opened: false }));
    models.push(model({ id: 'long', cls: longName, file: 'large/project/research/rare_implementation.py', opened: false }));
    render(<LocalBrowserModelMenu target={target} catalog={catalog(models)} onSelect={jest.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Switch model' }));
    const search = screen.getByRole('searchbox', { name: 'Find a model' });
    expect(search).toHaveFocus();
    fireEvent.change(search, { target: { value: 'rare_implementation' } });
    expect(screen.getByText(longName)).toBeVisible();
    expect(screen.queryByText('Model1')).not.toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'no such class' } });
    expect(screen.getByText('No matching models.')).toBeVisible();
  });

  it('supports arrow navigation, Escape, outside dismissal and disabled reasons', () => {
    const { rerender } = render(<LocalBrowserModelMenu target={target} catalog={catalog()} onSelect={jest.fn()} />);
    const trigger = screen.getByRole('button', { name: 'Switch model' });
    fireEvent.keyDown(trigger, { key: 'ArrowDown' });
    const first = screen.getByRole('button', { name: /AudioVAEV2.*From code/ });
    const second = screen.getByRole('button', { name: /Encoder/ });
    expect(first).toHaveFocus();
    fireEvent.keyDown(first, { key: 'ArrowDown' });
    expect(second).toHaveFocus();
    fireEvent.keyDown(second, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
    fireEvent.click(trigger);
    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    rerender(<LocalBrowserModelMenu target={target} catalog={catalog()} onSelect={jest.fn()} disabledReason="Finish the active model task first." />);
    fireEvent.click(trigger);
    expect(screen.getByRole('button', { name: /Encoder/ })).toBeDisabled();
    expect(screen.getByText('Finish the active model task first.')).toBeVisible();
  });

  it('shows honest loading, error, empty and partial-index states with refresh recovery', () => {
    const refresh = jest.fn();
    const { rerender } = render(<LocalBrowserModelMenu target={target} catalog={{ status: 'loading', models: [] }} onRefresh={refresh} />);
    fireEvent.click(screen.getByRole('button', { name: 'Switch model' }));
    expect(screen.getByRole('status')).toHaveTextContent('Finding models');
    expect(screen.getByRole('button', { name: 'Refresh models' })).toBeDisabled();
    rerender(<LocalBrowserModelMenu target={target} catalog={{ status: 'unavailable', models: [], detail: 'The workspace scan failed.' }} onRefresh={refresh} />);
    expect(screen.getByText('The workspace scan failed.')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
    expect(refresh).toHaveBeenCalledTimes(1);
    rerender(<LocalBrowserModelMenu target={target} catalog={{ status: 'ready', models: [], truncated: true }} />);
    expect(screen.getByText('No models found in this workspace.')).toBeVisible();
    expect(screen.getByText(/Showing part of the workspace/)).toBeVisible();
    expect(screen.queryByText('Captured')).not.toBeInTheDocument();
  });
});

import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import CodeExportDialog from '../../components/CodeExportDialog';
import { ApiService } from '../../services/apiService';
import { ModelGraph } from '../../types';

/**
 * The export dialog states two kinds of fact that regress silently.
 *
 * 1. Its "Options" are NOT settings. /api/models/export-package takes graph,
 *    input_shape and class_name and nothing else, so a live checkbox there would
 *    be a control that quietly does nothing. These tests fail if one becomes
 *    enabled, or if its state stops being read off the generated file.
 * 2. Every size it prints is measured from the bytes it is showing. The mockup's
 *    "3.1 KB" is not a fallback — a hardcoded size is indistinguishable from a
 *    real one on screen, which is exactly why it needs a test.
 */

// Only an Input node: findMissingRequiredParameters exempts input/output, so this
// graph exports without tripping the incomplete-parameters refusal.
const graph = {
  nodes: [
    {
      id: 'input-1',
      position: { x: 0, y: 0 },
      data: {
        layerType: { id: 'input', name: 'Input', displayName: 'Input' },
        parameters: { shape: '1, 3, 224, 224' },
      },
    },
  ],
  edges: [],
} as unknown as ModelGraph;

/** 400 bytes exactly, with two nn.Module classes and no annotations/assertions. */
const MODEL_PY = [
  'import torch',
  'import torch.nn as nn',
  '',
  '',
  'class ResBlock(nn.Module):',
  '    def __init__(self):',
  '        super().__init__()',
  '        self.conv = nn.Conv2d(3, 64, kernel_size=3)',
  '',
  '    def forward(self, x):',
  '        return self.conv(x)',
  '',
  '',
  'class ResNet18Ablation(nn.Module):',
  '    def __init__(self):',
  '        super().__init__()',
  '        self.layer1 = ResBlock()',
  '',
  '    def forward(self, x):',
  '        return self.layer1(x)',
  '',
].join('\n');

/** The same file padded to exactly 2048 bytes. */
const PADDED_MODEL_PY = `${MODEL_PY}# ${'a'.repeat(1646)}`;

/** One nn.Module class, so nothing was factored out into a sub-module. */
const SINGLE_CLASS_MODEL_PY = [
  'import torch',
  'import torch.nn as nn',
  '',
  '',
  'class ResNet18Ablation(nn.Module):',
  '    def forward(self, x):',
  '        return x',
  '',
].join('\n');

const ZIP_BYTES = '0'.repeat(5222); // 5.1 KB

function mockBackend(code: string, torch?: string) {
  const generateCode = jest
    .spyOn(ApiService, 'generateCode')
    .mockResolvedValue({ code, unsupported: [] });
  const exportPackage = jest
    .spyOn(ApiService, 'exportPackage')
    .mockResolvedValue(new Blob([ZIP_BYTES]));
  // useBackendEnvironment reads /health directly; jsdom has no fetch.
  (global as any).fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => (torch ? { environment: { torch } } : { environment: {} }),
  });
  return { generateCode, exportPackage };
}

function renderDialog() {
  return render(
    <CodeExportDialog
      isOpen
      onClose={jest.fn()}
      graph={graph}
      graphName="ResNet18 Ablation"
    />,
  );
}

afterEach(() => {
  jest.restoreAllMocks();
  delete (global as any).fetch;
});

describe('export options', () => {
  it('renders every option as a disabled control, because the endpoint takes no options', async () => {
    const { generateCode, exportPackage } = mockBackend(MODEL_PY);
    renderDialog();

    const assertions = await screen.findByRole('checkbox', { name: /Shape assertions/ });
    const hints = screen.getByRole('checkbox', { name: /Type hints/ });
    const subModules = screen.getByRole('checkbox', { name: /Sub-modules as classes/ });

    expect(assertions).toBeDisabled();
    expect(hints).toBeDisabled();
    expect(subModules).toBeDisabled();

    // A live control would flip here and then have nowhere to send the value.
    fireEvent.click(assertions);
    expect(assertions).not.toBeChecked();

    // The payload is exactly what the route accepts — no option fields.
    expect(generateCode).toHaveBeenCalledWith({
      graph,
      input_shape: [1, 3, 224, 224],
      class_name: 'ResNet18Ablation',
    });
    expect(exportPackage).toHaveBeenCalledWith(graph, [1, 3, 224, 224], 'ResNet18Ablation');
  });

  it('reads each option off the generated file rather than a fixed default', async () => {
    mockBackend(MODEL_PY);
    renderDialog();

    // MODEL_PY defines ResBlock alongside the model, carries no annotations and
    // no assert statements.
    expect(await screen.findByRole('checkbox', { name: /Sub-modules as classes/ })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: /Shape assertions/ })).not.toBeChecked();
    expect(screen.getByRole('checkbox', { name: /Type hints/ })).not.toBeChecked();
    expect(screen.getByText('factored')).toBeInTheDocument();
  });

  it('reports no sub-modules when the generated file has a single class', async () => {
    mockBackend(SINGLE_CLASS_MODEL_PY);
    renderDialog();

    await waitFor(() =>
      expect(screen.getByRole('checkbox', { name: /Sub-modules as classes/ })).not.toBeChecked(),
    );
    expect(screen.getByText('no repeats')).toBeInTheDocument();
  });
});

describe('sizes', () => {
  it('measures the file size from the generated content', async () => {
    mockBackend(MODEL_PY);
    renderDialog();

    expect(await screen.findByText('400 B')).toBeInTheDocument();
    // The mockup's figure, which nothing in the response supports.
    expect(screen.queryByText('3.1 KB')).not.toBeInTheDocument();
  });

  it('reports a different size for a different file', async () => {
    mockBackend(PADDED_MODEL_PY);
    renderDialog();

    expect(await screen.findByText('2.0 KB')).toBeInTheDocument();
    expect(screen.queryByText('400 B')).not.toBeInTheDocument();
  });

  it('states the zip size from the blob the download will hand over', async () => {
    mockBackend(MODEL_PY);
    renderDialog();

    expect(await screen.findByText('ResNet18Ablation.zip · 5.1 KB')).toBeInTheDocument();
  });

  it('lists the entries it cannot preview without inventing a size for them', async () => {
    mockBackend(MODEL_PY);
    renderDialog();

    await screen.findByText('400 B');
    const sampleRun = screen.getByRole('button', { name: /sample_run\.py/ });
    expect(sampleRun).toBeDisabled();
    expect(sampleRun).toHaveTextContent(/^sample_run\.py$/);
  });
});

describe('the torch version in the lede', () => {
  it('names the version the backend reported', async () => {
    mockBackend(MODEL_PY, '2.9.1');
    renderDialog();

    expect(
      await screen.findByText('A self-contained zip. Runs against torch 2.9.1 with no canvas dependency.'),
    ).toBeInTheDocument();
  });

  it('drops the clause when the backend reported no version', async () => {
    mockBackend(MODEL_PY);
    renderDialog();

    expect(
      await screen.findByText('A self-contained zip with no canvas dependency.'),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Runs against torch/)).not.toBeInTheDocument();
  });
});

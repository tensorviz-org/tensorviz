import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import CodeImportDialog from '../../components/CodeImportDialog';
import { ApiService } from '../../services/apiService';

const parsedResponse = {
  nodes: [],
  edges: [],
  input_shape: [1, 3, 32, 32],
  class_name: 'SmallNet',
};

beforeEach(() => {
  jest.restoreAllMocks();
});

function renderDialog() {
  return render(
    <CodeImportDialog
      isOpen
      onClose={jest.fn()}
      onImport={jest.fn()}
    />,
  );
}

it('sends a user-provided input shape to the parser', async () => {
  const parseCode = jest.spyOn(ApiService, 'parseCode').mockResolvedValue(parsedResponse);
  renderDialog();

  fireEvent.change(screen.getByPlaceholderText('Paste your PyTorch model code here...'), {
    target: { value: 'class SmallNet(nn.Module):\n    pass' },
  });
  fireEvent.change(screen.getByLabelText(/Input Shape/i), {
    target: { value: '1, 3, 32, 32' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Import Model' }));

  await waitFor(() => expect(parseCode).toHaveBeenCalledWith({
    code: 'class SmallNet(nn.Module):\n    pass',
    input_shape: [1, 3, 32, 32],
  }));
});

it('omits input_shape when the field is blank so the parser can detect it', async () => {
  const parseCode = jest.spyOn(ApiService, 'parseCode').mockResolvedValue(parsedResponse);
  renderDialog();

  fireEvent.change(screen.getByPlaceholderText('Paste your PyTorch model code here...'), {
    target: { value: 'class SmallNet(nn.Module):\n    pass' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Import Model' }));

  await waitFor(() => expect(parseCode).toHaveBeenCalledWith({
    code: 'class SmallNet(nn.Module):\n    pass',
  }));
});

it('rejects an invalid input shape before calling the parser', async () => {
  const parseCode = jest.spyOn(ApiService, 'parseCode').mockResolvedValue(parsedResponse);
  renderDialog();

  fireEvent.change(screen.getByPlaceholderText('Paste your PyTorch model code here...'), {
    target: { value: 'class SmallNet(nn.Module):\n    pass' },
  });
  fireEvent.change(screen.getByLabelText(/Input Shape/i), {
    target: { value: '1, channels, 32, 32' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Import Model' }));

  expect(await screen.findByText('Input shape must be a comma-separated list of positive integers')).toBeVisible();
  expect(parseCode).not.toHaveBeenCalled();
});

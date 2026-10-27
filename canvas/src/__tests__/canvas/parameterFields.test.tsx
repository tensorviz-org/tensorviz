/**
 * The Inspector's Properties fields.
 *
 * Two behaviours matter enough to pin:
 *
 * 1. A slice is picked ON THE AXIS. The chips come from the real input shape, so
 *    the track IS the bound and `start + length > size` cannot be expressed —
 *    the out-of-range error becomes unreachable rather than well-worded.
 * 2. When the shape is NOT known (inference has not run — node.data.inputShape is
 *    undefined until then) the control must degrade to plain fields and say why.
 *    A track with no bounds would be worse than a number box.
 *
 * Plus `showIf`, which exists because F.interpolate takes size XOR scale_factor
 * and raises when given both.
 */
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ParameterFields, {
  clearHiddenParameters,
  isParameterVisible,
} from '../../components/ui2/ParameterFields';
import { Parameter } from '../../types';
import { getLayerByType } from '../../data/layerTypes';
import { inspectorDraftParameters } from '../../utils/inspectorDraft';

const renderFields = (props: {
  layerTypeId: string;
  inputShape?: number[];
  values: Record<string, string>;
}) => {
  const onChange = jest.fn();
  const view = render(<ParameterFields {...props} onChange={onChange} />);
  return { onChange, view };
};

const SLICE_VALUES = { dim: '1', start: '0', length: '16' };

describe('a slice is chosen on the axis, not typed', () => {
  it('builds the axis chips from the actual input shape', () => {
    renderFields({ layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: SLICE_VALUES });
    // One chip per axis, each showing that axis's extent and its label. A chip's
    // text is the two concatenated, e.g. '64C'.
    expect(screen.getByTitle('Slice on C')).toHaveTextContent('64');
    expect(screen.getByTitle('Slice on C')).toHaveTextContent('C');
    expect(screen.getByTitle('Slice on H')).toHaveTextContent('32');
    expect(screen.getByTitle('Slice on W')).toHaveTextContent('32');
    // Four axes: the locked batch chip plus C/H/W.
    expect(screen.getByTitle('The batch axis is not sliceable here')).toHaveTextContent('1');
  });

  it('locks the batch axis, which is not a meaningful thing to slice here', () => {
    renderFields({ layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: SLICE_VALUES });
    const batch = screen.getByTitle('The batch axis is not sliceable here');
    expect(batch).toBeDisabled();
  });

  it('shows the range as start -> end even though the node stores length', () => {
    const { view } = renderFields({
      layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: { dim: '1', start: '16', length: '16' },
    });
    const readout = view.container.querySelector('.ui2-slice-read');
    // The user thinks in start:end; the node stores length, so end is derived.
    expect(readout).toHaveTextContent('16');
    expect(readout).toHaveTextContent('32');
    expect(readout).toHaveTextContent('16 of 64');
  });

  it('states the call it will emit', () => {
    renderFields({ layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: { dim: '2', start: '8', length: '16' } });
    expect(screen.getByText('torch.narrow(x, 2, 8, 16)')).toBeInTheDocument();
  });

  it('re-bounds the range when the axis changes', () => {
    // C is 64 and H is 32, so a length of 48 cannot survive a move to H.
    const { onChange } = renderFields({
      layerTypeId: 'narrow',
      inputShape: [1, 64, 32, 32],
      values: { dim: '1', start: '0', length: '48' },
    });
    fireEvent.click(screen.getByTitle('Slice on H'));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ dim: '2', start: '0', length: '32' }));
  });

  it('clamps a length that would run off the end', () => {
    const { onChange } = renderFields({
      layerTypeId: 'narrow',
      inputShape: [1, 64, 32, 32],
      values: { dim: '1', start: '60', length: '16' },
    });
    // start 60 + length 16 exceeds 64, so the control cannot present it.
    expect(screen.getByText('4 of 64')).toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Range end' }), { key: 'ArrowRight' });
    const [next] = onChange.mock.calls[onChange.mock.calls.length - 1];
    expect(Number(next.start) + Number(next.length)).toBeLessThanOrEqual(64);
  });

  it('moves a handle from the keyboard, with a coarse step on shift', () => {
    const { onChange } = renderFields({
      layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: SLICE_VALUES,
    });
    const end = screen.getByRole('slider', { name: 'Range end' });
    fireEvent.keyDown(end, { key: 'ArrowRight' });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ length: '17' }));
    fireEvent.keyDown(end, { key: 'ArrowRight', shiftKey: true });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ length: '24' }));
  });

  it('exposes the handles as sliders with real bounds for assistive tech', () => {
    renderFields({ layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: SLICE_VALUES });
    const start = screen.getByRole('slider', { name: 'Range start' });
    expect(start).toHaveAttribute('aria-valuemin', '0');
    expect(start).toHaveAttribute('aria-valuemax', '64');
    expect(start).toHaveAttribute('aria-valuenow', '0');
  });

  describe('intent presets say what people actually mean', () => {
    it.each([
      ['First half', { start: '0', length: '32' }],
      ['Second half', { start: '32', length: '32' }],
      ['Drop first', { start: '1', length: '63' }],
      ['All', { start: '0', length: '64' }],
    ])('%s', (label, expected) => {
      const { onChange } = renderFields({
        layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: SLICE_VALUES,
      });
      fireEvent.click(screen.getByText(label));
      expect(onChange).toHaveBeenCalledWith(expect.objectContaining(expected));
    });
  });

  it('resolves a negative dim to a real axis rather than failing', () => {
    renderFields({ layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: { dim: '-1', start: '0', length: '8' } });
    // -1 on a rank-4 tensor is W.
    expect(screen.getByText('8 of 32')).toBeInTheDocument();
  });
});

describe('the escape hatch', () => {
  it('drops to plain number fields for matching a paper verbatim', () => {
    renderFields({ layerTypeId: 'narrow', inputShape: [1, 64, 32, 32], values: SLICE_VALUES });
    expect(screen.getByRole('slider', { name: 'Range start' })).toBeInTheDocument();
    fireEvent.click(screen.getByText('Type exact numbers instead'));
    // The track is gone and the three raw fields are back.
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
    expect(screen.getByLabelText('parameter-dim')).toBeInTheDocument();
    expect(screen.getByLabelText('parameter-start')).toBeInTheDocument();
    expect(screen.getByLabelText('parameter-length')).toBeInTheDocument();
  });
});

describe('without a known shape the control degrades honestly', () => {
  it('offers plain fields and says how to get the visual picker', () => {
    renderFields({ layerTypeId: 'narrow', values: SLICE_VALUES });
    expect(screen.getByText(/Refresh shapes/i)).toBeInTheDocument();
    // No track: a range control that cannot enforce its own bounds is worse than
    // a number box.
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
    expect(screen.getByLabelText('parameter-dim')).toBeInTheDocument();
    expect(screen.getByLabelText('parameter-length')).toBeInTheDocument();
  });

  it('also degrades for a rank-1 shape, where there is no axis to choose', () => {
    renderFields({ layerTypeId: 'narrow', inputShape: [64], values: SLICE_VALUES });
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
  });
});

describe('showIf keeps mutually exclusive arguments apart', () => {
  const alignCorners: Parameter = {
    name: 'align_corners',
    type: 'boolean',
    defaultValue: false,
    required: false,
    description: '',
    showIf: { param: 'mode', oneOf: ['linear', 'bilinear', 'bicubic', 'trilinear'] },
  };

  it('hides a field whose predicate fails', () => {
    expect(isParameterVisible(alignCorners, { mode: 'nearest' })).toBe(false);
    expect(isParameterVisible(alignCorners, { mode: 'bilinear' })).toBe(true);
  });

  it('treats a parameter with no rule as always visible', () => {
    const plain: Parameter = { name: 'scale_factor', type: 'number', defaultValue: 2, required: false, description: '' };
    expect(isParameterVisible(plain, {})).toBe(true);
  });

  it('supports an equals rule as well as oneOf', () => {
    const only: Parameter = {
      name: 'x', type: 'number', defaultValue: 1, required: false, description: '',
      showIf: { param: 'mode', equals: 'area' },
    };
    expect(isParameterVisible(only, { mode: 'area' })).toBe(true);
    expect(isParameterVisible(only, { mode: 'nearest' })).toBe(false);
  });

  it('BLANKS a hidden value, which is what keeps the predicate out of Python', () => {
    // codegen already skips empty params, so clearing is the whole mechanism —
    // the rule never has to be reimplemented on the backend.
    const cleared = clearHiddenParameters([alignCorners], { mode: 'nearest', align_corners: 'true' });
    expect(cleared.align_corners).toBe('');
    expect(cleared.mode).toBe('nearest');
  });

  it('leaves a visible value alone, and does not copy needlessly', () => {
    const values = { mode: 'bilinear', align_corners: 'true' };
    expect(clearHiddenParameters([alignCorners], values)).toBe(values);
  });

  it('does not render a hidden field for interpolate', () => {
    renderFields({
      layerTypeId: 'interpolate',
      values: { mode: 'nearest', scale_factor: '2', size: '', align_corners: 'false' },
    });
    expect(screen.queryByLabelText('parameter-align_corners')).not.toBeInTheDocument();
    expect(screen.getByLabelText('parameter-scale_factor')).toBeInTheDocument();
  });

  it('renders it once the mode can use it', () => {
    renderFields({
      layerTypeId: 'interpolate',
      values: { mode: 'bilinear', scale_factor: '2', size: '', align_corners: 'false' },
    });
    expect(screen.getByLabelText('parameter-align_corners')).toBeInTheDocument();
  });
});

describe('metadata drives the widget, and nothing loses its editor', () => {
  it('renders an option list for a select parameter', () => {
    renderFields({ layerTypeId: 'interpolate', values: { mode: 'nearest' } });
    const select = screen.getByLabelText('parameter-mode');
    expect(select.tagName).toBe('SELECT');
    expect(screen.getByRole('option', { name: 'bilinear' })).toBeInTheDocument();
  });

  it('renders a number input with the catalogue bounds', () => {
    renderFields({ layerTypeId: 'conv2d', values: { kernel_size: '3' } });
    const input = screen.getByLabelText('parameter-kernel_size');
    expect(input).toHaveAttribute('type', 'number');
  });

  it('falls back to a text input for a value the catalogue does not describe', () => {
    renderFields({ layerTypeId: 'conv2d', values: { some_unknown_key: 'abc' } });
    expect(screen.getByLabelText('parameter-some_unknown_key')).toHaveAttribute('type', 'text');
  });

  it('still renders for an op with no catalogue entry at all', () => {
    renderFields({ layerTypeId: 'not-a-real-op', values: { whatever: '1' } });
    expect(screen.getByLabelText('parameter-whatever')).toBeInTheDocument();
  });
});

describe('a widget shows the value the node actually holds', () => {
  it('renders an enabled boolean as True, whichever spelling the draft carries', () => {
    // The draft is seeded through formatParamValue, which spells booleans the way
    // Python does. Comparing against the lowercase form showed every enabled flag
    // as False — and it was unrepairable by hand, because a <select> fires onChange
    // only when the DOM selection moves, so picking "False" on a control already
    // reading False did nothing.
    renderFields({ layerTypeId: 'linear', values: { bias: 'True' } });
    expect(screen.getByLabelText('parameter-bias')).toHaveValue('true');
  });

  it('still renders a disabled boolean as False', () => {
    // Guards the same line against over-correcting into "always true".
    renderFields({ layerTypeId: 'linear', values: { bias: 'False' } });
    expect(screen.getByLabelText('parameter-bias')).toHaveValue('false');
    renderFields({ layerTypeId: 'linear', values: { bias: 'false' } });
    expect(screen.getAllByLabelText('parameter-bias')[1]).toHaveValue('false');
  });

  it('shows a tuple-valued number parameter instead of an empty box', () => {
    // kernel_size is type:'number' in the catalogue, but every imported conv layer
    // and the shipped ResNet-50 / VGG-16 templates hold a list there. An
    // <input type="number"> blanks anything that is not a single float, so the
    // field read as empty — and typing 3 to "fill it in" silently dropped the
    // asymmetry of a [3, 5].
    renderFields({ layerTypeId: 'conv2d', values: { kernel_size: '[7, 7]' } });
    const field = screen.getByLabelText('parameter-kernel_size');
    expect(field).toHaveValue('[7, 7]');
    expect(field).toHaveAttribute('type', 'text');
  });

  it('keeps the number input for an ordinary scalar', () => {
    renderFields({ layerTypeId: 'conv2d', values: { kernel_size: '3' } });
    expect(screen.getByLabelText('parameter-kernel_size')).toHaveAttribute('type', 'number');
  });
});

describe('permute and transpose edit the axes torch actually reads', () => {
  // The catalogue exposed this field as `dim` while torch.permute, the shape engine
  // and codegen all read `dims` — so whatever was typed into it was read by nothing:
  // the card reported no output shape and Export refused the node with "needs dims",
  // which the panel gave you no way to act on.
  const newNodeParameters = (layerTypeId: string) =>
    inspectorDraftParameters(
      Object.fromEntries(
        (getLayerByType(layerTypeId)?.parameters || []).map((p) => [p.name, p.defaultValue])
      )
    );

  it.each(['permute', 'transpose'])('%s: the field is named dims', (layerTypeId) => {
    // The panel renders the keys the NODE carries, and a new node carries the
    // catalogue's defaults — so the catalogue name is the field name.
    const values = newNodeParameters(layerTypeId);
    expect(Object.keys(values)).toEqual(['dims']);
    renderFields({ layerTypeId, values });
    expect(screen.getByLabelText('parameter-dims')).toBeInTheDocument();
    expect(screen.queryByLabelText('parameter-dim')).not.toBeInTheDocument();
  });

  it('commits the typed axis order under that name', () => {
    const { onChange } = renderFields({ layerTypeId: 'permute', values: { dims: '' } });
    fireEvent.change(screen.getByLabelText('parameter-dims'), { target: { value: '0,2,3,1' } });
    expect(onChange).toHaveBeenLastCalledWith({ dims: '0,2,3,1' });
  });
});

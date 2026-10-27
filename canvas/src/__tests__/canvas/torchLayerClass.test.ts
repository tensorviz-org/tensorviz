/**
 * A behaviour test, not a source-text pin.
 *
 * The Layer Code preview's class-name rule lived inside BuilderPageV2, which is not
 * importable under the CRA jest transform, so the only thing guarding it was a
 * regex over the file's text. It guessed `nn.Input(shape=…)` for a graph endpoint
 * with no torch class, and no text pin noticed — because the text was exactly what
 * the author intended to write. This test asserts what the function must ANSWER.
 */
import { torchClassForLayer } from '../../utils/torchLayerClass';

describe('torchClassForLayer', () => {
  it('names the class for a layer it knows', () => {
    expect(torchClassForLayer('conv2d', 'Conv2d')).toBe('nn.Conv2d');
    expect(torchClassForLayer('Linear', 'Linear')).toBe('nn.Linear');
  });

  it('refuses to invent nn.Input for a graph endpoint', () => {
    // PyTorch has no nn.Input. The preview used to print
    // `nn.Input(shape=1,3,224,224, name=input)`, which crashes on the first line
    // a student pastes it into.
    expect(torchClassForLayer('input', 'Input')).toBeNull();
    expect(torchClassForLayer('output', 'Output')).toBeNull();
    expect(torchClassForLayer('module', 'MyBlock')).toBeNull();
  });

  it('leaves an already-qualified torch symbol alone', () => {
    // The old rule stripped non-alphanumerics, so torch.matmul became the
    // nonexistent nn.torchmatmul.
    expect(torchClassForLayer('matmul', 'torch.matmul')).toBe('torch.matmul');
  });

  it('says nothing for an op it cannot vouch for', () => {
    // The backend registry holds well over a hundred ops and this module cannot
    // see it, so "I don't know" is the only honest answer — never nn.<GuessedName>.
    expect(torchClassForLayer('rmsnorm', 'RMSNorm')).toBeNull();
    expect(torchClassForLayer('neighbor_sampler', 'Neighbor Sampler')).toBeNull();
    expect(torchClassForLayer('foobar_layer', 'FooBar')).toBeNull();
  });

  it('is not confused by an empty or odd layer type', () => {
    expect(torchClassForLayer('', '')).toBeNull();
    expect(torchClassForLayer('CONV2D', 'Conv2d')).toBe('nn.Conv2d');
  });
});

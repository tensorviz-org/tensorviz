/**
 * Single-file import: which class is imported, and what the user is told.
 *
 * parseCode took `graphs.find(g => g.graph)` — the FIRST class in the source — so a
 * file defining a helper above the model imported the helper and silently dropped
 * the model; moving the classes around changed the result. It also hardcoded
 * `input_shape: request.input_shape || [1, 3, 224, 224]`, throwing away the shape
 * the parser had discovered, which is why a blank Input Shape on an MLP failed
 * validation with a mat1/mat2 mismatch.
 */
import { ApiService } from '../../services/apiService';

const graphWith = (inputShape: number[], id: string) => ({
  nodes: [
    { id: 'input', data: { layerType: { id: 'input' }, parameters: { shape: inputShape } } },
    { id, data: { layerType: { id: 'linear' }, parameters: {} } },
  ],
  edges: [{ id: 'e1', source: 'input', target: id }],
});

const respond = (body: unknown) => {
  jest.spyOn(global, 'fetch' as any).mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
    headers: { get: () => 'application/json' },
  } as any);
};

afterEach(() => jest.restoreAllMocks());

describe('single-file import picks the model, not the first class', () => {
  it('imports the class no other class instantiates, wherever it sits', async () => {
    respond({
      success: true,
      graphs: [
        { name: 'Block', graph: graphWith([1, 3, 224, 224], 'c'), is_top_level: false },
        { name: 'ClassifierNet', graph: graphWith([1, 784], 'fc'), is_top_level: true },
      ],
    });

    const parsed = await ApiService.parseCode({ code: 'x' } as any);

    expect(parsed.class_name).toBe('ClassifierNet');
    expect(parsed.nodes.map((n: any) => n.id)).toContain('fc');
    expect(parsed.nodes.map((n: any) => n.id)).not.toContain('c');
  });

  it('reports every class the file defined, so the caller can say what it skipped', async () => {
    respond({
      success: true,
      graphs: [
        { name: 'Block', graph: graphWith([1, 1], 'c'), is_top_level: false },
        { name: 'ClassifierNet', graph: graphWith([1, 784], 'fc'), is_top_level: true },
      ],
    });

    const parsed = await ApiService.parseCode({ code: 'x' } as any);

    expect(parsed.available_classes).toEqual(['Block', 'ClassifierNet']);
  });

  it('falls back to the first usable graph when nothing is marked top-level', async () => {
    // An older backend, or a file whose classes all reference each other.
    respond({
      success: true,
      graphs: [{ name: 'Only', graph: graphWith([1, 5], 'fc') }],
    });

    const parsed = await ApiService.parseCode({ code: 'x' } as any);

    expect(parsed.class_name).toBe('Only');
  });

  it('keeps the shape the parser discovered instead of the image default', async () => {
    respond({
      success: true,
      graphs: [{ name: 'MLP', graph: graphWith([1, 784], 'fc'), is_top_level: true }],
    });

    const parsed = await ApiService.parseCode({ code: 'x' } as any);

    expect(parsed.input_shape).toEqual([1, 784]);
  });

  it('still lets an explicitly requested shape win', async () => {
    respond({
      success: true,
      graphs: [{ name: 'MLP', graph: graphWith([1, 784], 'fc'), is_top_level: true }],
    });

    const parsed = await ApiService.parseCode({ code: 'x', input_shape: [8, 784] } as any);

    expect(parsed.input_shape).toEqual([8, 784]);
  });
});

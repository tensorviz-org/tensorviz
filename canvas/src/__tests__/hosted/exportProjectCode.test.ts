/**
 * Exporting a project as Python from the list pages.
 *
 * The gates matter more than the happy path. /api/models/validate is purely
 * structural — it never reads input_shape and the endpoint drops its warnings,
 * so a graph with no Input node passes it (backend/utils/model_validation.py).
 * Codegen then needs exactly that Input node for its shape. These tests pin the
 * order: the client preflight refuses first, validation second, and neither is
 * allowed to reach the network when the one before it has already refused.
 */
import {
  exportProjectAsPython,
  exportProjectAsZip,
  exportPayload,
  pythonClassName,
} from '../../services/exportProjectCode';
import { ApiService } from '../../services/apiService';

jest.mock('../../services/apiService');

const validateModel = ApiService.validateModel as jest.Mock;
const generateCode = ApiService.generateCode as jest.Mock;
const exportPackage = ApiService.exportPackage as jest.Mock;
const verifyModel = ApiService.verifyModel as jest.Mock;

function layer(id: string, typeId: string, parameters: Record<string, any> = {}, type = 'layerNode') {
  return {
    id,
    type,
    position: { x: 0, y: 0 },
    data: { layerType: { id: typeId, name: typeId, parameters: [] }, parameters },
  };
}

/** A graph that clears every preflight: an Input node with a shape, and a layer
 *  whose required parameters are all filled. */
function healthyProject(overrides: Record<string, any> = {}) {
  return {
    name: 'My CNN',
    document: undefined,
    graph: {
      nodes: [layer('n1', 'input', { shape: '1,3,224,224' }), layer('n2', 'relu', {})],
      edges: [{ id: 'e1', source: 'n1', target: 'n2' }],
    },
    ...overrides,
  } as any;
}

let clickedAnchor: HTMLAnchorElement | null = null;

beforeEach(() => {
  jest.clearAllMocks();
  clickedAnchor = null;
  (global as any).URL.createObjectURL = jest.fn(() => 'blob:generated');
  (global as any).URL.revokeObjectURL = jest.fn();
  jest
    .spyOn(HTMLAnchorElement.prototype, 'click')
    .mockImplementation(function (this: HTMLAnchorElement) {
      clickedAnchor = this;
    });
  validateModel.mockResolvedValue({ isValid: true, errors: [] });
  // Structural validation and the forward pass are separate gates, and the
  // preflight now runs both — a model can validate and still fail to run.
  verifyModel.mockResolvedValue({ ok: true, stage: 'complete', errors: [] });
  generateCode.mockResolvedValue({ code: 'import torch\n', unsupported: [] });
  exportPackage.mockResolvedValue(new Blob(['PK'], { type: 'application/zip' }));
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe('the export pipeline', () => {
  it('validates before generating, and downloads the class as its own .py', async () => {
    const result = await exportProjectAsPython(healthyProject());

    expect(validateModel).toHaveBeenCalledTimes(1);
    expect(generateCode).toHaveBeenCalledWith(
      expect.objectContaining({ class_name: 'MyCNN', input_shape: [1, 3, 224, 224] })
    );
    // Validation is a gate, not a formality: it has to have resolved first.
    expect(validateModel.mock.invocationCallOrder[0]).toBeLessThan(
      generateCode.mock.invocationCallOrder[0]
    );
    expect(result).toEqual({ ok: true, fileName: 'MyCNN.py', unsupported: [] });
    expect(clickedAnchor?.download).toBe('MyCNN.py');
  });

  it('refuses a graph with no Input node without calling either API', async () => {
    const project = healthyProject({
      graph: { nodes: [layer('n2', 'relu', {})], edges: [] },
    });

    const result = await exportProjectAsPython(project);

    // The whole reason the preflight runs first: /validate would have passed this.
    expect(result).toEqual({
      ok: false,
      message: 'No Input node with a shape — open the project and add one before exporting.',
    });
    expect(validateModel).not.toHaveBeenCalled();
    expect(generateCode).not.toHaveBeenCalled();
  });

  it('refuses when a layer is missing a required parameter', async () => {
    // Linear without out_features emits `nn.Linear()`, which raises TypeError on
    // the first line that uses it — the file must never be handed over.
    const project = healthyProject({
      graph: {
        nodes: [layer('n1', 'input', { shape: '1,784' }), layer('n2', 'linear', {})],
        edges: [{ id: 'e1', source: 'n1', target: 'n2' }],
      },
    });

    const result = await exportProjectAsPython(project);

    expect(result.ok).toBe(false);
    expect((result as any).message).toMatch(/missing required parameters/);
    expect(generateCode).not.toHaveBeenCalled();
  });

  it('blocks the download and quotes the errors when validation fails', async () => {
    validateModel.mockResolvedValue({
      isValid: false,
      errors: [{ nodeId: 'root', message: 'Graph contains a cycle' }],
    });

    const result = await exportProjectAsPython(healthyProject());

    expect(result).toEqual({
      ok: false,
      message: 'Validation failed — Graph contains a cycle',
    });
    expect(generateCode).not.toHaveBeenCalled();
    expect(clickedAnchor).toBeNull();
  });

  it('says the backend did not answer, rather than blaming the model', async () => {
    // A validate call that threw means the graph was never checked. Reporting it
    // as invalid sends the user to fix a model nothing found fault with.
    validateModel.mockRejectedValue(new Error('Failed to fetch'));

    const result = await exportProjectAsPython(healthyProject());

    expect(result).toEqual({ ok: false, message: "Couldn't validate — Failed to fetch." });
    expect(generateCode).not.toHaveBeenCalled();
  });

  it('reports an incomplete export instead of a clean one', async () => {
    generateCode.mockResolvedValue({
      code: '# INCOMPLETE EXPORT\n',
      unsupported: [{ class_name: 'UNetBlock', node_id: 'm1', op: 'module', imported: true, reason: 'x' }],
    });

    const result = await exportProjectAsPython(healthyProject());

    // The file still downloads — it carries the backend's banner — but the
    // caller is handed the reason it is partial.
    expect(result.ok).toBe(true);
    expect((result as any).unsupported).toHaveLength(1);
    expect(clickedAnchor?.download).toBe('MyCNN.py');
  });
});

describe('the payload the backend is given', () => {
  it('drops module wrappers and synthetic edges', () => {
    // Display structure, not computation: sending it is a stated correctness bug,
    // which is why ModelBuilder strips the same two things before every verify.
    const payload = exportPayload({
      document: undefined,
      graph: {
        nodes: [
          layer('m1', 'module', {}, 'moduleNode'),
          layer('n1', 'input', { shape: '1,3,8,8' }),
          layer('n2', 'relu', {}),
        ],
        edges: [
          { id: 'e1', source: 'n1', target: 'n2' },
          { id: 'e2', source: 'n2', target: 'm1', data: { synthetic: true } },
          { id: 'e3', source: 'm1', target: 'n2' },
        ],
      },
    } as any);

    expect(payload.nodes.map((n) => n.id)).toEqual(['n1', 'n2']);
    // e3 survives the synthetic filter but points at a node that no longer
    // exists — a dangling edge is a 500 from the graph model, not a no-op.
    expect(payload.edges.map((e) => e.id)).toEqual(['e1']);
  });
});

describe('the class name', () => {
  it.each([
    ['My CNN', 'MyCNN'],
    ['3D-UNet', 'Model3DUNet'],
    ['!!!', 'GeneratedModel'],
  ])('turns %s into a legal Python identifier %s', (input, expected) => {
    expect(pythonClassName(input)).toBe(expected);
  });
});

describe('the zip package', () => {
  it('clears the same gates as the single file before fetching anything', async () => {
    // A project that refuses to export as a file must refuse to export as an
    // archive for the same reason, rather than the two formats disagreeing
    // about whether the graph is sound.
    const project = healthyProject({
      graph: { nodes: [layer('n2', 'relu', {})], edges: [] },
    });

    const result = await exportProjectAsZip(project);

    expect(result).toEqual({
      ok: false,
      message: 'No Input node with a shape — open the project and add one before exporting.',
    });
    expect(validateModel).not.toHaveBeenCalled();
    expect(exportPackage).not.toHaveBeenCalled();
  });

  it('blocks the archive when validation fails', async () => {
    validateModel.mockResolvedValue({
      isValid: false,
      errors: [{ nodeId: 'root', message: 'Graph contains a cycle' }],
    });

    const result = await exportProjectAsZip(healthyProject());

    expect(result).toEqual({ ok: false, message: 'Validation failed — Graph contains a cycle' });
    expect(exportPackage).not.toHaveBeenCalled();
  });

  it('downloads the archive named for the class', async () => {
    const result = await exportProjectAsZip(healthyProject());

    expect(exportPackage).toHaveBeenCalledWith(expect.anything(), [1, 3, 224, 224], 'MyCNN');
    expect(result).toEqual({ ok: true, fileName: 'MyCNN.zip', unsupported: [] });
    expect(clickedAnchor?.download).toBe('MyCNN.zip');
  });

  it('still names unreproduced modules, which the zip endpoint cannot report', async () => {
    // export-package returns only bytes — it never calls unreproducible_modules,
    // so the second request is the only thing standing between the user and a
    // silently partial archive.
    generateCode.mockResolvedValue({
      code: '# INCOMPLETE EXPORT\n',
      unsupported: [{ class_name: 'UNetBlock', node_id: 'm1', op: 'module', imported: true, reason: 'x' }],
    });

    const result = await exportProjectAsZip(healthyProject());

    expect(result.ok).toBe(true);
    expect((result as any).unsupported).toHaveLength(1);
    expect(clickedAnchor?.download).toBe('MyCNN.zip');
  });

  it('does not hand over an archive when the generator failed', async () => {
    // Both endpoints share one generator. If codegen could not produce model.py,
    // the model.py inside the archive is not to be trusted either.
    generateCode.mockRejectedValue(new Error('Error generating code'));

    const result = await exportProjectAsZip(healthyProject());

    expect(result.ok).toBe(false);
    expect((result as any).message).toBe('Export failed — Error generating code.');
    expect(clickedAnchor).toBeNull();
  });

  it('surfaces the 401 that only the authenticated zip endpoint can raise', async () => {
    exportPackage.mockRejectedValue(new Error('Export package failed (401): Not authenticated'));

    const result = await exportProjectAsZip(healthyProject());

    expect(result).toEqual({
      ok: false,
      message: 'Export failed — Export package failed (401): Not authenticated.',
    });
    expect(clickedAnchor).toBeNull();
  });
});


describe('a model that validates but does not run is refused by BOTH exports', () => {
  // The defect: /api/models/export-package refuses a non-running model in the
  // BACKEND, so "Export as zip" was safe while "Export as Python" — the menu row
  // directly above it — called only generate-code and downloaded a file that
  // raised "mat1 and mat2 shapes cannot be multiplied" on first use, under a
  // green "downloaded" banner. Two adjacent controls, opposite guarantees.
  const notRunnable = {
    ok: false,
    stage: 'forward',
    errors: [{ node_id: 'n2', message: 'mat1 and mat2 shapes cannot be multiplied (1x31337 and 32x4)' }],
  };

  it('refuses the single-file Python export, and never asks for the code', async () => {
    verifyModel.mockResolvedValue(notRunnable);

    const result = await exportProjectAsPython(healthyProject());

    expect(result.ok).toBe(false);
    expect(result.ok === false && result.message).toContain('does not run');
    // The torch error is quoted, because it names the layer to go and fix.
    expect(result.ok === false && result.message).toContain('cannot be multiplied');
    // Nothing was generated and nothing was downloaded: a refusal that still
    // hands over a file is the bug, not the fix.
    expect(generateCode).not.toHaveBeenCalled();
    expect(clickedAnchor).toBeNull();
  });

  it('refuses the zip export for the same reason, before the backend has to', async () => {
    verifyModel.mockResolvedValue(notRunnable);

    const result = await exportProjectAsZip(healthyProject());

    expect(result.ok).toBe(false);
    expect(result.ok === false && result.message).toContain('does not run');
    expect(exportPackage).not.toHaveBeenCalled();
    expect(clickedAnchor).toBeNull();
  });

  it('still exports a model that does run', async () => {
    const result = await exportProjectAsPython(healthyProject());

    expect(result.ok).toBe(true);
    expect(generateCode).toHaveBeenCalled();
    expect(clickedAnchor).not.toBeNull();
  });

  it('says the backend is unreachable rather than blaming the model', async () => {
    // Same distinction planExport already draws for validation: a server that
    // never answered is not a model that failed.
    verifyModel.mockRejectedValue(new Error('Network request failed'));

    const result = await exportProjectAsPython(healthyProject());

    expect(result.ok).toBe(false);
    expect(result.ok === false && result.message).toContain("Couldn't verify");
    expect(result.ok === false && result.message).not.toContain('does not run');
  });
});

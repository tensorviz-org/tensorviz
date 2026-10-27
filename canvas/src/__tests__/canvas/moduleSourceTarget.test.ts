import { moduleDefinitionNodeId } from '../../canvas/moduleSourceTarget';

describe('moduleDefinitionNodeId', () => {
  it('targets a resolved member whose span belongs to the imported definition', () => {
    expect(moduleDefinitionNodeId(['enc1.fc', 'enc1.act'], 'enc1', 'shape:enc1')).toBe('enc1.fc');
  });

  it('falls back to the wrapper for an opaque module with no resolved members', () => {
    expect(moduleDefinitionNodeId([], 'dynamic', 'shape:dynamic')).toBe('dynamic');
  });

  it('retains the shape-id fallback for legacy module cards', () => {
    expect(moduleDefinitionNodeId(undefined, '', 'shape:legacy')).toBe('legacy');
  });
});

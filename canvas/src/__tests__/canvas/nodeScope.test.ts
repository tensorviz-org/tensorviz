import { scopedNodeLabel } from '../../canvas/nodeScope';

/**
 * Why this matters: an expanded module frame already draws its own name. A child
 * that repeats it spends its subtitle on a word the reader can see directly
 * above, which is what made a five-op block read as five copies of "stage1".
 */
describe('scopedNodeLabel', () => {
  it('drops the scope the frame already shows', () => {
    expect(scopedNodeLabel('stage1.conv1', 'stage1')).toBe('conv1');
  });

  it('keeps the full id for a node with no frame', () => {
    expect(scopedNodeLabel('stem', undefined)).toBe('stem');
    expect(scopedNodeLabel('stem', null)).toBe('stem');
  });

  it('strips only the parent, so a nested frame still names its own level', () => {
    expect(scopedNodeLabel('stage1.block.conv', 'stage1')).toBe('block.conv');
    expect(scopedNodeLabel('stage1.block.conv', 'stage1.block')).toBe('conv');
  });

  it('keeps the id when the parent is not really its prefix', () => {
    // autoModularize/repeatBlocks invent a frame id and adopt existing nodes;
    // trimming a non-prefix there would rename the node rather than scope it.
    expect(scopedNodeLabel('conv1', 'auto_block_0')).toBe('conv1');
    expect(scopedNodeLabel('stage10.conv1', 'stage1')).toBe('stage10.conv1');
  });

  it('never returns an empty label when id and parent coincide', () => {
    expect(scopedNodeLabel('stage1.', 'stage1')).toBe('stage1.');
  });
});

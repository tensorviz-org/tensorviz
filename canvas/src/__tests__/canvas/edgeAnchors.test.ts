/**
 * Floating anchor rules. The regression this pins: edges used to leave the right
 * edge and enter the left edge no matter where the cards sat, so a card below
 * another had to route up and over to reach a side nowhere near it.
 */
import {
  anchorPoint,
  boxCenter,
  computePortSides,
  portAnchorPoint,
  portFraction,
  sideForDirection,
  sourceSide,
  targetSide,
} from '../../canvas/edgeAnchors';

const box = (x: number, y: number, w = 200, h = 100) => ({ x, y, w, h });

describe('edge side selection', () => {
  it('a source faces the target: right when level or to the right, bottom only when below', () => {
    const from = box(0, 0);
    expect(sourceSide(from, boxCenter(box(400, 0)))).toBe('right');
    expect(sourceSide(from, boxCenter(box(-400, 0)))).toBe('right');
    expect(sourceSide(from, boxCenter(box(0, 400)))).toBe('bottom');
    // Above the source, but only slightly to the side: still the horizontal pair.
    expect(sourceSide(from, boxCenter(box(40, -400)))).toBe('right');
  });

  it('a target faces the source: left when level or to the left, top only when above', () => {
    const at = box(0, 0);
    expect(targetSide(at, boxCenter(box(400, 0)))).toBe('left');
    expect(targetSide(at, boxCenter(box(-400, 0)))).toBe('left');
    expect(targetSide(at, boxCenter(box(0, -400)))).toBe('top');
    expect(targetSide(at, boxCenter(box(40, 400)))).toBe('left');
  });

  it('a vertical pair is consistent from both ends: bottom out, top in', () => {
    const source = box(0, 0);
    const target = box(0, 400);
    expect(sourceSide(source, boxCenter(target))).toBe('bottom');
    expect(targetSide(target, boxCenter(source))).toBe('top');
  });

  it('sideForDirection routes an output and an input differently', () => {
    const at = box(0, 0);
    const below = boxCenter(box(0, 400));
    const above = boxCenter(box(0, -400));
    expect(sideForDirection('out', at, below)).toBe('bottom');
    expect(sideForDirection('in', at, below)).toBe('left');
    expect(sideForDirection('out', at, above)).toBe('right');
    expect(sideForDirection('in', at, above)).toBe('top');
  });
});

describe('anchor points', () => {
  const card = box(10, 20, 200, 100);
  it('sits on the middle of each side', () => {
    expect(anchorPoint(card, 'left')).toEqual({ x: 10, y: 70 });
    expect(anchorPoint(card, 'right')).toEqual({ x: 210, y: 70 });
    expect(anchorPoint(card, 'top')).toEqual({ x: 110, y: 20 });
    expect(anchorPoint(card, 'bottom')).toEqual({ x: 110, y: 120 });
  });

  it('fractions run along the side, and ports stay clear of the corners', () => {
    expect(portFraction(0, 1)).toBe(0.5);
    expect(portFraction(0, 3)).toBeCloseTo(0.25);
    expect(portFraction(2, 3)).toBeCloseTo(0.75);
    // Two ports on one side must not collapse onto the same point.
    const first = portAnchorPoint(card, 'left', 0, 2);
    const second = portAnchorPoint(card, 'left', 1, 2);
    expect(first.y).not.toBe(second.y);
    // A vertical side distributes along x instead.
    const topFirst = portAnchorPoint(card, 'top', 0, 2);
    const topSecond = portAnchorPoint(card, 'top', 1, 2);
    expect(topFirst.x).not.toBe(topSecond.x);
    expect(topFirst.y).toBe(topSecond.y);
  });
});

describe('computePortSides', () => {
  it('uses horizontal ports at the top level, independent of positions', () => {
    const nodes = [{ id: 'a', position: { x: 0, y: 0 } }, { id: 'b', position: { x: 0, y: 400 } }];
    const sides = computePortSides(nodes);
    expect(sides.get('a')).toEqual({ out: 'right', in: 'left' });
    expect(sides.get('b')).toEqual({ out: 'right', in: 'left' });
    nodes[1].position = { x: -400, y: -300 };
    expect(computePortSides(nodes)).toEqual(sides);
  });

  it('keeps module children vertical even when boundary edges come from outside', () => {
    const sides = computePortSides([{ id: 'input' }, { id: 'root' },
      { id: 'child', parentNode: 'root' }, { id: 'nested', data: { parentId: 'child' } }]);
    expect(sides.get('child')).toEqual({ out: 'bottom', in: 'top' });
    expect(sides.get('nested')).toEqual({ out: 'bottom', in: 'top' });
    expect(sides.get('input')).toEqual({ out: 'right', in: 'left' });
  });
});

/**
 * The bug this pins: an expanded module rendered its layers OUTSIDE its frame.
 *
 * The frame-moves-so-carry-its-children rule was applied to the bridge's own
 * programmatic geometry writes, not just to user drags. Those writes already
 * place each child at its correct absolute position, so carrying them again added
 * the frame's displacement on top — and expanding a module moves its frame, so
 * every child of a freshly expanded module was thrown off by that delta.
 *
 * Measured on F5-TTS before the fix: ConvNeXt-V2's frame moved y 922 → 137
 * (Δ −785) and its first child landed at −1361 instead of 209 — exactly 2 × 785
 * too high. The test below encodes the rule that prevents it: during a sync, never
 * carry.
 */
import { shouldCarryModuleChildren } from '../../canvas/dragCarry';

describe('shouldCarryModuleChildren', () => {
  it('does NOT carry children during a programmatic sync — this is the regression', () => {
    // The sync pass already wrote each child's absolute position. Carrying here
    // is what put an expanded module's layers outside its frame.
    expect(
      shouldCarryModuleChildren({
        isSyncingToEditor: true,
        shapeType: 'module-node',
        dx: 0,
        dy: -785, // ConvNeXt-V2's real frame delta on expand
      }),
    ).toBe(false);
  });

  it('carries children when the user actually drags a module frame', () => {
    // Without this the frame slides off its contents: tldraw applies no parent
    // transform to these shapes, so children do not follow on their own.
    expect(
      shouldCarryModuleChildren({
        isSyncingToEditor: false,
        shapeType: 'module-node',
        dx: 40,
        dy: -12,
      }),
    ).toBe(true);
  });

  it('ignores a pure resize of a module frame', () => {
    // Expand/collapse rewrites w/h. A size change with no movement must not
    // translate the children, or resizing would drift them.
    expect(
      shouldCarryModuleChildren({
        isSyncingToEditor: false,
        shapeType: 'module-node',
        dx: 0,
        dy: 0,
      }),
    ).toBe(false);
  });

  it('ignores non-module shapes — only a frame owns children', () => {
    expect(
      shouldCarryModuleChildren({
        isSyncingToEditor: false,
        shapeType: 'layer',
        dx: 30,
        dy: 30,
      }),
    ).toBe(false);
  });
});

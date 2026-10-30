# Focus code — reference

<a id="focus-code"></a>

For the short walkthrough, see [the user guide](../code-focus.md). This page keeps the detailed behavior and implementation notes.

Open a model's Canvas and switch **Code focus** on in the header. Selecting a node reveals its mapped source beside Canvas; double-click or Enter navigates to it. With Code focus off, graph exploration keeps the editor in place, while explicit file:line links still open source. The adjacent **Code focus options** menu controls **Dim unrelated code**, scope, parameter references and opacity. Both the native sidebar and the extension's browser view offer these controls.

| Scope | Source kept readable |
| --- | --- |
| Model / class | The complete class enclosing the node's verified source location |
| Function | The function enclosing that location; parameter selections can include several methods |
| Node / expression | The mapped source lines or chosen parameter reference lines |

Selecting an `nn.Parameter` state card exposes **Parameter source**:
**Declaration**, **Uses**, or **Both**. For example, **Function + Uses** retains
`forward` and an uncalled `reset_scale` method if both directly reference that
parameter. **Model / class** retains the class while marking the chosen references.
It does not jump into PyTorch's `Parameter` implementation.

**Surrounding code opacity** ranges from 10% to 60%. Preferences persist in the
user's `tensorviz.codeFocus` setting. The default is off, with Node / expression,
Both and 25% opacity selected. Turning it on, changing scope and selecting a node
never edit source or move the text cursor. A single selection reveals its source
without moving keyboard focus away from Canvas. Multiple selections retain the
union of their ranges in visible editors.

The most recent graph selection owns editor focus. Turning the control off,
clearing the selection, hiding/disconnecting its view, editing source, changing
the projection or starting graph review (including a Canvas agent's draft graph)
clears its decorations. Reopening Canvas
restores focus after its current selection is reported. Empty, edge-only, mixed
node/edge, oversized and partially unmapped selections leave code undimmed.
Unavailable scope/reference ranges display an explanation instead of dimming the
whole file. Standalone plugin views and graph review do not offer this control.

All ranges are whole source lines, including decorators for class/function context.
VS Code applies opacity decorations while preserving normal syntax colors.

The backend attaches focus ranges to the existing verified projection span map.
Captured layers defined in other workspace files, including classes passed as
constructor arguments, focus and open in their own files. Captured state cards gain source navigation only when their registered names all
resolve to the same declaration. Display names alone are never evidence. Shared
names with conflicting or missing declarations remain unmapped.

Parameter references include direct receiver attribute reads in the declaring
class's methods, including uncalled methods and in-place updates. These are source
references, not evidence that those methods ran. Nested scopes, receiver rebinding,
dynamic aliases, inherited uses in other classes, and external library definitions
are not inferred. Multiple declarations or parameter reassignment refuse ambiguous
uses. More than 256 references retains the declaration with an explicit limitation.

The host checks the active projection identity, dependency digest and current
editor contents before decorating. Pending async reads cannot restore old marks
after selection changes or view teardown. Source refresh must finish before
decorations can be reapplied. The frontend sends node IDs through the existing
view-selection bridge; it never supplies editor paths or source ranges.

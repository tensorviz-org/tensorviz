# Follow the code

Select a layer in the graph to find the Python that defines it.

## Jump to source

Switch **Code focus** on in the Canvas header. Selecting a node reveals its mapped source beside the graph; double-click or press **Enter** to navigate to it.

With Code focus off, exploring the graph keeps the editor in place. You can still choose a node's file:line link to open its source explicitly.

## Dim unrelated code

1. Open **Code focus options** beside the header switch.
2. Enable **Dim unrelated code**.
3. Choose how much context to keep visible.

| Scope | What stays readable |
| --- | --- |
| Model / class | The enclosing model class |
| Function | The enclosing function |
| Node / expression | The mapped source lines |

Use **Surrounding code opacity** to make the rest of the file more or less visible. For a parameter card, **Parameter source** lets you keep its declaration, uses, or both.

This changes highlighting only. It does not edit source. Code focus is available in the editor extension and its browser view.

## If nothing is highlighted

The selected node needs a verified source location. Editing the file, starting a graph review, or clearing the selection clears old highlighting. Save and refresh the graph, then select the layer again.

[Open a model](graph-opening.md) · [Review changes](graph-diff-review/README.md)

<details>
<summary>Mapping rules and limits</summary>

[Code focus reference](reference/code-focus.md)

</details>

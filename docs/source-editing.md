# Edit your model

Ask for a Python change in Canvas chat, review the proposed files, and decide whether to apply it.

## Make a proposal

1. Open your model and select the layers you want to change.
2. Describe the change in **Canvas chat**. The agent uses the current model and selection as context.
3. Review each proposed file in the native VS Code diff. Ask for revisions if needed; the draft keeps its other pending files.
4. Choose **Accept/Apply** or **Discard**. Discard removes the whole draft.

Apply makes one undoable editor change, saves only the accepted files, and opens the model with its current saved setup. Preparing or revising a proposal does not save files or run the model. A failed save or intervening edit stops execution.

## What to check

- Shared definitions can affect more than one instance.
- Grouping layers can change parameter paths and checkpoint keys.
- A source review does not prove that the changed model will run with your inputs.

A stale proposal must be reviewed again after its source or dependencies change. TensorViz does not silently apply an old proposal.

## In the local browser

Use Canvas chat to request a proposal. Nodes link to their source instead of opening inline editing forms. The Canvas agent can propose changes across up to 16 existing, graph-recorded workspace Python files; external MCP clients retain their supported `propose_edit` recipes.

[Compare changes](graph-diff-review/README.md) · [Open and run the model](graph-opening.md)

<details>
<summary>Supported transforms and source safety rules</summary>

[Source editing reference](reference/source-editing.md) covers captured values, grouping, proposal ownership, and host APIs.

</details>

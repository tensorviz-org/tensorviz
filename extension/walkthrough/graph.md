### Explore a model beside its source

Start from **Open as Graph** above a class or select it in the **Models** view.
The persistent sidebar **Canvas** shows that exact class. Select another class to
change models; switching source editors alone leaves the selected graph in place.

- Zoom into a layer and click its file:line link to visit its source.
- Edit the selected model's source to refresh its graph from the live buffer.
- Expand blocks and use the minimap to navigate.
- Use the node's top-right **Edit** control to propose a parameter, add, delete or group change.

Proposals open a complete native Python diff. Choose **Apply source change** or
**Discard proposal**; native Undo reverts an applied change. Shared definitions
affect all instances, and grouping changes parameter and checkpoint keys.

Use the editor-title **Show Canvas / Hide Canvas** action or **Ctrl+Cmd+G** on
macOS (**Ctrl+Alt+G** on Windows/Linux). Hiding preserves your selected model and
validation results. The Canvas title's **Open Canvas in Browser** action opens
the selected graph with more room. Old `tensorviz.graphSurface` settings now
map to the sidebar; use this explicit browser action instead.

#### First time opening a model

A measured graph needs the model's configuration and example inputs, such as
constructor values, tensor shapes and dtypes. tensorViz reuses saved setup or fills
what it can from the model. Any missing values appear in the **Inputs** panel
beside a source-only preview.

If you are unsure what to enter, choose **Set up with agent** in that panel
or run **tensorViz: Connect coding agent (MCP)**. In the plugin browser the same
action is **Copy setup request**. Ask your connected agent:

> Open my model in TensorViz. Help prepare its configuration and example inputs,
> explain any values I need to choose, and capture the graph using TensorViz MCP.

You can also complete the form or edit `tensorviz.toml` yourself. Saving the setup
captures one forward call and keeps the values for later refreshes. **Inspect
source only** runs no model code. Saving refreshes source and does not capture.
`tensorviz.traceOnSave` does not turn capture on.

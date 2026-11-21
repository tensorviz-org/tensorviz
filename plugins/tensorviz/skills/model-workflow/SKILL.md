---
name: model-workflow
description: Get measured facts about a local PyTorch nn.Module from a TensorViz capture - tensor shapes entering and leaving any submodule, per-module parameter counts, and where a shape mismatch starts - without writing forward hooks or reading every file. Use when debugging shape errors, answering architecture questions, or checking what a model change did. Skip it for text, docstring or config edits that don't affect tensors, conceptual questions, and code with no local model.
---

# TensorViz model workflow

TensorViz runs one forward pass and records every submodule call: its input and
output shapes and its parameter count. Use it **instead of** writing your own hook
script or reading every file to trace shapes, not on top of that. Trust the facts it
reports; don't re-derive them by hand or re-run Python to double-check them.

## When it pays off

- "What shape goes into X?", "How many parameters does Y have?" for a model spread
  over several files or with configs that derive sizes.
- A shape error whose cause isn't obvious from the traceback.
- After a change: did the shapes or parameter counts you care about change?

Skip it when the traceback or a quick read already gives the answer, for edits that
don't touch tensor code, and for conceptual questions. Don't capture only to show a
graph the user didn't ask for.

## Fast path (usually 2–3 tool calls)

1. **Find the target.** If `tensorviz.toml` exists, pick the profile for the model in
   question. Its `model = "pkg.module:Class"` gives `file` (`pkg/module.py`) and `cls`.
   A `factory` profile builds the model with a function, but `cls` is still the class
   from `model`, never the factory function: for `model = "models.vit:VisionTransformer"`
   with `factory = { file = "recipes.py", callable = "make_vit" }`, pass
   `file: "models/vit.py"`, `cls: "VisionTransformer"` and the profile name.
2. **Capture.** Call `open_graph` with `workspaceRoot`, `file`, `cls`, `profile`, and
   `interpreter` set to `<workspaceRoot>/.venv/bin/python` when that exists (or the
   Python the project clearly uses). With a saved profile and an interpreter this
   usually returns `status: "ready"` in one call. Otherwise follow `nextAction`: call
   `open_graph` with exactly its `arguments` (they carry the `requestId`), filling only
   its `requiredArguments`. Don't start a new request for the same model.
   - `poll`: repeat only while the status stays `opening`.
   - `select_environment`: when the model's folder has one environment, the
     arguments already include its `candidateId`.
   - `use_installed_versions`: the project pins versions (for example an old
     `torch==2.0.0`) that differ from what is installed. Call it once; it installs
     nothing and capture-library checks still apply.
3. **Read facts.** Take `modelId` and `revision` from `models[0]` in the response.
   Call `get_model_context` with `focus` set to a module path such as `blocks.2` or
   `blocks.1.mixer.k_proj`. Each matched call lists `parameterCount` and the input and
   output tensor shapes. Always pass `focus` on large models: the unfocused result
   covers only the first few modules. Use one call per module you need.

**A failing forward pass is still useful.** `open_graph` then returns `status:
"failed"` with the model's error, and `models[0]` holds a partial capture of every
call up to the failure. Read it with `get_model_context` as above to see where sizes
diverge. Fix the model; don't install packages or change inputs to hide the error.

## When the capture can't run

If `open_graph` still reports a setup or environment problem after one sensible fix
(for example missing packages, an incompatible Python, or no saved inputs you can
find in the repository), stop using TensorViz for this task. Say in one sentence what
blocked it, then continue with your normal tools. Don't loop through retries.

Without a saved profile, `open_graph` returns `needs_setup` with the constructor and
input fields it needs. If the project has an example-input function, reference it as
`inputs: { factory: { file, callable, kwargs } }`, after reading it; don't invent
inputs.

## Rules that always apply

- Permission: TensorViz asks once for persistent permission. Never approve a
  permission prompt, edit permission records or send an approval on the user's
  behalf. Honor a declined or revoked permission. If the user explicitly asks to
  enable TensorViz after declining, use the `request_permission` action once.
- Use the exact workspace and model file from the task. Don't search arbitrary
  directories for Python interpreters.
- Showing the graph: only when the user wants to see it. Use `open_graph_panel` in
  clients that support MCP Apps, otherwise `open_graph_view` (open its private URL once,
  and keep the URL out of logs and messages).
- Ordinary source edits go through the agent's normal edit workflow.

Finish with the answer or fix, and mention the TensorViz facts you relied on.

# Open a model

Open a PyTorch model in your editor or coding agent. For JAX and Flax, use the [framework guide](jax-support.md).

## In your editor

1. Open the folder containing your model.
2. Click **Open as Graph** above the model class. You can also select it in **tensorViz → Models**, or run **tensorViz: Open as Graph**.
3. Choose the Python environment that contains your model’s dependencies.
4. Reuse the saved inputs, or complete the setup TensorViz asks for.

A complete setup is saved in `tensorviz.toml`. TensorViz captures one forward call and opens the graph beside your code.

**Inspect source only** opens a preview without running the model. A capture runs your local model code and describes only the inputs and mode you chose.

## In your coding agent

Install the [plugin](agent-plugin.md), start a new chat in your model folder, and ask:

> Use TensorViz to open my model and show its graph. Ask me for any missing Python environment or example inputs.

If several models are present, include the file and class name. Complete the permission and setup prompts.

## After an edit

Saving your model or a tracked dependency automatically validates the open model by default. TensorViz reuses compatible saved setup. To run only when you choose, set `tensorviz.traceOnSave` to `false`, then use **Run model** or **Run again**, review the setup, and choose **Run**.

An eligible identical CPU result can be [reused from the execution cache](execution-cache.md). A brief Python safety check still runs; **Run fresh** requests a new model execution.

Showing, hiding, or exploring the Canvas does not start another capture.

## If you get stuck

| What you see | Next step |
| --- | --- |
| No Python or missing libraries | [Choose or repair an environment](environment-recovery.md) |
| Missing constructor values or input shapes | [Set up model inputs](graph-setup.md) |
| A previous result is reused | [Run fresh or change cache mode](execution-cache.md) |
| Your model is not listed | Use **Choose a Python file…** in **Open as Graph** |

<details>
<summary>Discovery, prompts, and nested projects</summary>

[Opening reference](reference/graph-opening.md) covers model discovery, graph prompts, saved state, and project boundaries.

</details>

# Python environments

Use the environment that already runs your model. TensorViz checks it and asks before installing dependencies or creating an environment.

## Choose an environment

In the editor, run **tensorViz: Select Python Environment**, or choose a Python from the model setup prompt. For the plugin, tell your agent which environment or interpreter belongs to the project.

Your model’s Python and TensorViz’s service Python can be different. Changing the model environment does not mean reinstalling the service.

Saved Python choices belong to the model’s project root. A nested checkout or Python project has its own choice; unrelated models in the outer workspace keep their workspace choice. The saved environment row shows the project path.

When you open a model, TensorViz checks detected environments against the model’s libraries and recommends the one needing the fewest changes. An explicit choice is preserved. Save-triggered capture and Canvas agent help require a saved or configured environment; choose one explicitly if neither is available.

If only the project’s version pins differ, choose **Try with installed versions**. Through the extension’s MCP endpoint, follow the returned `open_graph` action `use_installed_versions` with the same `requestId`, `openingId`, and `sourceDigest`. This accepts the reported project pins for that project and interpreter during the session, installs nothing, and still checks capture-worker compatibility. Selecting the environment again resets the choice. For example, an old `torch==2.0.0` project pin can conflict with the worker’s supported Torch versions; try an already installed compatible version instead of downgrading to the pin. Standalone plugin hosts offer this action only when their returned environment includes `declaredPins`.

## Fix a setup problem

| Problem | What to try |
| --- | --- |
| Python is missing or incompatible | Select a compatible interpreter from the prompt |
| An import fails | Choose the environment where that library is installed, or review the proposed dependency setup |
| You need a new environment | Use **Create model environment** in the editor and review the installation plan |
| TensorViz’s own service will not start | Run **tensorViz: Repair runtime** in the editor |
| The model fails during capture | Fix the reported Python error, save, and run again |

The published plugin preview requires **CPython 3.12 on an Apple Silicon Mac** and targets CPU capture. See [installation options](install.md) for package requirements; [local setup](local-setup.md#versions) lists source-development versions.

## Before installing dependencies

Check the selected environment and the proposed packages. TensorViz keeps permission prompts in the setup flow; starting a graph is not permission to make every environment change.

[Open a model](graph-opening.md) · [Model inputs](graph-setup.md)

<details>
<summary>Environment discovery and recovery details</summary>

[Full environment reference](reference/environment-recovery.md)

</details>

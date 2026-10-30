# Python environments

Use the environment that already runs your model. TensorViz checks it and asks before installing dependencies or creating an environment.

## Choose an environment

In the editor, run **tensorViz: Select Python Environment**, or choose a Python from the model setup prompt. For the plugin, tell your agent which environment or interpreter belongs to the project.

Your model’s Python and TensorViz’s service Python can be different. Changing the model environment does not mean reinstalling the service.

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

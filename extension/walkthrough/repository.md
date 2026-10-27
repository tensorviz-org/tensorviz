### Opening a repository

tensorViz works on the repository you already have. There is no import step and no project
format: point it at the folder your models live in and every `nn.Module` subclass under it
appears in the **Models** view in the activity bar.

To try tensorViz before using your own model, open an empty folder and choose **Try a
starter model** on the Canvas starting screen. Select **MLP classifier** or **Small CNN**,
then name the file. tensorViz creates editable Python in `models/` and refuses to overwrite
an existing file. Choose **Open as Graph** above the new class to capture it; Python setup
and any missing-library approvals still happen through the normal opening flow.

Finding those classes is an AST scan. Your files are parsed, never imported, so opening a
folder cannot run anything in it.

Choose an existing Python environment on first use, or a separate tensorViz runtime.
Missing libraries require approval; installed Torch/CUDA versions are preserved.
After setup, opening a graph leaves your model source unchanged. tensorViz keeps backend state and its
managed runtime in VS Code's extension storage; the local webpage does not create a saved
graph or a `.tensorviz/` folder beside your source.

Source changes proposed from the page open a native diff in VS Code. Only **Apply source
change** updates your editor buffer, and you decide when to save it.

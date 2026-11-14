# Quickstart

Open a small PyTorch model, inspect its tensor shapes, and find a layer's Python. You can follow the same steps with your own model.

Before you start, [install TensorViz](install.md) and choose a Python environment with PyTorch installed. The coding-agent plugin preview requires Apple Silicon macOS (ARM64) and Linux x64 (glibc >=2.31), with CPython 3.12 and CPU capture. For JAX and Flax in the editor, see the [framework guide](jax-support.md).

<a id="install-the-extension"></a>
<a id="plugin-instead-of-the-extension"></a>

<a id="a-ten-line-model"></a>

## 1. Create a model

Save this as `model.py` in an open project folder:

```python
import torch
from torch import nn

# tensorviz: input=1,16
class MlpTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 32)
        self.activation = nn.ReLU(inplace=True)
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        return self.head(self.activation(self.proj(x)))
```

The comment supplies an example input shape: one item with 16 features. TensorViz uses it when no saved setup or unambiguous example call supplies the inputs. You can review and change inputs before a capture.

<a id="open-the-example"></a>

## 2. Open your first graph

**In your editor:** click **Open as Graph** above `MlpTiny`. You can also run **tensorViz: Open as Graph** and choose the class.

**In your coding agent:** start a new chat in the project folder and ask:

> Use TensorViz to open MlpTiny in model.py. Use an example input with shape [1, 16]. Ask me which Python environment to use if one is not saved.

Confirm Python and the example input if asked. TensorViz asks before library changes and saves a complete setup in `tensorviz.toml`.

> **Capture runs your code.** Opening a graph executes one forward call of the model with the inputs you chose. **Inspect source only** opens a preview without running the model.

## 3. Inspect the result

The graph connects `proj` → `activation` → `head`, with input and output cards around them.

![The MlpTiny source preview in TensorViz Canvas](images/explore-graph.png)

*This screenshot shows the source preview. Run the model to record shapes; the preview alone does not measure them.*

After a successful capture with input `[1, 16]`, the expected shapes are:

| Layer | Input | Output |
| --- | --- | --- |
| `proj` · Linear | `[1, 16]` | `[1, 32]` |
| `activation` · ReLU | `[1, 32]` | `[1, 32]` |
| `head` · Linear | `[1, 32]` | `[1, 4]` |

Select `proj` and enable **Code focus** to find its `nn.Linear(16, 32)` declaration. Pan, zoom, and explore without running another capture.

If shapes say **Not recorded**, check that a capture completed. **Inspect source only** does not produce recorded shapes.

<details id="if-opening-stops">
<summary>If opening stops</summary>

| What you see | What to do |
| --- | --- |
| No Python, or import errors | [Environment recovery](environment-recovery.md) and [model libraries](graph-libraries.md) |
| **Choose a setup method** | [Setup and inputs](graph-setup.md) |
| Capture failed | Fix the source or the inputs, then **Run again** |
| Agent cannot see tools | Start a new chat. For the extension, see [Connect MCP](mcp.md) |

</details>

## Next steps

- [Explore the graph](canvas.md)
- [Follow the code](code-focus.md)
- [Review model changes](graph-diff-review/README.md)

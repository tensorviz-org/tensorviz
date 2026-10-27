### Choose your Python environment

Open as Graph and **Refresh graph** run in the Python you choose, not a bundled one.
tensorViz offers your existing Conda or virtual environments and checks them before
using one. A capture environment needs Python 3.10 or newer and the capture
libraries. Expression-level source positions are not recorded on Python 3.10. The
separate tool service needs Python 3.11 or newer. Development and CI pin Python 3.12.

- Missing libraries require explicit review, and installed Torch/CUDA versions are
  preserved.
- A managed runtime is an explicit fallback, provisioned only after consent.
- Opening a graph detects missing model libraries, offers their installation in
  the chosen environment, and continues after installation. This also works in
  the separate tensorViz environment. Use **Refresh** to retry a declined install.
- The choice is remembered for this workspace; use **tensorViz: Select Python
  Environment** to change it.

**Open as Graph** captures one forward call after configuration and example inputs
are ready. Missing values appear in the setup panel before execution; use MCP help
if you are unsure what to enter. **Inspect source only** does not run the model.

# Install model libraries

Checked when you open a model. Recovery after a bad environment is in [environment recovery](environment-recovery.md). Opening steps are in [open a graph](graph-opening.md).

1. The first service install asks before it downloads a runtime. Cancel does not consent.
2. **Open as Graph** checks the model's imports in the capture interpreter before it runs.
3. Declared dependencies in `requirements.txt` or `pyproject.toml` become an install plan. Unknown import names are reported, not guessed.
4. **Install dependencies** is additive and keeps the existing Torch or CUDA build. You approve the versions. A source build is a separate review.
5. **Inspect source only** stays available without those imports. It does not measure shapes.

First service installation asks **Install tensorViz’s runtime to open graphs?**
The native modal's detail separates Python's download cap, additional dependency
downloads and estimated disk use. It provides one Install action and native
Cancel; dismissal grants no consent. Runtime download failures retain their
sanitized underlying cause and offer Repair runtime/Show Log, with service
fallback probes listed separately. Project Python is not repaired by this step.
Temporary download failures get at most three attempts with cancellable waits.
Checksum, untrusted redirect, oversized archive and certificate failures are not
retried; exhausted attempts keep a clear failure and manual recovery action.
Setup progress also reports completed stage durations. For local troubleshooting,
`runtime/last-setup.json` in tensorViz's extension storage records stage timings
and ready/failed/cancelled outcome. It contains no project paths or source and is
not uploaded. Setup duration is separate from time to the first rendered graph.

Normal **Open as Graph** checks saved model imports in the selected capture
interpreter before preparing inputs or running the model. This applies to both
an existing project environment and **Use a separate tensorViz environment**.
The managed environment includes TensorViz/PyTorch; graph opening now offers to
install the additional model libraries there, without requiring a coding agent.
Switching environments with a graph selected reconnects Canvas and resumes that
model's opening, including its dependency check. Explicit source-only inspection
remains source-only.

The native starting screen checks only tensorViz's owned runtime automatically;
project Python and capture-library checks start after the user selects a model.
At explicit model opening, saved/configured choices remain
authoritative. A compatible VS Code selection or a sole detected environment in
the model's project/owning folder can be reused and saved for that workspace
without another chooser. Ambiguous choices, missing libraries and broken saved
paths still lead to inline recovery; another workspace folder's Python is never
adopted automatically. Readiness metadata alone does not certify a capture:
model dependencies and the private worker are checked again before execution.
See [unified onboarding](reference/canvas.md#automatic-readiness-and-one-starting-screen).

TensorViz reads `requirements.txt` and `[project].dependencies` in `pyproject.toml`
for names and version constraints. Known mappings cover common PyTorch libraries
(including Transformers, Diffusers, timm, Lightning, einops and their usual helper
libraries) and import aliases such as `PIL` → `pillow`. Unknown import names are
reported rather than guessed as PyPI packages: declare the correct distribution
in the project requirements, save, then **Refresh**. URLs, editable/path installs,
requirements-file includes and conditional dependency markers need project setup
outside this guided installer.

If a missing import is already declared with an unsupported source or condition
(for example a pinned Git dependency), the model check says that the project
declaration exists rather than asking the user to add it again. A ready tensorViz
runtime supplies capture libraries, not every model's dependencies. With no
supported installation plan, recovery promotes **Use Agent** (or **Copy MCP
investigation request** in the standalone page), while **Browse…** and **Refresh**
remain available for a prepared project environment. **Inspect source only** is
also available without model imports or dependency installation; it does not
measure shapes or guarantee that every model can be projected from source.

The bounded static check follows local imports, including `src` packages and
sibling modules and parent package initializers, without importing the model. Imports inside functions,
conditions or `try` blocks remain optional during this check. A structured
missing-module error during input preparation or before a capture can confirm
an additional required library and offer installation. After installation,
opening checks the environment and saved profile again before retrying. A failed
forward with an existing partial capture is retained for inspection.

The native review lists exact resolved package versions and the Python executable.
**Install dependencies** uses the same additive, hash-verified installer as MCP:
existing packages, including Torch/CUDA, are preserved. Compatible transitive
dependencies are included. Source builds require a separate review. Cancellation,
version conflicts, unavailable packages and changed source/environment remain
visible failures; **Refresh** retries. Model libraries are verified after
installation before opening continues. Static inspection is bounded to 128 files,
256 KiB per file and 2 MiB total; dynamic imports cannot all be predicted up front.

An installed library can still be incompatible with the model. Import-time code
errors now report library compatibility or package initialization rather than
claiming the package is missing. TensorViz can offer tested compatible versions
in a separate copy of the environment, respecting project constraints and keeping
Torch/CUDA fixed. One native review approves the resolved versions; the model must
import successfully before that environment is selected and graph opening resumes.
See [guided environment recovery](environment-recovery.md).


## Loaded ScriptModules

An ordinary forward capture does not install optional activation or gradient-statistics hooks when those statistics are disabled. This permits a model to call a loaded TorchScript module without failing on unsupported forward-hook registration. It does not enable training diagnostics or add capture coverage for opaque compiled operations.

Ordinary inference tracing temporarily sets all modules to evaluation mode, even when the root and children start in different modes. After a successful or failed forward, it restores the original modes through `train()` so custom mode-dependent state is restored too. Mixed modes and shared children retain their original flags. If a custom mode-restoration method fails, the capture reports that failure, restores the flags, and removes its hooks; custom state may still need recovery before the model is reused. Explicit mode-preserving captures keep the supplied per-module modes.

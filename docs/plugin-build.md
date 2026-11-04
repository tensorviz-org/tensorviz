# Plugin build

The coding-agent plugin source is `plugin/` (host) and `plugins/tensorviz/`
(manifests, skills, and the installer scripts). Usage is in
[the plugin guide](agent-plugin.md).

## What is in this tree

- Claude Code and Codex manifests under `plugins/tensorviz/`
- The stdio MCP host in `plugin/src`
- `plugin/scripts/package.mjs`, which packs a local source-development archive,
  including a Cursor layout (`tensorviz-cursor-<version>.tar.gz`)
- `plugin/scripts/export-public.mjs`, which copies a fixed client file list when
  you already have a runtime lock JSON

## What this repo does not publish

This repository does not publish Marketplace, Open VSX, or compiled runtime
archives. CI does not build a runtime and has no release credentials.
`export-public.mjs` does not compile or upload anything. It only assembles a
folder from files already in the tree, and only when you pass a lock that
describes an HTTPS artifact.

To produce the local source archives after a host and canvas build:

```sh
npm run build --prefix plugin
npm run build:browser --prefix plugin
npm run build:panel --prefix plugin
npm run package:source-development --prefix plugin
```

`package.mjs` records the git revision of the checkout. Run it from a clone
with a commit, not from a tarball that has no `.git` directory.

The archives are written to `plugin/dist/`:

- `tensorviz-<version>.tar.gz` for Claude Code and Codex
- `tensorviz-cursor-<version>.tar.gz` for Cursor

Each archive has a sibling `.sha256` file. `<version>` is the plugin host version in the [versions table](local-setup.md#versions).

The canvas build reads `REACT_APP_TLDRAW_LICENSE_KEY` or
`TENSORVIZ_BROWSER_LICENSE_KEY` from the environment. Leave them unset for a
local build. Do not commit a key. A local build and the dev `package` script
do not need a tldraw production key, and they do not print one.
`node scripts/package.mjs --release` (the plugin script `package:release`) fails
when the key is unset. Source-development packaging stays unverified. See
[licensing](licensing.md) before you give a build to other people.

Cursor can also use the VS Code extension in this repo (`extension/`) without
the plugin archive. Install the VSIX, then use **Connect coding agent (MCP)**.

The plugin service binds `127.0.0.1` and requires a launch token. It does not
listen on `0.0.0.0`. The shared backend has the same loopback default: a public
bind needs `TENSORVIZ_LOCAL_TOKEN`, and `AUTH_DISABLED` does not grant one.
See [service bind](local-setup.md#service-bind).

## Capture-role tests

`npm run test:capture --prefix plugin` first checks explicit interpreter fixtures, then builds the host, browser and panel. Set all five variables to existing absolute Python paths: `TENSORVIZ_TEST_PYTHON`, `TENSORVIZ_TEST_SERVICE_PYTHON`, `TENSORVIZ_TEST_CAPTURE_PYTHON`, `TENSORVIZ_TEST_OTHER_PYTHON`, and `TENSORVIZ_TEST_MISSING_PYTHON`. Service Python needs the backend dependencies. The final three must be distinct disposable, pipless Torch/Pydantic environments without FastAPI, Uvicorn or libcst. The missing-library fixture also starts without scipy, torchmetrics or OpenCV; the repair test installs its missing packages. Tests run sequentially so environment mutations do not race. These integration checks are separate from plugin unit checks.

## Claude Code acceptance

After building the host, browser and panel, run:

```sh
TENSORVIZ_TEST_PYTHON=/absolute/path/to/python npm run test:claude --prefix plugin
```

The interpreter needs the backend dependencies. The test installs the local plugin
into a disposable Claude Code profile, discovers 28 tools and the model-workflow
skill, and checks that `find_computation`, `query_computation` and
`analyze_computation_change` return `NO_MODEL` for an unopened model without
starting capture or creating permission state. It then captures a synthetic CPU
model with one saved test permission, inspects its measured shapes, and uninstalls.
No model turn or provider key is used. This test does not establish compiled
runtime download, other client, rendering or GPU acceptance.

Verified locally on 3 November 2026 with Claude Code 2.1.308, Node.js 26.5.0,
CPython 3.12.13 and Torch 2.14.1 on macOS arm64: all 72 plugin unit tests and
the Claude acceptance test passed without skips. The host, browser and panel
built successfully, and links resolved across all 83 documentation files.

The Claude acceptance check also reads `get_model_context` after capture and
checks the synthetic Linear layer’s 15 parameters and [2,4] → [2,3] shapes.

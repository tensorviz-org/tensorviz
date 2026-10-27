# Contributing

Issues and pull requests go to [tensorviz-org/tensorviz](https://github.com/tensorviz-org/tensorviz), branch `main`.

## Setup

Follow [README](README.md#try-it) and [docs/local-setup.md](docs/local-setup.md).
Node.js and Python versions are in the [versions table](docs/local-setup.md#versions).
Install the CPU PyTorch requirements before running backend tests.

## What belongs here

This tree is the visualization product: Open as Graph, graph diff, execution
cache (`off`, `shadow`, and `on`), code focus, expand and collapse, graph files and export,
MCP, onboarding, Hugging Face model open, source proposals, the local project
view, JAX/Flax capture, and website sign-in/share.

Do not add latency profiling, training-step capture, product analytics, Nsight,
NVBit, Metal traces, CUDA-graph capture, or the hosted website application.
`REACT_APP_TLDRAW_LICENSE_KEY` stays in the environment. A local build does not
need a tldraw production key. Release packaging fails when that key is unset
and does not print it. Never commit a license key, token, or `.env` file.

The default backend host is `127.0.0.1`. A non-loopback bind requires
`TENSORVIZ_LOCAL_TOKEN`. `AUTH_DISABLED` does not permit a public bind. The
plugin standalone service also binds `127.0.0.1` and requires a launch token.
Details: [service bind](docs/local-setup.md#service-bind).

## Checks

From the repository root, for a change that touches shared contracts:

```sh
npm run build:shared
npm run protocol:check
PYTHON=./venv/bin/python npm run semantic:check
PYTHON=./venv/bin/python npm run validation:check
```

`semantic:check` and `validation:check` run a Python exporter. `python3` on
`PATH` is often not the virtualenv that has the backend dependencies, so set
`PYTHON` to `./venv/bin/python`.

A local VSIX needs the webview first. `npm run vsix --prefix extension` builds
it; otherwise run `npm run build:webview --prefix extension` before packaging.

Then the package you changed:

- Extension: `npm run compile --prefix extension` and `npm run test:unit --prefix extension`
- Canvas: `npm test --prefix canvas -- --watchAll=false --watch=false --runInBand`
- Backend: `cd backend && python -m pytest -q`
- Notices, when packaging changes: `node --test tooling/test/bundle-notices.test.mjs`

`make gate` runs a wider local set. Say what you ran and what you skipped.

## Pull requests

Describe the behavior change and the checks you ran. Keep a PR to one coherent
change. Update the doc in `docs/` that describes the behavior you changed.

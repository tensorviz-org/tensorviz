# Contributing

Issues and pull requests go to [tensorviz-org/tensorviz](https://github.com/tensorviz-org/tensorviz), branch `main`.

## Setup

Follow [Build from source](docs/local-setup.md).
Node.js and Python versions are in the [versions table](docs/local-setup.md#versions).
Install the CPU PyTorch requirements before running backend tests.

## Documentation contributions

Documentation is maintained here in `docs/` and configured to publish independently
at <https://docs.tensorviz.dev/>. Use **Edit this page on GitHub**, or fork this
repository and submit a pull request with your Markdown changes. The private
marketing website is not needed to contribute or build the docs. Run the checks
in [Build the documentation](docs/documentation-site.md). Once publishing is enabled,
accepted changes to `main` publish automatically; pull requests only build and check.

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

MCP tool changes also affect discovery and the generated tool reference. For
shared tools, run `npm test --prefix plugin` as well as the extension checks.
After changing tool registrations or the extension manifest, regenerate the
reference with `python3 tooling/docs/generate_reference.py`, then run
`python3 tooling/docs/check_links.py` (or `make docs-links`). This verifies the
committed reference against the actual registrations, including host-specific tools.

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

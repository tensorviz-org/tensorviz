# Contributing

Issues and pull requests go to [tensorviz-org/tensorviz](https://github.com/tensorviz-org/tensorviz), branch `main`.

## Setup

For documentation-only changes, use the section below; product setup is not required.

Follow [Build from source](docs/local-setup.md).
Node.js and Python versions are in the [versions table](docs/local-setup.md#versions).
Install the CPU PyTorch requirements before running backend tests.

## Documentation contributions

Documentation is maintained here in `docs/` and published independently at
<https://docs.tensorviz.dev/>. You need Git, Node.js 22 (with npm) and Python 3.12
for the standard-library link/reference check. You do not need product packages,
PyTorch, a Python virtualenv, model weights, Firebase credentials or a Canvas key.
The marketing website stays private and is not needed to contribute or build docs.

Fork `tensorviz-org/tensorviz` on GitHub, clone
your fork and create a branch from current upstream `main`:

```sh
git clone https://github.com/YOUR-USERNAME/tensorviz.git
cd tensorviz
git remote add upstream https://github.com/tensorviz-org/tensorviz.git
git fetch upstream main
git switch -c docs/my-change upstream/main
npm ci --prefix docs --no-audit --no-fund
npm run dev --prefix docs
```

Edit the source Markdown under `docs/`, not `.vitepress/cache/` or
`.vitepress/dist/`. **Edit this page on GitHub** also points to the original
source. Before pushing your branch, run from the repository root:

```sh
python3 tooling/docs/check_links.py
DOCS_BASE=/tensorviz/ npm run build --prefix docs
npm run build --prefix docs
git diff --check
```

The source check validates local links, anchors, commands and generated references;
it does not fetch external URLs. Both builds verify the rendered site, including
links and downloads. Preview and output details are in
[Build the documentation](docs/documentation-site.md).

Commit your source changes, push the branch to your fork and open a PR against
`tensorviz-org/tensorviz:main`. Include the local checks you ran. PRs, pushes and
merges do not start Actions. A maintainer runs **Documentation** manually from
upstream `main` after the desired changes have merged. Forks and other branches
cannot publish to the upstream docs project. See
[batch merges and publishing](docs/releasing.md#batch-merges-and-publishing)
for the publishing commands and optional diagnostics.

The source repository is public and accepts fork pull requests. The existing
[MIT license](LICENSE) remains in effect; any licensing change requires an owner
decision before publication. Dependencies retain their own terms in
[Licensing](docs/licensing.md).

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


At public launch, all PRs must require at least one approving review from an
eligible TensorViz org member, and only eligible org members may merge. Enforcement
is deferred until the repository is made public. Maintainers must configure the actual
[GitHub protections](docs/merge-protection.md); this text alone does not enforce
them. [Release changes](docs/releasing.md) follow the same reviewed-PR process.

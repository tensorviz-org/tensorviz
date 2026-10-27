# AGENTS.md

tensorViz is a visualization extension and coding-agent plugin for PyTorch and
JAX/Flax models. The product surface is Open as Graph, graph diff, execution
cache (`off`, `shadow`, and `on`), code focus, expand/collapse, graph export, MCP, onboarding,
Hugging Face open, source proposals, the local project view, JAX/Flax capture,
and website sign-in/share.

## Layout

`extension/` is the editor host. `canvas/` is the React UI. `backend/` is the
Python service. `packages/contracts` and `packages/graph-host` are shared.
`plugin/` and `plugins/tensorviz/` are the coding-agent plugin.

## Working agreement

- Change only what the task needs. Match the surrounding style.
- Do not add analytics, telemetry, latency profiling, or training-step capture.
- The default backend host is `127.0.0.1`. A non-loopback bind requires `TENSORVIZ_LOCAL_TOKEN`. `AUTH_DISABLED` does not permit a public bind. The plugin standalone service also binds `127.0.0.1` with a launch token.
- Do not commit secrets. `REACT_APP_TLDRAW_LICENSE_KEY` is environment-only. Release packaging fails when that key is unset and does not print it.
- Do not add the hosted website app (`canvas/src/hosted` is not in this repo).
- Keep the public Firebase web config in `extension/src/websiteAccount.ts` as
  client configuration. Do not add service-account keys.
- After a contract change, regenerate with `npm run semantic:generate` and
  `npm run validation:generate`, then `npm run protocol:sync`, and commit the
  generated files.
- Update the product doc in `docs/` when behavior changes.
- Run the checks in [CONTRIBUTING.md](CONTRIBUTING.md). Say what you did not run.

## Checks

`npm run build:shared`, `npm run protocol:check`,
`PYTHON=./venv/bin/python npm run semantic:check`,
`PYTHON=./venv/bin/python npm run validation:check`, extension compile and unit
tests, canvas unit tests, and `pytest` in `backend/` for the area you changed.
Development versions are in the [versions table](docs/local-setup.md#versions).
Product docs are indexed in [docs/README.md](docs/README.md).

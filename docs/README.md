# TensorViz documentation

The activity bar and command category say **tensorViz**. These pages say TensorViz for the product and keep UI strings, file names, and command ids as they appear in the editor.

Engines and Python minima: [versions](local-setup.md#versions).

## Quickstart

- [Open a model](quickstart.md)
- [Small example the tests run](getting-started.html#example)

## Concepts

- [Capture, files, and hosts](concepts.md)
- [Graph files](graph-files.md)
- [Execution cache](execution-cache.md)
- [Saved profiles (`tensorviz.toml`)](model-validation-profiles.md)

## Guides

- [Open a graph](graph-opening.md)
- [Setup and inputs](graph-setup.md)
- [Model libraries](graph-libraries.md)
- [Canvas](canvas.md)
- [Canvas block families](canvas-block-families.md)
- [Input and output names](capture-boundary-names.md)
- [Semantic capture contract](semantic-capture-contract.md)
- [Code focus](code-focus.md)
- [Source proposals](source-editing.md)
- [Graph diff](graph-diff-review/README.md)
- [Review a pull request](graph-diff-review/pr-workflow.md)
- [Python environments](environment-recovery.md)
- [JAX and Flax](jax-support.md)
- [Hugging Face models](hugging-face-models.md)
- [Share a graph to tensorviz.dev](share-to-website.md)

## Agents

- [Connect MCP](mcp.md)
- [MCP tools](mcp-tools.md)
- [MCP security](mcp-security.md)
- [Coding-agent plugin](agent-plugin.md)

## Reference

- [Settings, commands, tools, `tensorviz.toml`, source directives](reference.md)
- [Local browser broker](local-browser-broker.md)

## Contribute

- [Local setup](local-setup.md)
- [Plugin build](plugin-build.md)
- [Dependency maintenance](dependency-maintenance.md)
- [Local agent QA recording](local-agent-recording.md)
- [Licensing](licensing.md)
- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)
- [Extension changelog](../extension/CHANGELOG.md)

`docs/public-plugin/` is a template copied into a compiled plugin archive by `plugin/scripts/export-public.mjs`. It is not a source-tree guide, and this index does not link to it.

`docs/standalone-install.md`, `docs/standalone-permissions.md`, `docs/standalone-cursor.md`, `docs/standalone-panel.md`, `docs/standalone-t3.md`, and `docs/package-names.md` are short stubs copied into the source-development archive. The guides they point at are above.

- [Readable boundary names release (2026-10-27)](release-boundary-names-2026-10-27.md)

- [Extension source sync (2026-10-30)](sync-2026-10-30.md)

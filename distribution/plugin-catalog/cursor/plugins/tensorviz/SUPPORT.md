# Support and compatibility

This catalog targets macOS ARM64/x64, Linux x64 (glibc >=2.31) and ARM64
(glibc >=2.35), plus Windows x64/ARM64 through native Codex entries. It requires
matching CPython 3.12 for its compiled backend, but does not require VS Code or
Node.js. Startup/service compatibility is independent of PyTorch wheel or model
compatibility. Explicitly selected incompatible Python versions produce an error;
TensorViz does not silently replace that environment.
Windows Claude Code/Cursor, musl, CUDA and other Python ABIs remain outside this
catalog's acceptance scope. Runtime selection follows the MCP execution host,
not the client desktop. Native CI does not establish every desktop GUI, older
OS release, hardware configuration, or SSH-network behavior. Existing release
archives remain unchanged; only a published catalog with the matching native
asset supplies support.

The integration works with clients that support its local MCP transport and
plugin format. Directory publication is distinct from local installation, capture
and rendering. No official marketplace acceptance is claimed for this candidate.

When reporting a problem, include the integration version, OS/client version,
whether it failed during download, startup or capture, and a small synthetic
model. Do not share credentials, private viewer links, model secrets or unredacted
environment dumps. Use
[the integration repository's issues](https://github.com/tensorviz-org/tensorviz-plugins/issues)
for public, non-sensitive bug reports.


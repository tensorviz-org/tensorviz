# Linux compiled plugin runtime

## Published 0.1.45 scope

The published catalog supplies macOS ARM64/x64, Linux x64 (glibc >=2.31),
Linux ARM64 (glibc >=2.35), and Windows x64/ARM64 native runtimes. Windows
acceptance is through native Codex entries, not Windows Claude Code/Cursor.
Musl is unsupported by this published catalog. Model/backend execution requires
matching CPython 3.12; PyTorch wheel and model compatibility are separate from
TensorViz startup and service compatibility.

Runtime selection follows the MCP execution host, including a Linux SSH server,
not the client desktop. Raw localhost graph links refer to that execution host
and require forwarding for a remote desktop browser. Native CI does not establish
real SSH-network or every signed-in desktop client's behavior.

## Catalog and integrity

Format-2 catalogs hold a pinned entry per platform with the same version,
exact ABI, immutable HTTPS URL, size and SHA-256. The published
[0.1.45 runtime lock](https://github.com/tensorviz-org/tensorviz-plugins/blob/v0.1.45/plugins/tensorviz/runtime-lock.json)
lists the six assets; the
[support statement](https://github.com/tensorviz-org/tensorviz-plugins/blob/v0.1.45/plugins/tensorviz/SUPPORT.md)
defines acceptance limits. Unsupported targets must not substitute another
platform's archive or bypass integrity checks.

Release workflows live in this repository. A source checkout, locally assembled
candidate, or fixture test does not by itself publish or validate a native
archive. Source instructions remain in [Plugin build](plugin-build.md).

## glibc compatibility gates

The Linux x64 0.1.45 release is built and tested on glibc 2.31, including
ELF symbol auditing and compiled-launcher acceptance. Linux ARM64 retains a
glibc 2.35 floor. Native release and upgrade/rollback gates passed for 0.1.45;
that is not a blanket claim for every Linux distribution or hardware configuration.

Version 0.1.43 was built for glibc 2.35; editing a cached launcher's version
check cannot repair an incompatible binary. Install a published matching
artifact instead. The 0.1.44 Linux x64 release already lowered its floor to 2.31.
Existing historical release archives remain unchanged.

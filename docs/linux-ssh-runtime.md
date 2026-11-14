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

## Historical validation and release boundary

The following records the original port's evidence, not current release approval
or a substitute for the compatibility CI above.


Verified locally on Apple Silicon Mac with Node 22.23.4 and Codex CLI 0.165.0:
full plugin build/typecheck/panel and 98 units pass; units plus export audit pass
99/99, with no skips. Shared graph-host tests pass 1,178 cases. Both browser
surfaces build; four export/browser/notices checks and the HTTPS shell installer
fixture pass. Version 0.3.104/0.3.105 exports preserve 41 files per catalog and
138 local links/anchors each. No native client inference turn was run.

Initial dependency/panel setup races caused a missing type declaration and two
closed-host unit failures. After installation/build completed, full reruns passed
without source changes or bypasses. Logs remain outside this repository. Launcher tests inject uname, libc and a local downloader: they exercise
Linux/Darwin cold and cached startup, stdin, hash failures and unsupported targets.
Synthetic archive tests exercise multi-platform assembly and integrity; they do
not execute native Linux or Darwin binaries. Native rebuilds, old-glibc execution,
real SSH networking and installed-client acceptance were not run for this port.
Public dependency/toolchain versions differ from the original Linux patch and
need native acceptance before distribution.

The text above records historical port validation only. Current publication
and platform availability are described at the top of this page.

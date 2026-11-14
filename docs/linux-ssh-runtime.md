# Linux x64 compiled plugin runtime

## Scope

The compiled-plugin infrastructure now supports `darwin-arm64` and `linux-x64`.
Linux requires glibc >=2.31, `/usr/bin/python3` for metadata, `curl`, `sha256sum`
and `tar`; model/service execution requires CPython 3.12 with ABI
`cpython-312-x86_64-linux-gnu`. Darwin uses `cpython-312-darwin`. Windows, Linux
ARM64, Intel Mac and musl remain unsupported. No compiled source fallback exists.
The existing source-development packaging path is separate and unchanged.

The launcher selects the MCP execution host with `uname`, including an SSH server.
A Mac desktop displaying Codex does not select a Mac runtime for a Linux server.
POSIX background startup now preserves MCP stdin through fd 3 and closes the
duplicate, retaining signal/wait/session cleanup. Raw localhost graph links refer
to that execution host and require forwarding for a remote desktop browser.

Both native launchers place PyTorch's Inductor cache inside the private extraction
session, overriding an inherited `TORCHINDUCTOR_CACHE_DIR`. Model-derived cache
files are removed with that session after normal exit, failure or handled
termination; a caller's existing cache is neither modified nor deleted. `TMPDIR`
is unchanged so separate hosts continue to use the same cross-host lock location.

## Catalog and local assembly

Format-1 locks remain compatible. Format-2 catalogs hold one pinned entry per
platform with the same version, exact ABI, immutable HTTPS URL, size and SHA-256.
A Linux host refuses a Darwin-only lock before download. The current published
plugin supports both platforms; see the [public catalog](https://github.com/tensorviz-org/tensorviz-plugins)
for current release metadata. Older v0.3.104 releases remain Darwin-only.
This source checkout and its local validation do not publish a Linux archive.

Native builders select `node22-macos-arm64` or `node22-linux-x64`. Linux strips
compiler debug paths with failure propagation. Packaging audits the complete
readelf version table under an 8 MiB limit and rejects GLIBC symbols above 2.31.
Source/path exclusion, internal hashes, provenance, renderer coverage, exact ZIP
inventories and the 41-file compiled integration allowlist remain enforced.

Given separately built and audited archives with sibling `.json` locks:

```sh
node plugin/scripts/assemble-native-release.mjs /tmp/tensorviz-integration \
  /absolute/path/tensorviz-runtime-VERSION-darwin-arm64.tar.gz \
  /absolute/path/tensorviz-runtime-VERSION-linux-x64.tar.gz
```

This creates local candidate files only. Templates remain in
`plugin/packaging/public/`; the complete 29-tool inventory is 25 agent tools plus
four Model graph app transports. Current versions substitute `{{VERSION}}`; the
dated v0.3.104 public asset audit remains historical. Source instructions are in
[Plugin build](plugin-build.md).

## glibc 2.31 compatibility gate

The next release is built and tested inside an Ubuntu 20.04 container, rather
than merely lowering the launcher check on an Ubuntu 22.04 binary. The build
asserts glibc 2.31, audits the complete ELF symbol tables against that floor,
and exercises the real compiled launcher and CPU capture in that container.
Version 0.1.43 was built for glibc 2.35: changing its cached launcher is not a
supported repair. A new, verified release artifact is required.

Native upgrade/rollback tests involving old 2.35-only artifacts run separately
on a compatible host. Local fixture passes alone do not establish native ABI
compatibility or signed-in desktop acceptance.

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

Private build/publisher orchestration and native release workflows are intentionally
omitted. This source tree retains its existing packaging policy and has no new
runtime publication automation. Its existing `docs.yml` is unchanged: PRs, main
pushes and manual dispatch trigger Docs CI, with no pull_request_target trigger.
The coordinator must include `[skip ci]` in every feature commit to meet the
zero-nonrelease-CI requirement; no workflow is dispatched by this task.
NO MERGE/RELEASE APPROVAL.

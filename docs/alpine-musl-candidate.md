# Alpine x64 native runtime candidate

This is **candidate coverage, not a published support promise**. No production release,
version, or public runtime catalog is changed. It stacks on the multi-platform work.

## What changes

- Distinct `linux-x64-musl` catalog key, exact `cpython-312-x86_64-linux-musl`
  extension ABI, and native Node 22.23.4 Alpine host. Never select GNU artifacts for musl.
- Official SHA256-pinned dynamically linked CPython 3.12.14 musl build. Static Python
  is not interchangeable because the backend loads extension modules.
- Explicit ELF audit: x86_64, musl interpreter, no `GLIBC_*` symbols, and a narrow
  shared-library allowlist. GNU builds retain their existing symbol floors.
- Full plugin candidate CI on Alpine 3.24.4: licensed renderer, bytecode-only host,
  compiled backend, existing source-leak/package audits, native MCP tests and an
  authenticated static service test with Torch absent. Artifacts are private CI
  candidates only, not automatically published.
- Portable system tar location for packaging. The default public shell launcher
  remains GNU-only; musl candidate integrations must use the portable Node launcher.
- Prevent existing source bytecode caches from leaking into compiled output.

## Actual local evidence (2026-11-14)

The following ran inside an Alpine 3.24.4 x86_64 root filesystem, not simulated OS flags:

1. Official rootfs SHA256
   `cdaa38cb8de3537967ee5b6c9b6cd00cc962da2b018e6eea48696bb274923012`.
2. Official musl Python archive SHA256
   `1f37044c8cdbd74d5ee112a753c65ef209fedd169c98f3e4e748a93e27eb27a4`.
3. All standalone service dependencies installed from wheels, no Torch installed.
4. Full backend compiled with pinned Nuitka 2.8.10. Both source and compiled service
   health/authentication/discovery/projection/setup tests passed without importing
   Torch, TensorFlow or JAX.
5. Native Node22 Alpine executable built from main `f285a383` host source initialized
   MCP, listed 29 tools and handled `list_models`. A second native build using the
   distinct musl manifest selector passed the same test.
6. Empty-cache bootstrap used verified local HTTPS, exact size and SHA256 checks,
   fresh extraction and the native MCP test. Both actual native binaries passed
   the strict musl ELF dependency audit.

Limitations: the local host prototype used a minimal renderer fixture, **not UI
acceptance**. Full production renderer/package CI in this PR remains required.
The cold host archive test did not exercise cold managed-Python provisioning.
No actual model capture, ChatGPT/Claude desktop installation, Alpine ARM64,
older Alpine version or glibc below 2.31 is claimed. Capability advertisement is
not evidence that the user's framework is installed or that model capture works.

## Release gates

Before advertising musl: require candidate CI success on the exact revision, a
cold managed-runtime provisioning test, real browser/client acceptance, and
clear missing/incompatible-framework diagnostics. Add an exact musl entry to a
public catalog only for an audited published artifact. Never repoint the GNU key.

The native host and backend inspected here depend only on musl libc; build tools
are not customer prerequisites. Python wheels can have their own runtime-library
requirements. Keep wheel-only installation and report provisioning failures
separately from user-framework failures. Do not promise all musl versions from a
single Alpine test.

## Older glibc

Service wheel resolution for manylinux_2_28 succeeded, but that is not execution
proof. Node22 x64's upstream binary floor permits investigating glibc2.28; the
compiled backend, linked dependencies, source-only tools and actual client must
all be built/audited/tested there before lowering TensorViz's floor. Do not use an
end-of-life Node release to gain older glibc compatibility.

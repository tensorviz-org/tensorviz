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

## Release gates

Before advertising musl: require candidate CI success on the exact revision, the
cold managed-runtime regression, real browser/client acceptance, and
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

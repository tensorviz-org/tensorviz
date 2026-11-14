# Cross-platform candidate work

This work does not change the supported targets of an already published plugin.
The catalog must contain a verified native runtime for the exact operating system
and architecture before a launcher can use it. Hardware class (server, laptop or
desktop) is not a restriction; operating-system ABI is. Capture dependencies
are a separate capability and must not prevent the TensorViz host from starting.

The build schema now models x64 and ARM64 on Linux, macOS and Windows, including
Windows `.exe` and `.pyd` filenames. Candidate CI exercises the optional bootstrap
on native runners for all six targets and attempts native Windows x64/ARM64,
Intel Mac and Linux ARM64 builds without PyTorch installed. These
are build candidates, not a claim that all six complete installations work.

`plugin/integration/launch.cjs` is an optional, dependency-free Node.js 22+
bootstrap. It is **not** the public catalog default: existing Unix installations
do not gain a Node prerequisite. It verifies exact platform, ABI, version, size
and SHA-256, requires HTTPS, rejects unsafe archive paths and links, and preserves
MCP stdio. A native Windows catalog/bootstrap decision remains necessary to avoid
requiring users to install Node solely for bootstrapping.

## Dependency evidence and remaining gates

- Published Python 3.12.14 / 20260901 archive hashes add Linux ARM64 and Windows
  ARM64 managed-runtime candidates. Pins come from the upstream release digests,
  not from downloading arbitrary replacement hashes at runtime.
- The PyTorch CPU index contains 2.14.1 CPython 3.12 wheels for Linux x64/ARM64 and
  Windows x64/ARM64. The PyPI index alone omits the Windows ARM64 CPU wheel.
- Current PyTorch macOS wheels require Apple Silicon. Intel Mac requires a
  separate capture compatibility decision, not a TensorViz runtime block. Its
  older 2.2.2 wheel is not silently selected or advertised as equivalent to the
  current tested capture stack.
- The pkg Node Linux ARM64 prebuilt has a higher libc baseline than Linux x64.
  A published ARM runtime needs its own symbol audit and verified minimum libc.
- Runtime acceptance requires native compile, authenticated service startup,
  cancellation/cleanup, client installation and upgrade tests. CPU capture is
  tested separately in a compatible user-selected PyTorch environment. Merely passing the
  bootstrap's executable fixture is not complete TensorViz acceptance.
- Real signed-in desktop clients, graph UI and the broader manual checklist are
  not established by this candidate workflow.

Sources: [Python release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901),
[PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/torch/),
[pkg binary compatibility](https://github.com/yao-pkg/pkg-fetch#binary-compatibility).

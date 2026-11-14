# TensorViz 1.0.0 installation and migration guide (unpublished draft)

**Launch preparation only. Do not use these 1.0.0 download links yet.** Replace
this banner only after every linked artifact and documented client flow has been
verified. Current users should follow the current published release's guide.

## Choose the execution machine, then the client

Install where the model executes. An SSH server is the execution host even when
your screen is a Mac or Windows laptop. A server's localhost graph URL is not
your desktop's localhost; forward the port or use a supported embedded panel.

Native plugin targets are macOS Apple Silicon/Intel, Linux x64 (glibc >=2.31),
Linux ARM64 (glibc >=2.35), and Windows x64/ARM64. Compiled capture uses matching
CPython 3.12. Linux also needs `/usr/bin/python3`, `curl`, `sha256sum`, and `tar`.
Alpine/musl is not included. Windows plugin instructions below cover Codex only;
Windows Claude/Cursor and locked-down unsigned-executable environments need
separate acceptance. Do not disable OS security as an installation step.

TensorViz's own service and your model's Python environment are separate. Service
startup must work without PyTorch. Real model capture needs the model's framework
and dependencies in the selected environment. Preserve existing Torch/CUDA
versions; an unavailable framework wheel is not a TensorViz service failure.

## VS Code and VSCodium extension

1. In VS Code Extensions, find publisher **tensorviz**, extension ID
   **tensorviz.tensorviz-vscode**, and verify version **1.0.0** after launch.
   VSCodium users use the corresponding Open VSX listing, not a different product.
2. Offline/side-loaded route: download the exact VSIX listed in the canonical
   [1.0.0 release](https://github.com/tensorviz-org/tensorviz/releases/tag/v1.0.0),
   verify its SHA-256 against the release manifest, then use **Install from VSIX**.
   Do not infer a VSIX filename before the release inventory is finalized.
3. Open a trusted model project. Let TensorViz set up its own service runtime.
   Select the model interpreter via **tensorViz: Select Python Environment**.
4. Use **Open as Graph** on a small supported model first. Confirm the graph,
   tensor shapes and environment diagnostics before trying larger architectures.
5. Restart/reload the editor after upgrading. For rollback, keep the previous
   exact VSIX and use the editor's install-specific-version/VSIX route. Preserve
   your project and Python environments.

Gate before publishing this section: native clean installation and GUI capture,
missing/broken framework guidance, recovery and upgrade/rollback on each advertised
OS/architecture. Marketplace and Open VSX must show the same approved version.

## Coding-agent plugins: complete catalog ZIP

Download the complete
[1.0.0 catalog ZIP](https://github.com/tensorviz-org/tensorviz/releases/download/v1.0.0/tensorviz-catalog-1.0.0.zip)
and verify it against `release.json`. Extract into a persistent, new directory.
The ZIP retains the existing catalog-relative layout; it is **not** GitHub's
source ZIP and is **not** a per-client plugin ZIP. First connection still needs
internet to download the verified platform runtime. Keep the folder available.

### Codex on macOS or Linux

```sh
codex plugin marketplace add /absolute/path/to/tensorviz-catalog-1.0.0
codex plugin add tensorviz@tensorviz-standalone
```

### Codex on Windows

Use native Codex and the entry matching the machine architecture:

```powershell
codex plugin marketplace add C:\path\to\tensorviz-catalog-1.0.0
codex plugin add tensorviz-windows-x64@tensorviz-standalone
```

On ARM64 substitute `tensorviz-windows-arm64`. Do not install the Unix `tensorviz`
entry on Windows. If Windows blocks the executable, retain the exact error and
use the signed/approved build path once available; do not bypass protection.

### Claude Code on macOS or Linux

```sh
claude plugin marketplace add /absolute/path/to/tensorviz-catalog-1.0.0
claude plugin install tensorviz@tensorviz-standalone --scope user
```

### Cursor on macOS or Linux

Copy the complete extracted `cursor/plugins/tensorviz` directory into
`~/.cursor/plugins/local/tensorviz`, then reload Cursor. Confirm
`.cursor-plugin/plugin.json` exists there. Do not copy the repository or the
common `plugins/tensorviz` directory instead. An existing marketplace install can
take precedence over the local copy; keep only one intended installation enabled.

### ChatGPT app and T3

Use only the app/profile's supported installed-plugin or MCP connection route.
There is no claimed universal ChatGPT CLI installation command. The exact current
app import flow and end-to-end graphical acceptance must be recorded before this
section is finalized. T3 uses its selected provider's CLI and home; installation
in a different Codex profile does not install into T3. Restart the agent session
and request the graph panel; capture success and display success are separate.

## Migrating from tensorviz-plugins

The marketplace/plugin names stay the same. Moving the repository does not
change an already-installed marketplace source. Before changing it:

1. Record the installed client version/profile, current catalog location,
   plugin version and runtime digest. Back up configuration without sharing keys.
2. Keep the previous complete catalog and native release assets for rollback.
3. Close TensorViz sessions. Replace the old marketplace source through the
   client's supported source-management UI/commands, or replace an existing local
   catalog folder with the complete new export. **Do not combine exports.**
4. Refresh and reinstall into the same execution host/profile:

```sh
codex plugin marketplace upgrade tensorviz-standalone
codex plugin add tensorviz@tensorviz-standalone --json
```

Use the Windows-specific plugin name on Windows. Check the actual returned
`version` and `installedPath`; `plugin list` alone is not replacement evidence.

For Claude Code, after updating the underlying marketplace source:

```sh
claude plugin update tensorviz@tensorviz-standalone
```

For Cursor, move the old complete plugin directory outside the discovery tree,
install the new complete directory, reload and start a new chat. Never merge
files. Disable duplicate/manual TensorViz MCP entries so only one host runs.

Keep old source removal commands out of this guide until validated against the
pinned clients: CLI syntax and cached-source behavior vary. The direct canonical
repository install route also remains withheld until the new nested catalog
paths pass real client testing. The complete ZIP route avoids assuming that a
source checkout is equivalent to a generated distribution catalog.

Test a real capture, graph refresh/export and reconnection after upgrading.
Rollback restores the complete old catalog and reinstalls its actual version;
do not edit checksums or repoint an old lock to new bytes. Old release tags/assets
remain available at tensorviz-plugins. Archival stops new catalog updates; it is
not a promise that existing clients automatically follow to the new repository.

## Troubleshooting and removal

- **Service does not start:** collect the TensorViz startup error, host OS/arch,
  runtime version and verified archive digest. Distinguish native/OS policy
  failure from capture dependencies.
- **Missing/broken Torch:** select or repair the model environment with approval.
  Do not reinstall the TensorViz runtime or overwrite the project's frameworks.
- **No graph panel:** restart the originating agent session, check the profile
  and enabled plugin, then request a fresh graph/browser view.
- **Download/checksum failure:** verify access to the exact pinned GitHub release.
  Remove only the corrupt archive named by diagnostics; never disable validation.
- **Unsupported host:** check the advertised target and Python ABI. Desktop OS
  compatibility does not establish remote-host compatibility.

Close active sessions before removal:

```sh
codex plugin remove tensorviz@tensorviz-standalone
claude plugin uninstall tensorviz@tensorviz-standalone --scope user
```

Use the architecture-specific Windows name where applicable. Remove only the
TensorViz local plugin directory for Cursor. Removing the plugin does not remove
models, project recipes, Python environments or saved permissions/runtime caches.

## Publication acceptance record required

For each client/platform/route above, attach exact source SHA, artifact SHA-256,
client version, date, installation commands, logs/screenshots and pass/fail/blocked
for fresh install, no-Torch service, real capture, graphical display, upgrade,
restart, settings preservation and rollback. Existing 0.1.x results are baseline
evidence, not proof that these 1.0.0 routes pass.

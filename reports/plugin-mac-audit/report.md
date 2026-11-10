# Mac plugin audit — 2026-11-11

The published 0.3.109 plugin installs in isolated Codex CLI and Claude Code profiles,
captures a deterministic CPU model, exposes measured context and renders the browser
fallback. Repeated requests for an embedded graph in a client with no mounted app
consume all eight panel slots. The candidate fixes this without rejecting hosts
that omit UI capability advertisement. It also supports an explicitly isolated
runtime cache, precise damaged-cache recovery and verified-only prefetch.

This report is in the private source repository. The public integration allowlist
is unchanged. No release, merge, deployment or catalog promotion was performed.

## Versions and ownership

| Component | Exact baseline |
| --- | --- |
| Mac | macOS 26.4, build 25E246, arm64 |
| Installed desktop | ChatGPT 26.1103.60933 (25894), bundle `com.openai.codex` |
| Desktop bundled CLI | 0.167.0-alpha.3.2; not used as evidence of desktop rendering |
| Tested Codex CLI | 0.165.0 |
| Tested Claude Code | 2.1.308 |
| Published plugin/runtime | v0.3.109 prerelease, published 2026-11-09T19:26:01Z |
| Catalog | `5ae962b40ea8d0b875d0542c739c4c35c076575e` |
| Published runtime source | `8b74db34b74014aa62ebd726b12c14b095b948dd`, clean provenance |
| Source checkout base | `f8caa1dddd27dcee45e43a9b367f7ab01aac591b` |
| Capture interpreter | CPython 3.12.13, PyTorch 2.7.1, CPU |
| Provisioned service interpreter | pinned CPython 3.12.14 |
| Native build | dedicated CPython 3.12.13, Nuitka 2.8.10 |
| Build Node | 26.5.0 locally; CI pins Node 22 |
| Initial candidate | 0.3.61 from the archived source package, tested locally before the repository move; **not published** |
| Final PR candidate | 0.3.110 from active `tensorviz` source; separate fresh build/tests recorded below; **not published** |

The active private `tensorviz` repository owns the runtime, backend, renderer, integration templates, exporter and release workflow. The original runtime was published from `tensorViz-extension`, which is now archived/read-only. An attempted push there was rejected; no archived repository setting was changed. The final fixes are ported onto active source base `d87d8be36ab346dcd5a8ba685eb26f55d4cb46b0`. The public
`tensorviz-plugins` repository contains the generated catalog/integration only.
The published runtime-owned source scopes match the checkout base; unrelated
extension commits do not establish a newer plugin release.

All four common/Mac release assets were downloaded over verified HTTPS and checked
against GitHub asset SHA-256 digests and runtime pins. The common ZIP's 15 files
and Cursor ZIP's 16 files match the catalog byte-for-byte. Exact URLs, byte sizes
and hashes are in [release.json](evidence/release.json). No Linux binary was
needed or downloaded. Linux launcher branches have simulated unit coverage only.

## Initial published/legacy-candidate matrix

| Test | Published 0.3.109 | Local candidate | Evidence scope |
| --- | --- | --- | --- |
| Published marketplace commands, installed/enabled | Codex and Claude pass | Local exported integration installs | Actual isolated CLI profiles |
| Runtime start/tool discovery | 29 tools in both CLIs after download | 29 tools in both CLIs | Native CLI control bridges, no provider turn |
| Fresh runtime download | Cold Claude connection initially failed; process-only `MCP_TIMEOUT=180000` retry connected | Both CLIs download through a local HTTPS archive fixture into separate caches | Real client processes; fixture CA passed only to child curl |
| Capture/context | `[2,4] → [2,3]`, 15 parameters; edit to `[2,2]`, 10 parameters | `[2,4] → [2,3]` through both CLIs; native edit matrix also passes | Actual compiled binaries and CPU execution |
| Browser fallback/render | Pass; visible graph, inspection, selection receipt, pan, zoom, fit | Production browser code used in embedded harness; fallback unchanged | Actual browser interaction; not signed-in desktop GUI |
| Edit/refresh | ReLU/width 3 → Sigmoid/width 2; old snapshot becomes stale; visible measured `[2,2]` | Native and bridge receipt tests pass | Before/after screenshots and native tests |
| Repeated unmounted graph openers | 8 allocations then errors; 16 requests produce 8 errors | 16 requests, no errors; same pending panel ID in both real CLI bridges | Regression reproduced in real CLIs and native MCP |
| Embedded graph render | Signed-in desktop blocked | Render, zoom and refresh pass in MCP Apps iframe harness | Explicit synthetic host, real renderer; no native desktop acceptance |
| New host/session and reconnect | Pass | Pass | Native MCP process restart, persisted task-only consent |
| Update | Codex marketplace refresh succeeds; Claude reports latest 0.3.109 | No publication/update transition attempted | Current-to-current update only |
| Uninstall/reinstall | Published task profiles removed/reinstalled separately | Both real-client package tests install then uninstall and preserve fixture source/recipe | Isolated profiles only |

The initial local candidate in the matrix above was compiled from source version 0.3.61. Its numerically
older label is a build identifier, not a claim that the published plugin was
replaced. Packaging audits enforce binary-only runtime payloads and the public
allowlist. Candidate distribution coverage stays pending; no production license
key or publication credential is used by the new validation workflow.

## Reproduced defects and fixes

1. **Invisible panels exhaust the connection.** `GraphPanels.open()` allocates a
   new slot even when no app ever exchanges with it. The ninth call fails with
   `PANEL_LIMIT`. An unopened panel is now reused until first exchange. Mounted
   frames retain independent identities, the eight-active-panel limit, owner
   isolation, stale-frame rejection and cleanup. Legacy hosts without `clientId`
   remain supported. The regression failed on the original implementation.
2. **Mac launcher cannot isolate its archive cache independently of HOME.**
   `TENSORVIZ_RUNTIME_CACHE_DIR` selects an explicit absolute cache. Default paths
   are preserved; relative paths and symlinks remain rejected. Native CLI test
   fixtures now use this variable instead of redirecting HOME. HTTPS, bounded
   size, SHA-256, platform/ABI checks and per-session extraction stay enforced.
3. **Truncated-cache error omits the recovery filename.** Size mismatch now
   identifies the exact archive to remove, matching checksum-failure recovery.
4. **Slow cold download can exceed client startup deadlines.** The launcher now
   offers `--prepare-runtime`: download and verify only, without extracting,
   executing the host, starting Python or changing client configuration. Tests
   prove no execution and refusal of damaged bytes. This is an explicit recovery
   path, not a promise that every cold connection starts within a host deadline.

Do not add `startup_timeout_sec` to portable `mcp.json`. A task-only compatibility
experiment confirmed Codex 0.165.0 rejects it as an unknown field and skips the
server. It was removed from the test profile and is absent from the exporter.
Claude's process-only timeout is documented for the existing published version;
0.3.109 does not have the new prefetch flag.

## Error/recovery checks

[Published native results](evidence/published-native.json) and
[candidate native results](evidence/candidate-native.json) use the same fixture:

- Known CPU input is all ones, weights 0.25 and bias 0.5. Initial Linear output
  is 1.5 with shape `[2,3]`; context reports exactly 15 parameters.
- Missing Pydantic in the disposable capture environment produced actionable
  dependency setup. The published install-dependencies continuation installed it
  and resumed to ready without replacing Torch. No normal Python env was used.
- Shape mismatch `[2,5]` and a deliberate model exception both return failed,
  `validated:false`, and an inspect-failure continuation. Correcting the fixture
  restores a successful measured capture.
- Required constructor/string input returns `needs_setup` and `complete_setup`;
  incompatible system Python returns `needs_setup` and `select_environment`.
- Repeated capture is rejected with `OPENING_BUSY`; cancellation returns
  `cancelled`; new process reconnect captures successfully. A prior process's
  request ID returns `UNKNOWN_REQUEST`. A stale browser handoff/reference is
  rejected with `UNKNOWN_HANDOFF`; a new handoff can be requested.
- Source suites separately verify cancellation kills task-owned workers and their
  children, launcher crash cleanup, edits during execution, dependency-only edits,
  retained runs and refusal of stale approval.

Initial audit-harness failures were corrected without weakening product checks:
source-host tests need the generated host version to match the copied manifest;
native backend tests require a canonical path and a full service/test environment.
A numeric constructor parameter was inferable by design, so the missing-input
fixture now includes an explicit required string. An intentionally rejected
portable timeout field was not a published-artifact defect.

## Reproduction

Published install commands, each with a separate task-owned profile:

```sh
CODEX_HOME="$AUDIT/codex-profile" codex plugin marketplace add tensorviz-org/tensorviz-plugins
CODEX_HOME="$AUDIT/codex-profile" codex plugin add tensorviz@tensorviz-standalone
CLAUDE_CONFIG_DIR="$AUDIT/claude-profile" claude plugin marketplace add tensorviz-org/tensorviz-plugins
CLAUDE_CONFIG_DIR="$AUDIT/claude-profile" claude plugin install tensorviz@tensorviz-standalone --scope user
```

Never point these variables at a normal profile. The real CLI package regressions
are [codex.cjs](../../plugin/test/clients/codex.cjs),
[claude.cjs](../../plugin/test/clients/claude.cjs) and
[native/clients.cjs](../../plugin/test/native/clients.cjs). They use native
control protocols, scoped synthetic elicitation and no provider/model turn.

With an extracted verified runtime, a task-owned compatible Python and an empty
absolute `$AUDIT` directory:

```sh
TENSORVIZ_TEST_RUNTIME="$RUNTIME" \
TENSORVIZ_TEST_PYTHON="$PYTHON" \
TENSORVIZ_AUDIT_ROOT="$AUDIT" \
TENSORVIZ_AUDIT_INCOMPATIBLE_PYTHON=/usr/bin/python3 \
node plugin/test/manual/mac-audit.cjs
```

The script generates deterministic models/recipes and a sanitized `report.json`
inside a newly allocated directory. It asserts actual shapes, parameters,
failures, cancellation and reconnect. Optional `TENSORVIZ_TEST_SERVICE_PYTHON`
reuses a prepared service through the existing isolated fixture helper. Without
it, normal verified service provisioning applies. Inspect report paths locally;
do not upload runtime journals, permission receipts or handoff files.

For an actual rendered iframe test, run
[manual/panel.cjs](../../plugin/test/manual/panel.cjs) with the same runtime and
Python variables and an absolute, task-owned `TENSORVIZ_TEST_HANDOFF_FILE`. Read
that mode-0600 file locally to open the capability URL. Send `{"action":"capture"}`,
then `change` and `refresh` on stdin. The header explicitly identifies the
synthetic bridge and omits project paths. This is not a signed-in Codex test.

For a new exported candidate, prefetch with:

```sh
TENSORVIZ_RUNTIME_CACHE_DIR="$AUDIT/runtime-cache" \
sh "$PLUGIN/scripts/launch.sh" --prepare-runtime
```

Use that same cache in the MCP server environment. Do not pass this new flag to
0.3.109, and do not disable TLS or checksum checks.

## Verification and limits

The counts below describe the initial archived-source audit. They are retained as historical evidence. Final active-source verification is recorded in the next section and PR checks.


- TypeScript check and syntax/diff checks pass. Feature record render/check and four generator tests pass. The repository-wide link check is recorded separately; one existing link to an absent generated presentation remains in docs/research/tensorviz-launch-strategy.md; it is unrelated to this change.
- 101 plugin unit tests pass, including the new panel/cache/prefetch regressions.
- 1,230 shared tests pass.
- 10 selected source integration tests pass: discovery, CPU capture, freshness,
  cancellation/crash/source/dependency lifecycle and retained execution evidence.
- Both real CLI candidate installation/download/capture/panel/uninstall tests pass.
- Six native backend/host/installer/export tests pass with the documented complete
  test environment; the first invocation omitted that environment and failed.
- Native open-graph integration passes exact selection/focus/refresh receipts.
- Public export remains exactly 41 allowlisted files. No arbitrary report,
  fixture, source, backend or compiled runtime enters the public integration.
- The final `Release safety checks` workflow has a manually dispatched Mac candidate job checking the **head SHA** using a read-only token and no repository secrets, with no publish/deploy jobs.
  CI results belong to the PR checks; this document does not predeclare success.

Signed-in desktop acceptance remains blocked: Computer Use refused the installed
`com.openai.codex` app both by bundle and by the requested “Codex” name. No second
instance/profile was guessed, no normal GUI plugin settings were changed, and no
login was performed. Browser/CLI evidence must not be counted as desktop GUI
acceptance. ChatGPT web, Claude Desktop, Cowork, Cursor and T3 were not tested by
this audit. Those surfaces are not interchangeable. No paid model/provider turn
was requested; an unauthenticated Codex startup attempted its normal warmup and
received 401, which is not agent-model acceptance.

The custom-profile auth, Claude settings and normal marketplace fingerprints
remain unchanged. Claude's normal legacy state file refreshed caches and only
`oauthAccount.profileFetchedAt`; account identifiers, project data and global MCP
settings match its exact pre-test backup. The normal Codex config hash changed
around app-managed worktree registration (desktop worktree preferences). Its full
original bytes were unavailable, so it was not blindly overwritten. Byte-identical
normal-state preservation cannot be asserted. No API key, provider, routing,
endpoint, Keychain, account identifier or unrelated MCP configuration was edited.

The failed app-managed worktree registration was followed by a separate source
worktree under the task directory; existing work was preserved. Task-owned servers
and the temporary `caffeinate -di` process will be stopped at handoff. No monitoring or
automation was configured. Disposable profiles, candidates and caches remain in
the local audit workspace for reproduction; raw local data is not committed.

Next review: inspect the focused fixes and exact-head CI, then perform supported,
verified-isolated signed-in GUI acceptance before any release decision. Review
concurrent iframe mounts that request the same opener before their first exchange;
mounted-frame lifecycle tests remain green, but actual supported desktop hosts
are needed for final compatibility acceptance.

## Active source port and final validation

The final branch is based on `tensorviz` main at
`d87d8be36ab346dcd5a8ba685eb26f55d4cb46b0`, plugin 0.3.110. Existing newer
Inductor-cache isolation, graph export, account/share behavior and dependency
updates are preserved. The public catalog remains pinned to 0.3.109 until a
separately authorized release. The scoped fix does not publish or sign in.

The active source plugin unit suite passes all 154 tests and all 1,189 shared tests. The 0.3.110 native error/recovery matrix passes (see evidence/active-native.json), the 281-file compiled package audit passes, and all 81 documentation-source links resolve. The documentation build verifies 54 pages and 3,494 local links/anchors. Both actual CLI install/download/capture/uninstall flows pass against the 0.3.110 native candidate (Codex 0.165.0, Claude Code 2.1.308): each discovers 29 tools and reuses one pending panel for sixteen requests. Both clients also pass source edit/recapture with `[2,2]` and 10 parameters, changed revision, actionable shape/model failures, correction/recovery and cancellation through their own control bridges. The shared synthetic fixture is `plugin/test/fixtures/client-recovery.cjs`. The expanded local flow takes about 132 seconds across both clients; time budgets grow with the added captures while all outcome assertions remain strict. All seven release/package/native checks and the native open-graph selection/focus/refresh integration pass. Release provenance/packaging/workflow units pass 97 tests, plus the bundle-notices check. Exact-head CI results are recorded in the PR handoff. Do not treat the
legacy 0.3.61 evidence files or screenshots as evidence for an installed 0.3.110
release.

The final 0.3.110 MCP Apps iframe was visibly rendered and zoomed, refreshed from ReLU to Sigmoid, selected through Find a layer, and inspected in Captured details (completed invocation, 15 parameters). The native `get_selection` returned the same selected node. See [refreshed graph](evidence/active-embedded-refresh.png) and [captured details](evidence/active-embedded-details.png). The separate native matrix changes width as well and verifies `[2,3] → [2,2]`; the iframe fixture keeps width 3. Both are synthetic host checks, not signed-in desktop acceptance.

The audit lives under `reports/`, outside the public documentation-site source
and outside the integration exporter allowlist. The Markdown report and sanitized evidence are ordinary source audit artifacts; archived feature-generator scaffolding is omitted because the active repository does not own that generator.

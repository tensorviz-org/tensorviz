# Mac plugin acceptance — November 14, 2026

The actual published **0.1.44 and 0.1.45** plugins passed isolated installation,
removal, verified runtime prefetch, native stdio discovery, and installed
Codex/Claude static MCP bridge checks on this Apple Silicon Mac. Capture stopped
at the ungranted TensorViz permission prompt in both releases. These checks do not
establish model capture, graphical rendering, or other-platform acceptance.

Environment: macOS 26.4 (25E246), arm64, Node 26.5.0, npm 11.17.0,
Codex CLI 0.165.0, Claude Code 2.1.308. Compiled dependency probes used genuine
arm64 CPython 3.12.13 environments, with Torch absent or Torch 2.14.1 installed.

| Check | Published 0.1.44 | Published 0.1.45 |
|---|---|---|
| Provenance | Source `f285a383f628af23ac4992970eeab3b006b1f0d2`; all 41 catalog and 281 native-manifest files verified | Source `eae4dd5d012c090d31ca1149891287ea01febc6c`; all 57 catalog and 281 native-manifest files verified; common plugin ZIP matches the catalog |
| Clean install/remove | Both clients installed/enabled exact version; inventories empty after removal | Same checks passed for exact 0.1.45 |
| Actual installed MCP bridges | Codex app-server and Claude stream-json: installed skill, 29 tools, host 0.1.44, empty models | Same checks passed for host 0.1.45; no provider/model turn |
| Native stdio with no project Python supplied | Initialization, 29 tools, empty models, and schema-valid `NO_MODEL` responses passed | Same checks passed |
| Cold documented prefetch | Passed in 73.48 seconds; one verified cached archive, no TensorViz state | Passed in 100.38 seconds; one verified cached archive, no TensorViz state |
| Older-version update/remove | 0.1.43 → 0.1.44 passed in both clients | 0.1.44 → 0.1.45 passed in both clients |
| Local-ZIP update instructions | Codex Git-only `marketplace upgrade` command fails for local folders; complete catalog replacement then `plugin add` passed in a separate retest | Same documentation error reproduced; corrected sequence independently passed with the actual new catalog |
| Direct compiled missing-Torch probe | Exits with `ModuleNotFoundError` traceback, no structured diagnostic | Returns structured `MISSING_DEPENDENCIES`, no stderr |
| Direct compiled incompatible-Torch probe | Synthetic incompatible module produces `AttributeError` traceback | Returns structured `CAPTURE_RUNTIME_UNSUPPORTED`, no stderr |
| Direct compiled healthy-Torch probe | Passed with Torch 2.14.1 | Passed with Torch 2.14.1 |
| Native capture permission | Blocked; harness sent no approval and created no permission record | Same boundary observed |
| Graphical rendering | Not run by this plugin harness | Not run by this plugin harness |

The prefetch durations are observations from separate downloads, not a controlled
performance comparison or cold client-startup acceptance. Direct compiled probes
test internal dependency diagnostics. Permission blocked verification of the
user-facing capture error behavior, so their results do not certify that UI.

| Published artifact | SHA-256 |
|---|---|
| 0.1.44 catalog ZIP | `c9309b0d02b6e6f5efa7199718e5ba991d3213b88d1741f19718d85291c96057` |
| 0.1.44 Darwin arm64 runtime | `78a201724c5ebe2eaaf494826f29799a2c5ac2b4ef3b86bf20d4d75748aef450` |
| 0.1.45 common plugin ZIP | `24fb75738cbdc31dc0e82306327f398cd0d6615c54b10edaad89d02a7aaa357a` |
| 0.1.45 catalog ZIP | `70ac8455c043f9bca3deb79dd4a809a6b766581de6d29ba655d675826279972a` |
| 0.1.45 Darwin arm64 runtime | `a8a3849819860bb25ab7d79b3cb0ff136caa6efe14dfb17725a9b98d133364ab` |

Publication chronology is preserved: `v0.1.45` was absent at 16:31:05 UTC, then
GitHub recorded its prerelease publication at **16:32:18 UTC**. The subsequent
published-0.1.45 pass above used its actual downloaded assets. Its manifest lists
six native targets (Darwin arm64/x64, Linux arm64/x64, Windows arm64/x64).
Only Darwin arm64 was executed here. Other target availability is metadata;
production workflow gate success was not independently verified.

Before that publication, a separate local **source-development 0.1.45** archive
also passed both static client bridges. It is not the published native package.
Source revision `7dc2004edfbda4218ea683f3219a52d05d02e89a` differs from base
`eae4dd5d012c090d31ca1149891287ea01febc6c` only in the two documentation files in
the fix below. Its archive SHA-256 is
`055f3e8a614d9fa7f566e49d25f68b9ef8e1fd96ddd77ecba5683b6cf234f2f3`;
all 434 package-manifest files were checked before task-only state/cache overrides.

The local-catalog documentation correction is
[draft PR #73](https://github.com/tensorviz-org/tensorviz/pull/73).
Validation includes both published upgrade pairs, documentation link/reference
checks across 83 files, and both required documentation builds (56 pages and
3,617 local links/anchors at each base path). No Actions workflow, merge, release,
or publication of built product archives was performed by this test work.

An initial 0.1.45 bridge run left two extracted runtime directories after the
harness used a fixed shutdown delay. They were removed within the test area.
A focused rerun waited for Codex's actual `thread/closed` event: both clients then
exited through stdin closure with zero runtime directories remaining. This
identified a harness teardown issue, not an established TensorViz lifecycle bug.

All client profiles, runtime caches, fixtures and generated state were task-owned.
No account credentials were copied and no permission was granted by the harness.
Raw protocols, stderr, profiles and archives remain private; published receipts
contain allowlisted versions, hashes, counts and outcomes. No graphical
screenshots or private local source paths are included here.

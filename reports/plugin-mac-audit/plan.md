# Published plugin Mac audit

## Problem and authorization
The owner requested fresh published-plugin installation and real-client tests on this Mac, focused fixes, regression coverage and a pushed draft PR. Preserve live sessions, credentials, routing, user configuration, unrelated MCP connections and existing work. No merge, publication or deployment is authorized.

## Baseline and owning layers
Published preview 0.3.109 (source revision 8b74db34b74014aa62ebd726b12c14b095b948dd), public catalog 5ae962b40ea8d0b875d0542c739c4c35c076575e. Both release ZIPs match the catalog byte-for-byte. Source branch starts at f8caa1dddd27dcee45e43a9b367f7ab01aac591b. Public integration is exporter-owned and allowlisted; this private source repo owns runtime, renderer and release workflow.

macOS 26.4 (25E246), arm64. Installed host is ChatGPT 26.1103.60933 (25894), bundle com.openai.codex. Computer Use denies access to that app. Do not launch an unverified second profile or claim signed-in GUI acceptance. Codex CLI 0.165.0 and Claude Code 2.1.308 each installed and enabled 0.3.109 in dedicated profiles using published commands.

Direct published-runtime protocol test reproduced nine panel openings with no UI capabilities: eight invisible allocations followed by a panel-limit failure. Missing capability advertisement alone is not sufficient to reject a host because supported hosts can omit it.

## Approach and alternatives
1. Keep release assets immutable, verify digests/provenance and run baseline captures before fixes.
2. Use deterministic CPU Tiny models, exact input shapes, task-owned Python and state. Classify raw MCP, native CLI control bridge, actual browser rendering and signed-in desktop GUI separately.
3. Investigate terminal-client identification and repeated unopened panels. Prefer a narrowly evidenced compatibility policy or bounded unopened-panel reuse over blanket rejection of missing UI capabilities. Preserve advertised MCP Apps and legacy supported hosts.
4. Reproduce other errors and recovery, implement focused fixes with failing regressions, then rerun affected flows.
5. Run source checks, relevant suites, packaging/export allowlist audit and exact-head authorized CI. Record failures and blockers honestly.
6. Sanitize fixture evidence, render/check this record, push a task branch, open a draft PR and verify its remote head.

## Acceptance
Release and candidate evidence stays separate. A capture is not display. A protocol bridge is not a signed-in GUI. No private URLs, credentials, user paths or unrelated data in committed evidence. Remaining GUI/auth blockers stay explicit. Task-owned processes are stopped or explained.

## Conclusion
Focused fixes and local runtime/CLI/browser checks pass. See report.md for the full matrix, initial harness failures, preservation drift and desktop GUI blocker. Exact-head CI and the draft PR remain part of the handoff.

## Verified fixes
Unopened panel requests now reuse one pending opener until the app begins its
first exchange. Mounted/legacy panels retain independent identities and the
eight-active-panel bound. No client is rejected merely for omitting capabilities.
The new regression failed against the baseline and passes with this fix.

The launcher gains an explicit absolute `TENSORVIZ_RUNTIME_CACHE_DIR` for isolated
profiles. Native-client harnesses use it instead of changing HOME. Truncated-cache
errors now name the exact archive and recovery action. Checksums, HTTPS, symlink
rejection, platform pins and default locations are preserved.

Implementation update: add explicit verified-only runtime prefetch for cold-download startup deadlines. Keep portable MCP schemas unchanged: live Codex 0.165.0 rejects `startup_timeout_sec` in mcp.json. Add a read-only, secret-free PR validation workflow; the existing main-only publication gate remains unchanged.

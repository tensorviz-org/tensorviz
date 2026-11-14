# Public 1.0.0 cutover (DRAFT: do not merge or execute yet)

1.0.0 is the public-launch milestone, not the next 0.1.x release. This preparation
must not change the current release, repository visibility, branch rules or old
catalog. Keep this PR draft and rebase/revalidate immediately before launch.

## Implemented offline preparation

`plugin/scripts/canonical-catalog.mjs` turns an **audited** compiled integration
export into a fresh staging directory. It copies the byte-identical export to
`distribution/plugin-catalog/` and emits the three root client marketplace
indexes pointing there. It preserves marketplace/plugin identities and runtime
URLs. It does not overwrite root README, LICENSE, source `plugins/tensorviz`, or
any source files. It never deletes anything, commits, tags, pushes or publishes.
The existing production publisher is intentionally unchanged.

```sh
node --test plugin/test/unit/canonical-catalog.test.mjs
node plugin/scripts/canonical-catalog.mjs /absolute/audited-integration /absolute/new-staging-directory
```

The output is a reviewed additive Git tree overlay, **not a release**. Review
existing root marketplace files before applying it. Keep binary runtimes in
release assets, not Git. The current Windows bootstrap executable is part of the
existing catalog contract; assess an asset-based bootstrap separately without
breaking installed clients. Repeated staging uses a new directory and refuses an
existing destination. `CANONICAL.json` records per-file hashes and original
source revision. The original PUBLICATION file is reserialized deterministically;
its manifest hash refers to that serialization, not an original JSON file byte
hash. Runtime archive hashes remain unchanged.

## Required implementation still pending before this PR is merge-ready

- Add a separately reviewed canonical asset publisher. **Never retarget
  `publish-native-release.mjs`: it deletes every destination entry except .git.**
- Canonical release tag must identify reviewed source, not the generated catalog
  commit. Build source S; release `v1.0.0` points to S; artifact manifest records S;
  catalog update C descends from S and references verified assets at that tag.
  This avoids a self-referential source/catalog checksum cycle. Never move tags.
- Extend export/candidate validation to support explicitly selected canonical
  repository; default remains the legacy repository until owner cutover.
  Do not rewrite a runtime URL without rebuilding and validating its manifest.
- One publication workflow orchestrates extension VSIX, Marketplace/OpenVSX,
  plugin assets/catalog and docs from exactly S and one version. Upload missing
  assets only to a draft, reject different bytes, publish only after verification,
  then advance catalog with an expected-head lease. No force push/deletion.
  External registries are non-atomic: report partial publication and retry exact
  bytes, never silently mark every channel complete.
- Connect local evidence receipts to publication gates. Preserve the historical
  release-specific migration contracts. Do not replace them with a boolean
  environment bypass. No test jobs on pushes/PRs; Actions used for necessary
  publication only. Local/native tests run before approval.
- Align package manifests, lockfile roots, client manifests, archive names,
  extension/plugin versions and changelogs to 1.0.0 only at cutover. Keep extension
  publisher/name, plugin names and historical extension versions unchanged.
- Test nested marketplace paths with real pinned Codex, Claude Code and Cursor
  clients before relying on the generated indexes. Pure transform tests are not
  client installation tests.

## Installation and upgrade documentation deliverables

Update the canonical README, `docs/`, plugin public packaging docs, website
installation/download links and Marketplace/OpenVSX listing together. Keep
currently working 0.1.x instructions until verified 1.0.0 links exist. Document:

| Client | Installation acceptance and instructions required |
| --- | --- |
| VS Code | Marketplace and offline VSIX install on Windows, macOS and Linux; verify extension identity/version; service setup, model interpreter selection, first graph, upgrades and rollback |
| VSCodium/Open VSX | Listing and offline VSIX route, exact release parity, first graph |
| Codex | Canonical marketplace add/install commands copied from tested pinned client; Unix and Windows architecture-specific entries; restart/cache cleanup and existing legacy marketplace migration |
| Claude Code | Tested marketplace add/install commands, stdio readiness, first real capture, restart, existing source replacement and rollback |
| Cursor | Verified native plugin/marketplace instructions and first real graph; never infer support from Claude metadata |
| ChatGPT app | Document the actual supported plugin/MCP import route and account prerequisites only after graphical acceptance; do not invent a CLI install command |

For every route distinguish TensorViz service/runtime startup from the user's
model environment. Starting the service must not require or silently install
PyTorch. Missing/broken Torch, unsupported Python ABI or CUDA errors should name
the model environment and recovery action, not blame TensorViz startup. Never
replace an existing model framework without approval.

Platform matrix: native darwin-arm64/darwin-x64, linux-x64 (glibc >=2.31),
linux-arm64 (glibc >=2.35), win32-x64/win32-arm64. Those are current advertised
runtime targets, not a claim every client is validated on every target. Alpine/
musl is not public-supported until packaging and native acceptance are complete.
Windows signing/Smart App Control compatibility is an explicit blocker for
restricted Windows systems. Do not tell users to disable protection as the
normal installation method.

Each install page must include prerequisites, a clean-install command/UI flow,
expected version/output, download provenance/checksum verification, local
loopback/security behavior, missing-framework diagnostics, proxy/offline limits,
uninstall, troubleshooting, upgrade and rollback. Maintain a tested-command
matrix with client version, OS/architecture, source SHA, artifact digest, date,
log/screenshot links and pass/fail/blocked (not just a checkbox).

### Existing users

A documentation URL change does not migrate installed marketplaces. Test removal
or replacement of the legacy source using each client's supported UI/commands,
retaining user settings and avoiding duplicate `tensorviz` servers. Test cached
installs, cold installs, native runtime reuse, restart, upgrade and rollback to
the last 0.1.x assets. Keep all legacy tags/assets immutable and anonymously
readable. A compatibility catalog may point new downloads to core, but archived
repositories cannot keep publishing updates. Do not promise perpetual automatic
updates; archive only after the migration path works and is documented.

## Release and public-source gates

- Full source/history/all-refs audit for credentials, private data, fixtures,
  reports/screenshots and artifacts. Do not import private archival history.
- License/SBOM review, notices and explicit tldraw commercial distribution scope;
  TensorViz's own license does not relicense dependencies.
- At least one eligible TensorViz org reviewer and org-only merge rights verified
  on the eventual public repo. CODEOWNERS must reference real eligible teams;
  configure actual branch/ruleset enforcement at cutover, not in this preparation.
- Full applicable checklist from
  https://github.com/tensorviz-org/tensorViz-extension/issues/179 after every
  release-candidate merge. Record exceptions; build success is not graphical QA.
- Fresh native service without Torch; real small capture; missing/broken framework
  guidance/recovery; graph expand/collapse/diff/refresh/export; client reconnect;
  existing model suites and large architectures; install/upgrade/rollback matrix.
- Anonymous downloads and checksums, release/source identity and marketplace
  parity verified after publication. Public visibility/archival require the
  owner's launch decision, not an automated side effect of merging this PR.

## Repository disposition inventory (no actions in this PR)

| Repository | Intended disposition / required review |
| --- | --- |
| tensorviz | Canonical source, plugin/catalog, extension, docs, issues and releases |
| tensorviz-plugins | Migrate catalog and client install paths; preserve old tags/assets; archive only after migration acceptance |
| tensorViz-extension | Already archived; retain history/issues including #179 |
| tensorviz-examples | Import useful examples with attribution/provenance, then archive |
| tensorViz-ide-old | Inspect unique code/history before migration/archive |
| tensorviz-fallback-lab | Inspect experiments before migration/archive |
| desktop / web | Inspect design/planning previews; not assumed shipping products |
| tensorviz-auto-research | Review scheduled collector/dashboard dependencies before consolidation |
| tensorviz-history-private | Retain private archive; never dump history into public repo |
| tensorViz-website | Retain, update install links/docs; explicitly excluded from archival |
| app / klars-website | Separate product; excluded unless owner identifies desired scope |
| marketing | Privacy and ownership review before any move/archive |
| the-incredible-pytorch / Awesome-pytorch-list / awesome-vscode | Resource repositories, not automatically TensorViz product archival scope |

## Rollout order

1. Finish and independently review this isolated PR; locally test all changed
   publication logic with simulated partial failures and retries.
2. Complete platform/client migration evidence and exact installed-user guide.
3. Owner approves source/privacy/license audit and exact archive inventory.
4. Rebase; prepare coherent 1.0.0 versions and immutable reviewed source S.
5. Merge immediately before owner-authorized public cutover; publish only the
   approved source/artifacts; verify every channel and anonymous installation.
6. Publish tested legacy migration notice/compatibility catalog. Archive old
   product repos only after all consumers and automation are accounted for.
7. Keep website and immutable old downloads available. Report remaining blockers.

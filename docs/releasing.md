# Releasing TensorViz

All new release automation belongs to `tensorviz-org/tensorviz`. The migration is
not active merely because these files exist. `RELEASES_ENABLED` must remain unset
or `false` until the owner completes and verifies the cutover below. The initial cutover is manual-only: merging this code cannot start publication,
even if a repository variable was already enabled. No legacy workflow, repository
visibility, permission, secret or environment is changed by this code change.

Publishing uses repository Actions secrets, matching the legacy workflows.
No deployment environment or Enterprise upgrade is required for this migration.
Optional deployment-environment hardening is separate from the cutover, as is
the required org-member merge/review enforcement at public launch.

## Release inventory

| Workflow | Trigger after activation | Output |
| --- | --- | --- |
| `release-vsix.yml` | Manual dispatch from reviewed main | `tensorviz.tensorviz-vscode` on VS Code Marketplace; exact same VSIX and checksum on this repository's `vX.Y.Z` release; rolling `latest` download |
| `release-openvsx.yml` | Successful trusted Release VSIX run, or explicit existing version | Identical Marketplace/GitHub bytes on Open VSX |
| `release-plugin.yml` | Manual dispatch from reviewed main | Apple Silicon macOS and Linux x64 compiled runtimes, client ZIPs, complete catalog ZIP and provenance on `tensorviz-plugins` `vX.Y.Z` releases |
| `docs.yml` | Existing documentation workflow | Independent documentation publishing, unchanged by this migration |

The old repository contains exactly the first three release workflows above.
They must not remain publishers after cutover. The public `tensorviz-plugins` repository remains the active download/catalog
location. Native workflows run here and advance only its audited integration
catalog after verifying public assets. Historical tags/releases are immutable. The hosted website is not part of this product
repository and is not copied or deployed here.

## Reviewed versions and immutable sources

Version bumps and matching package-lock versions must be committed through a
reviewed PR. The prepared extension and plugin candidates are both `0.1.44`; publication is a separate verified workflow step.
This aligns the plugin's old `0.3.110` series without changing extension history.
It is a one-time version-series migration, not a normal semver upgrade.

Publication permits this transition only from public export
`1a3c3618082603030b5ef68b793d9834b8bb7f3d` and requires native-client migration
receipts for both macOS ARM64 and Linux x64. The receipts bind the exact source
revision and compiled runtime hashes, and require Codex `0.165.0` and Claude
`2.1.308` install/update/rollback, native capture and preserved settings checks.
Missing or failed evidence blocks publication. Synthetic manifests alone do not
satisfy these gates. Every subsequent plugin release must match the extension
version in the reviewed source. All other catalog downgrades remain rejected.
Existing tags and release assets are immutable and must not be overwritten.

The native acceptance scope is isolated CLI installation and actual compiled
runtime capture, not signed-in ChatGPT/Claude desktop UI or Cursor acceptance.
Do not claim all clients migrated automatically. Keep the old public release
available for rollback; use an explicit install of its catalog if an updater
rejects the version-series change.

Recheck GitHub, Marketplace and Open VSX version floors immediately before
publication. If a floor has advanced to the candidate version or beyond, prepare
another reviewed version bump; never reuse or overwrite a published version.
The workflow run number is not a version. No release job commits, pushes, merges,
or rewrites source branches. All checkouts use the run's exact SHA.

Extension tags use `vX.Y.Z`. Native plugin tags use `vX.Y.Z` in the separate public `tensorviz-plugins`
repository, so independent version lines cannot collide. Extension tags identify
reviewed source commits. Plugin tags identify export-only commits in the public
repository; `PUBLICATION.json` and `release.json` bind them to the exact private
source SHA without copying source Git objects or history.
The `latest` tag is a fixed download anchor; its mutable VSIX assets point users
to the numbered release for actual source provenance. Numbered assets must never
be overwritten. Public plugin tags and their archives contain the audited integration catalog,
not the private core source. Runtime binaries remain separate release assets.

VSIX build and publish are separate jobs. Open VSX consumes the originating
run's source/version artifact, checks its repository and exact SHA,
and skips build-only results. It verifies source, package identity and byte
checksums against Marketplace and the numbered GitHub release before uploading.

The native workflow and both `pkg` targets pin Node.js `22.23.4`. The embedded
runtime is selected by the locked `@yao-pkg/pkg-fetch` dependency, independently
of the Node executable running the build. When updating these tools, keep the
exact targets, workflow versions and `plugin/licenses/node-<version>.txt` notice
aligned. Copy the unmodified `LICENSE` from the corresponding
[upstream Node release](https://github.com/nodejs/node/blob/v22.23.4/LICENSE);
packaging still checks the version reported by the compiled host and fails if
its exact notice is missing.

Native platform jobs audit their archives and run existing installation/capture
checks. Assembly accepts integration and archive paths relative to the caller's
working directory or absolute paths, and returns an absolute candidate path
beside the first archive. Final catalog assembly requires both targets.
Publication audits the complete catalog,
ZIP contents, runtime manifests and checksums, rejects dirty source provenance,
and checks the exact source is on main. It publishes native assets and the audited catalog only to `tensorviz-plugins`,
without replacing the private source tree. The source token has read-only access;
the separate distribution token writes only to the public destination.

Every release workflow has a non-cancelling concurrency group. This serializes
same-product publication within this repository; it cannot serialize against the
legacy repository. Disable the old publishers before activating new ones.

## Build-only validation before release setup

Release jobs do not reference a deployment environment. With `RELEASES_ENABLED`
unset or `false`, dispatch `release-vsix.yml` and `release-plugin.yml` from
`main` with `publish=false`. This validates candidates without publisher tokens.
Publication jobs retain their manual/main and activation gates. The workflows
inject publishing credentials only into their respective publishing steps.

Production renderer checks remain required. Before this validation, obtain owner
approval to configure `TLDRAW_LICENSE_KEY` as a repository Actions secret using
secure secret management; build jobs cannot read an environment-only secret.
Confirm renderer coverage before setting repository variable
`TLDRAW_DISTRIBUTION_CONFIRMED=true`. Never put keys or tokens in source or logs.

Keep `RELEASES_ENABLED` unset or `false` throughout this validation. If it is
already true, a successful VSIX build can trigger the Open VSX job and inject its
publisher secret before the handoff rejects `publish=false`. Therefore the
input alone is not a guarantee of publisher-credential-free validation. Build artifacts
remain subject to repository access; artifacts of a public repository are public.

## Owner setup before activation

The source repository may remain private. Native assets and the existing public
catalog stay in `tensorviz-plugins`; do not change source visibility for cutover.
The publisher verifies source-main ancestry separately and publishes only the
closed integration allowlist plus audited compiled artifacts. Existing Codex and
Claude marketplace commands and old runtime URLs remain valid.
The destination metadata check uses `repos/tensorviz-org/tensorviz-plugins`
without a trailing slash; GitHub returns 404 for the slash-suffixed endpoint.

1. Before the public launch, configure and verify [org-member merging and required org review](merge-protection.md).
   These are GitHub settings, not assertions enforced by this document. Audit
   outside write/admin access and app/deploy-key bypasses. At the migration audit,
   main was unprotected and the private repository plan did not support rulesets.
   Do not change visibility or billing merely to bypass that blocker.
2. Use repository Actions secrets in `tensorviz-org/tensorviz`, as in the legacy
   publisher setup. A `release` environment is not a prerequisite. Limit repository
   write/admin access to trusted maintainers while public-launch protections are
   deferred; workflow gates are not a substitute for those protections.
3. Configure `VSCE_PAT`, `OVSX_PAT`, and `PLUGIN_PUBLISH_TOKEN` as repository Actions
   secrets using the providers' supported secret-management process. Never copy secrets into
   source or logs. Grant publisher tokens only their existing namespace scope.
   Keep the renderer license available to build jobs as described above.
   This code has not transferred, inspected, or verified secret values.
4. Set repository variable `TLDRAW_DISTRIBUTION_CONFIRMED=true` only after owner confirmation that
   renderer distribution is covered. No analytics token is needed or restored.
   `PLUGIN_PUBLISH_TOKEN` must be scoped only to `tensorviz-plugins`, with Contents
   read/write (and metadata read), not source-repository or organization administration.
   Use a fine-grained token or supported scoped GitHub App credential. Creating or
   configuring persistent credentials requires explicit owner approval; never copy
   them through chat or logs. Native publication uses the job-scoped `GITHUB_TOKEN`
   only for source reads. The distribution token is exposed only to the publish
   step, never dependency installation or compilation.
5. Leave repository variable `RELEASES_ENABLED=false`. Merge the reviewed migration
   after required CI. Org-only review enforcement is deferred until public launch.
   Run manual build-only VSIX and plugin
   workflows on main (`publish=false`). Both still require production renderer
   configuration because they produce distributable packages. Public repository
   artifacts are public, not private candidate storage.
6. Verify both native jobs, clean provenance, artifact inventories, VSIX identity,
   real client install/capture, and platform behavior. Full product regression and
   editor/browser QA are separate gates, not implied by release unit tests.
7. Freeze the legacy release stream. Wait for existing publishing runs to finish,
   record their final versions, then disable `release-vsix.yml`,
   `release-openvsx.yml`, and `release-plugin.yml` in `tensorViz-extension`.
   Confirm no queued or running publisher remains. The 2026-11-10 audit still
   reported manual runs [native 41063482462](https://github.com/tensorviz-org/tensorViz-extension/actions/runs/41063482462)
   and [VSIX 41063477447](https://github.com/tensorviz-org/tensorViz-extension/actions/runs/41063477447)
   queued since October 30. Recheck their current state and resolve any remaining
   runs under owner authorization before activation; this version preparation does
   not cancel them or disable the legacy publishers. Do not delete old releases,
   move old tags, or revoke a token still needed by another authorized workflow.
8. Set repository variable `RELEASES_ENABLED=true` last, then explicitly dispatch
   one reviewed release with `publish=true`. Verify Marketplace, numbered GitHub,
   rolling VSIX and Open VSX bytes. Verify both native runtime downloads and the
   catalog/client ZIPs on `tensorviz-plugins` `vX.Y.Z`.
9. After those checks pass, update public install documentation to the new direct
   download locations. Preserve the old URLs as historical links. Existing installs continue using the public `tensorviz-plugins` marketplace.
   The complete catalog ZIP also supports local installation. Never register
   private source main as a marketplace.
10. Archive tensorViz-extension only after replacement publishing and consumers
   are verified. Preserve historical releases, tags and useful issue records.
   Confirm the archived state and existing download continuity afterward.

## Optional deployment-environment hardening

A protected `release` environment can be added later as a separately approved
security change. It is not required to move or run the legacy publishing model.
If adopted, configure and verify the environment before referencing it in YAML:
require a verified org team reviewer, select only deployment **branch** `main`,
disable admin bypass, and consider preventing self-review. A missing environment
referenced by a workflow is automatically created without protection.

Environment required reviewers are supported for public Free/Team repositories;
private repositories need an eligible Enterprise plan for that optional feature.
This restriction does not apply to repository-secret publishing. Keep renderer
configuration available to build jobs if publisher secrets move into an environment.
See the [merge protection design](merge-protection.md) for the separate public-launch policy.

## Retry and rollback

Rerun the failed publish job with its original artifact and exact SHA. VSIX
artifact names are scoped to run ID so publish-only retries reuse the completed
build. Full VSIX reruns may replace only that run's artifacts and still must
match the existing Marketplace package contents. Candidate artifacts are retained
for seven days (source handoffs for thirty). A whole
native rebuild can change archive timestamps; it is not a byte-identical retry.
The publisher refuses changed assets under an existing version. If the artifact
is lost, prepare a new reviewed version rather than overwrite a release.

An Open VSX retry may explicitly select an existing numbered extension version.
A denied lookup, checksum mismatch, unexpected asset, source mismatch or missing
published asset is a failure, not permission to replace data. Diagnose it first.
For rollback, disable new publication, retain immutable releases, and direct
users to a previous complete catalog or VSIX. Do not restore legacy publication
until the owner has checked version ordering and excluded concurrent publishers.

## Validation

`Release safety checks` runs on PRs, including forks, with a read-only token and
no secrets. It validates workflow syntax and credential-free release regressions.
Publication never runs on `pull_request` or `pull_request_target`. Org-only review
enforcement is deferred until public launch; workflow conditions do not replace
the GitHub protections that must be verified then.

Local focused tests cover source/version validation, forged and stale handoffs,
Marketplace/Open VSX parity, immutable tags/assets, draft recovery, catalog ZIP
contents, native platform contracts and clean provenance. Passing these tests does
not establish a successful native compile, credentialed publication or live
repository protection. Record those results separately during cutover.

## Cross-repository native publication safety

The publisher clones only the public distribution repository. New public commits
contain the closed, audited export and have only public distribution parents.
A retry checks the immutable tag's full exported inventory and source provenance,
recovers paginated drafts, and rejects changed, duplicate, unexpected or missing
published assets. Every asset is anonymously downloaded and checksum-verified
before catalog main advances. Concurrent catalog changes and non-fast-forward
pushes fail closed; a lower version cannot replace a newer catalog. Keep legacy
publishers disabled before activation because concurrency groups do not span repos.

Validate build-only artifacts on macOS arm64 and Linux x64, installed-client capture,
and public catalog/ZIP compatibility before activating publication or archiving
`tensorViz-extension`. Source privacy is not itself a blocker for this distribution
layout. Credentials, renderer coverage, and live consumer continuity remain
cutover gates; this code does not configure or prove them. Optional environment
protections are future hardening, not a migration prerequisite.



## Linux 0.1.44 compatibility gate

Linux x64 native builds and capture tests run in Ubuntu 20.04 (glibc 2.31). Publication also requires native 0.1.43 to 0.1.44 upgrade and rollback tests on Ubuntu 22.04 and Apple Silicon; the previous Linux binary cannot run on the older baseline. The linker-symbol audit must reject requirements newer than glibc 2.31. A reduced launcher check alone is not compatibility evidence.

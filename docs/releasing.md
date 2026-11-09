# Releasing TensorViz

All new release automation belongs to `tensorviz-org/tensorviz`. The migration is
not active merely because these files exist. `RELEASES_ENABLED` must remain unset
or `false` until the owner completes and verifies the cutover below. The initial cutover is manual-only: merging this code cannot start publication,
even if a repository variable was already enabled. No legacy workflow, repository
visibility, permission, secret or environment is changed by this code change.

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

Version bumps, matching package-lock versions, and extension changelog entries
must be committed through a reviewed PR. The migration starts at extension
`0.1.41` and plugin `0.3.109`, above the verified old releases `0.1.40` and
`0.3.108`. Recheck these floors just before cutover if legacy releases continue.
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

Native platform jobs audit their archives and run existing installation/capture
checks. Assembly requires both targets. Publication audits the complete catalog,
ZIP contents, runtime manifests and checksums, rejects dirty source provenance,
and checks the exact source is on main. It publishes native assets and the audited catalog only to `tensorviz-plugins`,
without replacing the private source tree. The source token has read-only access;
the separate distribution token writes only to the public destination.

Every release workflow has a non-cancelling concurrency group. This serializes
same-product publication within this repository; it cannot serialize against the
legacy repository. Disable the old publishers before activating new ones.

## Build-only validation before release setup

Build jobs do not reference a deployment environment. With `RELEASES_ENABLED`
unset or `false`, dispatch `release-vsix.yml` and `release-plugin.yml` from
`main` with `publish=false`. This validates candidates without creating the
`release` environment or needing publisher tokens. Publication jobs still use
that environment and retain their activation gates.

Production renderer checks remain required. Before this validation, obtain owner
approval to configure `TLDRAW_LICENSE_KEY` as a repository Actions secret using
secure secret management; build jobs cannot read an environment-only secret.
Confirm renderer coverage before setting repository variable
`TLDRAW_DISTRIBUTION_CONFIRMED=true`. Never put the key in source or logs, and do
not move publisher tokens out of the protected publishing environment.

Keep `RELEASES_ENABLED` unset or `false` throughout this validation. If it is
already true, a successful VSIX build can trigger the Open VSX job and access its
`release` environment before the handoff rejects `publish=false`. Therefore the
input alone is not a guarantee of environment-free validation. Build artifacts
remain subject to repository access; artifacts of a public repository are public.

## Owner setup before activation

The source repository may remain private. Native assets and the existing public
catalog stay in `tensorviz-plugins`; do not change source visibility for cutover.
The publisher verifies source-main ancestry separately and publishes only the
closed integration allowlist plus audited compiled artifacts. Existing Codex and
Claude marketplace commands and old runtime URLs remain valid.

1. Before the public launch, configure and verify [org-member merging and required org review](merge-protection.md).
   These are GitHub settings, not assertions enforced by this document. Audit
   outside write/admin access and app/deploy-key bypasses. At the migration audit,
   main was unprotected and the private repository plan did not support rulesets.
   Do not change visibility or billing merely to bypass that blocker.
2. Create `release` before enabling publication. Build-only validation above does
   not need it. Require a verified org team
   review, select only deployment **branch** `main`, and disable admin bypass.
   A referenced missing environment is automatically created unprotected.
   Public Free/Team supports environment reviewers; private repositories need an
   eligible Enterprise plan for these reviewer gates.
3. Store `VSCE_PAT`, `OVSX_PAT`, and `PLUGIN_PUBLISH_TOKEN` in that environment using
   the providers' supported secret-management process. Never copy secrets into
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
   Confirm no queued or running publisher remains. Do not delete old releases,
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
Publication never runs on `pull_request` or `pull_request_target`. Keep upstream
main review enforcement active; workflow conditions do not replace it.

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
layout. Environment protections, credentials, renderer coverage, and live consumer
continuity remain separate gates; this code does not configure or prove them.

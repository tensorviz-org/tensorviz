# Releasing TensorViz

All new release automation belongs to `tensorviz-org/tensorviz`. The migration is
not active merely because these files exist. `RELEASES_ENABLED` must remain unset
or `false` until the owner completes and verifies the cutover below. No legacy
workflow, repository visibility, permission, secret or environment is changed by
this code change.

## Release inventory

| Workflow | Trigger after activation | Output |
| --- | --- | --- |
| `release-vsix.yml` | Relevant reviewed main push, or manual dispatch | `tensorviz.tensorviz-vscode` on VS Code Marketplace; exact same VSIX and checksum on this repository's `vX.Y.Z` release; rolling `latest` download |
| `release-openvsx.yml` | Successful trusted Release VSIX run, or explicit existing version | Identical Marketplace/GitHub bytes on Open VSX |
| `release-plugin.yml` | Relevant reviewed main push, or manual dispatch | Apple Silicon macOS and Linux x64 compiled runtimes, client ZIPs, complete catalog ZIP and provenance on this repository's `plugin-vX.Y.Z` release |
| `docs.yml` | Existing documentation workflow | Independent documentation publishing, unchanged by this migration |

The old repository contains exactly the first three release workflows above.
They must not remain publishers after cutover. The old `tensorviz-plugins`
repository remains an immutable historical download/catalog location; the new
pipeline does not push to it. The hosted website is not part of this product
repository and is not copied or deployed here.

## Reviewed versions and immutable sources

Version bumps, matching package-lock versions, and extension changelog entries
must be committed through a reviewed PR. The migration starts at extension
`0.1.41` and plugin `0.3.109`, above the verified old releases `0.1.40` and
`0.3.108`. Recheck these floors just before cutover if legacy releases continue.
The workflow run number is not a version. No release job commits, pushes, merges,
or rewrites source branches. All checkouts use the run's exact SHA.

Extension tags use `vX.Y.Z`. Native plugin tags use `plugin-vX.Y.Z`, so independent
version lines cannot collide. Both point directly to the reviewed source commit.
The `latest` tag is a fixed download anchor; its mutable VSIX assets point users
to the numbered release for actual source provenance. Numbered assets must never
be overwritten. Plugin source archives are source, not the compiled catalog.

VSIX build and publish are separate jobs. Open VSX consumes the originating
run's source/version artifact, checks its repository and exact SHA,
and skips build-only results. It verifies source, package identity and byte
checksums against Marketplace and the numbered GitHub release before uploading.

Native platform jobs audit their archives and run existing installation/capture
checks. Assembly requires both targets. Publication audits the complete catalog,
ZIP contents, runtime manifests and checksums, rejects dirty source provenance,
and checks the exact source is on main. It publishes only to this repository,
without replacing its source tree with an integration catalog.

Every release workflow has a non-cancelling concurrency group. This serializes
same-product publication within this repository; it cannot serialize against the
legacy repository. Disable the old publishers before activating new ones.

## Owner setup before activation

1. Configure and verify [org-member merging and required org review](merge-protection.md).
   These are GitHub settings, not assertions enforced by this document. Audit
   outside write/admin access and app/deploy-key bypasses. At the migration audit,
   main was unprotected and the private repository plan did not support rulesets.
   Do not change visibility or billing merely to bypass that blocker.
2. Create `release` before running release workflows. Require a verified org team
   review, select only deployment **branch** `main`, and disable admin bypass.
   A referenced missing environment is automatically created unprotected.
   Public Free/Team supports environment reviewers; private repositories need an
   eligible Enterprise plan for these reviewer gates.
3. Store `VSCE_PAT`, `OVSX_PAT`, and `TLDRAW_LICENSE_KEY` in that environment using
   the providers' supported secret-management process. Never copy secrets into
   source or logs. Grant publisher tokens only their existing namespace scope.
   This code has not transferred, inspected, or verified their values.
4. Set repository variable `TLDRAW_DISTRIBUTION_CONFIRMED=true` only after owner confirmation that
   renderer distribution is covered. No analytics token is needed or restored.
   `PLUGIN_PUBLISH_TOKEN` is retired: the job-scoped `GITHUB_TOKEN` needs
   `contents: write` only for same-repository release/tag publication.
5. Leave repository variable `RELEASES_ENABLED=false`. Merge the reviewed migration
   only after required CI and org review. Run manual build-only VSIX and plugin
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
   catalog/client ZIPs on `plugin-vX.Y.Z`.
9. After those checks pass, update public install documentation to the new direct
   download locations. Preserve the old URLs as historical links. New compiled
   installs use the complete catalog ZIP as a local marketplace; do not register
   source main as the generated marketplace.

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

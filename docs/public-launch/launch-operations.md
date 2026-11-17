# Release TensorViz 1.0.0 from the public source repository

The extension and plugin source candidates are **1.0.0**. The candidate workflow
builds without publication by default. Set `publish_after_checks=true` for the
standard launch: successful candidate checks automatically dispatch the trusted
publisher. Existing 0.1.48 downloads and catalog entries stay usable until the
new assets and canonical catalog are verified.

## Build and publish once

After the release change has passed review and merged into `main`, record its
exact commit S. Before publication can succeed, the immutable lightweight
`v1.0.0` tag must identify S; prepare that tag for the reviewed source and never
move an existing tag. Run the candidate workflow manually:

```sh
gh workflow run release-unified.yml --repo tensorviz-org/tensorviz --ref main \
  -f publish_after_checks=true
```

The workflow builds the licensed renderer, all six native runtimes, extension
VSIX and documentation from S. It retains native installed-catalog acceptance,
no-Torch startup and capture checks, and performs actual Codex 0.165.0 and Claude
2.1.308 upgrade/rollback from the pinned 0.1.48 catalog on macOS ARM64 and Linux
x64. All those jobs and the extension/documentation checks must succeed.

The resulting `unified-release-candidate` artifact contains the complete plugin
candidate, versioned VSIX, exact documentation ZIP, closed SHA256 inventory,
migration receipts and an additive `catalog-overlay/`. GitHub OIDC signs an
artifact attestation for `unified-release.json`; that inventory binds every
published package. No maintainer signing key or additional repository secret is
needed. This workflow uses hosted runners only.

After the complete candidate bundle succeeds, its `dispatch_publication` job
starts `docs.yml` with source S and this candidate's run ID R. A standard launch
needs no second manual publisher dispatch. Check the downstream **Documentation**
run as well: a successful dispatch is not successful publication. For a build-only
run, omit `publish_after_checks` or set it to `false`.

This automated evidence replaces the original proposed 20 manually signed
client/platform receipts for this launch. It proves the named native/CLI checks
and package provenance. It does not claim graphical acceptance of every editor,
Cursor/Claude desktop, older operating system, or model environment. The original
manual-signature verifier remains available for policies that explicitly use it.

## Publish the same bytes

The automatically dispatched **Documentation** workflow publishes the candidate's
exact bytes. Its separate unified publication job uses the Google Cloud trust
already granted to that workflow's exact main-branch identity. The build workflow
does not receive Firebase access or registry credentials.

For a build-only run, failed dispatch, or recovery after verifying a partial
publication, retain the successful candidate run ID R and source S. Ensure the
lightweight `v1.0.0` tag identifies S, then dispatch the same publisher manually:

```sh
gh workflow run docs.yml --repo tensorviz-org/tensorviz --ref main \
  -f release_source_sha=SOURCE_SHA -f candidate_run_id=CANDIDATE_RUN_ID
```

Replace both placeholders. Do not launch a second publisher while the automatic
publication run is active. Keep historical tags and assets unchanged.

Before publisher credentials, trusted-main preflight checks S is in main history,
the candidate run belongs to this repository/main/workflow, and all versions agree.
Admission then verifies the signed exact-byte inventory, actual successful native
and migration jobs, previous catalog pin, runtime hashes and migration receipts.
GitHub artifact verification pins repository, workflow, source/ref, hosted runner
and exact run attempt. No source or catalog branch is pushed by publication.

The publisher first uploads and reads back the combined GitHub release inventory
at S, then checks its anonymous availability. Only then does it publish the exact
VSIX to Marketplace and Open VSX, and deploy the signed docs bytes to
`tensorviz-docs`. Existing matching versions are reused; differing bytes stop the
channel without overwrite. The old VSIX/Open VSX/plugin workflows reject 1.x.

All four existing secrets remain: `TLDRAW_LICENSE_KEY` for builds, and `VSCE_PAT`
and `OVSX_PAT` only for the corresponding publication tools. The legacy
`PLUGIN_PUBLISH_TOKEN` is not used by canonical publication. The job-scoped
`GITHUB_TOKEN` writes the same repository's release assets. Existing isolated
docs variables and Google workload identity are reused.

Registry/docs submission intents are written and read back **before** calling the
external publisher. They live as small JSON assets in a separate retained draft
release named `publication-journal-vVERSION-SOURCE_SHA`. It contains no credentials
or test logs. Keep this draft and its assets permanently; do not publish, delete,
reset or expire them. Runner loss therefore cannot erase an uncertain submission.
A retry first checks public bytes; an unresolved intent blocks blind re-upload.
Partial publication reports the affected channel and never reports full success.

Normal independent documentation updates still use `docs.yml` with both inputs
blank. They build and deploy the latest selected `main` as before.

## Advance the canonical catalog

Publication does not advance the Git catalog. Review the candidate's
`catalog-overlay/` in a second, separate additive source PR to advance
`distribution/plugin-catalog/` and root client indexes from the retained 0.1.48
overlay to 1.0.0. This PR must preserve the
source tree, license and root product documentation. The catalog records build
source S even if its reviewed catalog commit C comes later. Release `v1.0.0`
continues to identify S. Never retarget the legacy destructive catalog writer.

Verify actual client installation from the canonical Git catalog, preserved
settings and rollback before changing current install commands or archiving the
legacy catalog. The downloadable 1.0.0 catalog and client ZIPs originate from
this repository's release; historical 0.x downloads stay in their original repo.

## Public repository settings

Main requires a pull request and one approval under the active ruleset. The
approval can come from any eligible writer; organization-only reviewer and merge
restrictions remain to be configured. Release-tag immutability rules are active.
Secret scanning, push protection, private vulnerability reporting, and Discussions
were enabled at the public launch check. These settings do not establish tested
behavior for every reviewer or release actor; retain the live settings check in
the release audit.

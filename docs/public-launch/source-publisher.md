# Unified source release: implementation and activation gates

The new additive `plugin/scripts/publish-source-release.mjs` never clones,
clears, commits or pushes source, creates/moves refs, replaces assets or changes
legacy workflows. It publishes one exact combined inventory to an existing
lightweight source tag. The core repository must already be public. A draft is
exposed only after uploaded bytes are read back and hashed and the source tag is
rechecked. Partial drafts resume; differing published bytes never get replaced.

## Implemented, locally tested safeguards

- Repository selection is version-based: existing 0.x exports/URLs stay in the
  legacy repository, 1.x and later use core. The legacy destructive publisher
  explicitly refuses 1.x, regardless of destination. Historical URLs unchanged.
- Source-tag publication requires a clean exact reviewed checkout, canonical
  main ancestry, matching source tag, and aligned nine source/version documents.
- Combined bundle admission reuses complete plugin candidate validation, requires
  all its bytes plus one versioned VSIX and extension source/hash provenance,
  verifies VSIX package identity/version, and rejects extra/missing assets.
- Policy is loaded from the reviewed Git object, never an untrusted bundle. The
  committed policy defaults to disabled and has no trusted signing keys.
- Admission requires 20 Ed25519-signed local evidence records: six native
  OS/architecture targets for extension/Codex, plus macOS/Linux targets for
  Claude/Cursor. Windows Claude/Cursor are not currently advertised and are not
  silently claimed as supported. Each binds the
  complete artifact inventory, source SHA, version, exact prior catalog commit,
  client version, old/new repo identities, log hash and completion time. Required
  checks are clean install, upgrade, rollback, preserved settings, native capture,
  no-Torch startup, missing-Torch guidance and catalog-source migration.
- A signature authenticates a trusted reviewer's attestation; it does not itself
  demonstrate that tests were executed honestly. Review original logs before
  signing. No trusted key or passing evidence is manufactured by preparation.

No hosted test jobs or live publication were run for this preparation. The
workflow is an inactive `.example` file, release-only when eventually activated.
Local tests use mock transport and generated test-only signing keys, not native
acceptance evidence.

## Bundle layout

```
plugin/                 # existing audited candidate directory, canonical URLs
assets/                 # flat immutable upload inventory
  tensorviz-runtime-... # all files from plugin release.json, unchanged
  plugin-release.json  # byte-identical copy of plugin/release.json
  tensorviz-VERSION.vsix
  extension-provenance.json
unified-release.json    # schemaVersion:1, repository, version, sourceRevision, files
admission.json          # 20 signed receipt envelopes
```

`files` maps every assets/ filename to SHA256. Extension provenance has
`schemaVersion:1`, `version`, `sourceRevision`, `sha256` (VSIX hash). The manifest
and admission envelopes are additionally uploaded verbatim. A receipt envelope
has `keyId`, base64 `payload` (JSON receipt), base64 `signature`; the policy maps
key IDs to Ed25519 public PEM keys and pins client versions/prior catalog. Sign
exact serialized payload bytes with a reviewer-owned key only after reviewing
local logs. Private signing keys do not belong in source or artifacts.

## Before activating

1. Pin the final previous 0.x version/catalog commit, extension identity and
   client versions in the policy. Add trusted reviewer public keys and enable
   only in a reviewed launch change. Do not rewrite the old release train.
2. Generate/audit the new canonical catalog and check its source provenance
   relationship explicitly. Catalog commits can contain an earlier build-source
   SHA; do not invent a circular self-referential SHA. Source-tag publisher never
   advances catalog pointers automatically; reviewed catalog routing must already
   be committed and tested.
3. Build the combined exact-artifact bundle. Perform all real local native/client
   tests, retain original logs, obtain signed evidence and verify the bundle.
   These acceptance tests remain outstanding, not represented by unit test results.
4. Review/pin workflow action revisions and candidate artifact origin, establish
   release environment protection, and move the template only at launch.
5. Coordinate marketplace release channels with per-channel receipts. GitHub asset
   publication does NOT publish VS Code Marketplace/OpenVSX or vendor directories.
   Separate publishers must not race to create/rewrite the same source release.
6. Verify migrated clients no longer need legacy catalog updates before archiving
   old repos. Preserve immutable historical tags/assets and rollback downloads.

Run local preparation tests:

    node --test plugin/test/unit/source-publisher.test.mjs plugin/test/unit/source-release-admission.test.mjs

No source versions have been bumped and no repository has been archived or made
public by this work.

## Catalog review and multi-channel orchestration

`canonical-catalog-commit.mjs` prepares an additive Git tree/commit using a pinned
main parent. It touches only generated catalog namespaces, preserves the source
tree, fails on concurrent main changes, and refuses silent removal of stale
catalog files. It does NOT update any branch or tag. Open and review a PR for the
returned commit; merging remains an explicit launch step. The transport adapter
must inventory actual existing catalog files, not an assumed empty directory.

`release-channels.mjs` coordinates GitHub, VS Code Marketplace, OpenVSX and docs
through explicit adapters. Verified already-published channels are reused;
conflicting bytes are never overwritten; partial failures are reported per
channel. GitHub availability failure prevents advertising through marketplaces.
Production marketplace/docs adapters, vendor authentication and release receipts
still require wiring to existing release commands. This is tested orchestration,
not a claim those channels have been published or are fully automated.

Canonical1.x public document rendering now rewrites installation repository
commands/current URLs and stale private-source assertions while preserving old
versioned asset URLs. Actual installed-client acceptance remains required.

Extension evidence names core as both the old and new source repository, because
extension releases already originate there. Its `sourceMigration` check confirms
preserved marketplace identity and source routing, not a fictitious move from the
plugin repository. Plugin-client evidence names the legacy plugin repository as
the prior source. Bundle manifest/evidence bytes and VSIX metadata are snapshotted
before validation and reused for upload; later input-path replacements cannot
change the bytes admitted for publication.

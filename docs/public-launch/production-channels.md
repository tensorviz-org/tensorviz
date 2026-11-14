# Staged production release adapters

This is preparation code only. Both launch policy and channel policy remain
**disabled**. No Marketplace, OpenVSX, Firebase or GitHub publication was performed
to test these adapters. Unit tests use local fixtures and mocked registry/CLI calls.

## What is implemented

`publish-unified-channels.mjs BUNDLE SOURCE_ROOT DURABLE_JOURNAL_ROOT` reuses the
exact-source, source-main ancestry, version, signed local acceptance and combined
inventory admission before creating any channel adapters. It never calls the old
GitHub/rolling-release synchronization scripts. The bundle now also requires
`tensorviz-docs-VERSION.zip`; its hash is covered by the same signed inventory as
the VSIX/plugin archives. Docs cannot be substituted by a source-SHA label alone.

The GitHub channel verifies the complete immutable inventory anonymously. Only
after GitHub passes do VS Code Marketplace/OpenVSX receive the verified VSIX.
Existing matching versions are reused; differing public bytes stop that channel
without overwrite. A Marketplace that re-signs or repackages bytes will fail
strict equality: do not normalize or silently accept a different artifact. Review
that vendor behavior using the actual launch package before activation.

Docs publish only to the separately reviewed **tensorviz-docs** Firebase project
and site. `prepareDocsArtifact` verifies local `build-info.json` source/revision/
root base, captures files, stamps `release-info.json` with version and content
hashes, and packages without following links or including hidden/source files.
The verifier checks the same exact bytes without extracting paths. Deployment
stages only verified files and a generated site-specific Firebase configuration.
Anonymous readback checks the marker and every served file. A newer or conflicting
published version is never automatically rolled back.

## Tool supply chain

The private `tooling/release-tools` package pins vsce **4.0.0**, ovsx **1.3.0**, and
Firebase Tools **15.36.0**. Its lock was generated from official npm metadata only
with `--package-lock-only --ignore-scripts --no-audit --no-fund`; no packages were
installed or executed in preparation. All 763 dependency records are official
npm registry HTTPS URLs with SHA512 integrity.

At activated publication, the wrapper copies package/lock bytes from the exact
reviewed Git commit into a private temporary directory, rejects local/link/custom
registry packages, and runs fresh `npm ci --ignore-scripts`. It does not trust
pre-existing node_modules or cached credentials. CLI version/location checks alone
are not supply-chain attestation; the isolated integrity-checked installation is
mandatory. Dependencies' actual publisher compatibility still needs a local
preflight before enabling the release.

## Credential isolation

Registry tokens are passed only through the respective CLI environment, never
arguments or returned errors. Each submission has a fresh HOME/config directory.
No inherited Firebase, npm, GitHub, or other application credentials are used.

Docs require approved GitHub OIDC context and the reviewed isolated-project
service account. The ADC must specify official Google STS, the exact configured
impersonation target, and the current job's exact GitHub OIDC URL/bearer token.
Executable/file/arbitrary credential sources are refused. A sanitized private ADC
snapshot is passed to Firebase, not the original mutable file. The OIDC JWT audience
is `https://iam.googleapis.com/...`; the ADC STS audience is separately
`//iam.googleapis.com/...`. These deliberately differ as in auth v3.

## Retry semantics and remaining activation steps

Submission intent is saved **before** calling a publisher. A timeout or uncertain
result is not automatically retried as a new upload. Subsequent invocations inspect
public state first; matching visible bytes recover without another upload. If an
intent exists but nothing is visible, investigate the registry/job before clearing
it. Journal directories are scoped by version and source SHA so a prior release
cannot block the next release. Results report per-channel verified/blocked status;
partial success never becomes an all-channels success claim.

Use a genuinely durable journal root. Ephemeral hosted-runner storage alone is not
durable across crashes. Persist/restore the journal using an approved storage
arrangement before connecting this CLI to Actions. The existing workflow example
remains inactive and publishes only the source assets until that arrangement is
reviewed. Do not delete an uncertain journal merely to make a retry proceed.

Before activation: pin real previous catalog/client versions and reviewer keys;
perform actual local OS/client acceptance; choose actual docs site/service account/
WIF configuration and explicitly reviewed artifact CDN hosts; verify the npm CLI
preflight; disable/reject competing legacy 1.x publishers; review protected launch
credentials/environment and durable journal storage. Nothing here changes repo
visibility, archives repositories, merges a launch PR, or runs hosted tests.

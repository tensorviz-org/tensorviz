# Build and publish the documentation

The documentation source, theme and deployment live in this repository. The site
is configured for <https://docs.tensorviz.dev/>. The marketing website stays in
its separate private repository and links to the docs; contributors do not need
access to the website code or its deployment credentials.

- `index.md`: docs overview.
- `README.md`: guide index.
- Markdown guides, `reference/` and `images/`: documentation content.
- `.vitepress/`: configuration, theme, content preparation and verification.
- `public/`: logos and static assets.
- [Documentation workflow](../.github/workflows/docs.yml): checks and Firebase Hosting deployment.
- [Hosting configuration](firebase.json): static docs site, deployed as `hosting:docs`.

## Contribute and preview

Use Git, Node.js 22 (with npm) and Python 3.12. The docs-only
[fork and PR workflow](../CONTRIBUTING.md#documentation-contributions) requires
neither the product development environment nor deployment credentials.

Edit the original Markdown or use **Edit this page on GitHub** on a published
page, then submit a pull request. GitHub offers a fork to contributors without
write access. Edit links map generated pages to their original files on `main`.

From the repository root:

```sh
npm ci --prefix docs
npm run dev --prefix docs
```

Before submitting changes:

```sh
python3 tooling/docs/check_links.py
DOCS_BASE=/tensorviz/ npm run build --prefix docs
npm run build --prefix docs
git diff --check
```

The source check uses only Python's standard library and does not fetch external
URLs. Both builds verify local rendered links and anchors. The prefixed build
catches paths that would fail in a subdirectory preview; the final build restores
the production `/` base. Run `npm run preview --prefix docs` to inspect that build.

The build writes `docs/.vitepress/dist/`, with articles such as `/quickstart.html`
at the docs host. It exports Markdown for **Copy page**, **View Markdown** and
`/llms.txt`, and verifies rendered links, anchors, source-edit links, Markdown
downloads and font licenses. `/build-info.json` identifies the source repository,
commit and prefix. Generated content stays in ignored folders; edit the sources.

VitePress generates `/sitemap.xml` for `https://docs.tensorviz.dev/`, including
the configured `DOCS_BASE` in prefixed builds. It lists documentation pages and
excludes the 404 page and static resources. `public/robots.txt` allows all
crawlers, explicitly includes search and AI crawlers in the same allow group,
and points to the production sitemap. Both builds verify these resources and
the quickstart sitemap URL.

## Publishing with GitHub Actions and Firebase

GitHub Actions builds the docs from this repository and publishes the checked
artifact to Firebase Hosting in the separate `tensorviz-docs` project. The live
docs host is <https://docs.tensorviz.dev/>. The website at <https://tensorviz.dev/>
redirects its old `/docs/` URLs to this host while preserving article paths.
GitHub Pages is not involved, and publishing does not require public repository
visibility.

### Maintainer deployment configuration

The existing deployment uses short-lived credentials and a docs-only identity.
Keep these isolation requirements when maintaining it; contributors do not need
to provision anything:

1. Keep the Hosting site and `docs.tensorviz.dev` in `tensorviz-docs`, separate
   from the website project. A different Hosting site in the website project is
   insufficient: Hosting roles apply to the whole project, and the workflow's
   target selection is not an IAM boundary.
2. Keep the docs deployment service account in `tensorviz-docs`, with Firebase
   Hosting Admin (`roles/firebasehosting.admin`) on that project only, plus a
   custom role containing only `serviceusage.services.get`; the Firebase CLI uses
   it to check whether the Hosting API is enabled. Do not reuse the website's
   deployment identity or grant the docs identity access to the website project.
3. Keep the dedicated Workload Identity provider for GitHub's issuer
   `https://token.actions.githubusercontent.com`. Map `google.subject` to
   `assertion.sub` and `attribute.repository_id` to `assertion.repository_id`.
   Restrict the provider's attribute condition to this repository, `main`, this
   workflow and its publishing events:

   ```text
   assertion.repository_id == '1435323397' &&
   assertion.repository_owner_id == '328278954' &&
   assertion.ref == 'refs/heads/main' &&
   assertion.workflow_ref == 'tensorviz-org/tensorviz/.github/workflows/docs.yml@refs/heads/main' &&
   assertion.event_name in ['push', 'workflow_dispatch']
   ```

   Grant `roles/iam.workloadIdentityUser` on the docs service account to only
   `principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/attribute.repository_id/1435323397`.
   No service-account JSON key or Firebase login token is needed in GitHub.
4. Maintain these repository Actions variables. A GitHub deployment environment
   or paid environment-protection feature is not required:

   | Variable | Value |
   | --- | --- |
   | `DOCS_FIREBASE_PROJECT_ID` | `tensorviz-docs` |
   | `DOCS_FIREBASE_SITE_ID` | The dedicated docs Hosting site ID |
   | `DOCS_WORKLOAD_IDENTITY_PROVIDER` | Full `projects/…/locations/global/workloadIdentityPools/…/providers/…` resource name |
   | `DOCS_DEPLOY_SERVICE_ACCOUNT` | Docs service account email in `tensorviz-docs` |
   | `DOCS_PUBLISH_ENABLED` | `true` once setup is complete |

5. Merge the desired PRs, then run **Documentation** manually from upstream
   `main`. Nothing runs or publishes on a PR, push or merge. When
   `DOCS_PUBLISH_ENABLED=true`, the manual run builds, checks and publishes the
   latest selected source together. When it is `false`, the manual run only
   builds and checks. Review and approval before merging remain the
   maintainers' responsibility. The workflow does not enforce PR review policy.

```sh
gh workflow run docs.yml --repo tensorviz-org/tensorviz --ref main
```

For a batch of three or four PRs, merge them all before running this command once.
No skip marker, temporary variable change or workflow toggle is needed. A manual
run can publish a commit containing `[skip ci]`; that marker suppresses automatic
push/PR events and does not block an explicitly requested run.

Manual builds run the source check and both builds with read-only repository
access. There is no `pull_request_target` execution and no PR deployment.
The build job cannot request an OIDC token. Only the separate deployment job on
upstream `main` for `workflow_dispatch` can request a Google identity. The existing
provider's permission for `push` may remain configured, but no push event starts
this workflow. The deployment job first checks
the project, service account and artifact provenance, then selects the docs target
from the configured site ID and deploys only `hosting:docs`. Fork repositories and
other branches cannot enter that job, even if they enable publishing variables.
The configuration contains no functions, database rules or website source.
The build needs no website checkout, production Canvas key or model weights.

Provider setup is described in
[Google's authentication action](https://github.com/google-github-actions/auth#workload-identity-federation-through-a-service-account).
Site targeting is described in
[Firebase's multisite guide](https://firebase.google.com/docs/hosting/multisites).

### Verify a publication

Fetch <https://docs.tensorviz.dev/build-info.json> and compare `revision` with the
successful upstream **Documentation** run's commit, `source` with
`https://github.com/tensorviz-org/tensorviz`, and `base` with `/`. A local build or
a successful PR build does not prove which revision is live. Check the overview,
quickstart, local search, GitHub edit links and Markdown downloads at the custom
domain as well.

The website's legacy redirects use Firebase regex rules: `^/docs/?$` to
`https://docs.tensorviz.dev/`, and `^/docs/(?P<path>.+)$` to
`https://docs.tensorviz.dev/:path`. The named capture retains trailing slashes on
directory-style article URLs.

Guide images use the shared Canvas renderer with a source-preview fixture.
They illustrate controls; they do not show measured runtime shapes.

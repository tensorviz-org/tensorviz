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
npm run build --prefix docs
DOCS_BASE=/tensorviz/ npm run build --prefix docs
```

The build writes `docs/.vitepress/dist/`, with articles such as `/quickstart.html`
at the docs host. It exports Markdown for **Copy page**, **View Markdown** and
`/llms.txt`, and verifies rendered links, anchors, source-edit links, Markdown
downloads and font licenses. `/build-info.json` identifies the source repository,
commit and prefix. Generated content stays in ignored folders; edit the sources.

## Publishing with GitHub Actions and Firebase

GitHub Actions builds the docs from this repository and publishes the checked
artifact to Firebase Hosting. This follows the website's deployment approach;
GitHub Pages is not involved. Publishing does not require changing repository
visibility. Deployment stays disabled until the hosting identity is configured.

An administrator completes this setup once:

1. Create a dedicated docs Hosting site. It may share the website's Firebase
   project, but must have a different site ID. Attach `docs.tensorviz.dev` in the
   Firebase console and apply the exact DNS records Firebase supplies. Wait for
   Firebase to provision its HTTPS certificate.
2. Create a docs deployment service account and grant Firebase Hosting Admin
   (`roles/firebasehosting.admin`) on the selected project. Do not reuse or extend
   the website's deployment identity. Hosting roles apply to the project; use a
   separate Firebase project if the docs identity must be unable to deploy the
   website's Hosting site. The workflow's target selection is not an IAM boundary.
3. Create a dedicated Workload Identity provider for GitHub's issuer
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
4. Configure these repository Actions variables. A GitHub deployment environment
   or paid environment-protection feature is not required:

   | Variable | Value |
   | --- | --- |
   | `DOCS_FIREBASE_PROJECT_ID` | Firebase project containing the docs site |
   | `DOCS_FIREBASE_SITE_ID` | The dedicated docs Hosting site ID |
   | `DOCS_WORKLOAD_IDENTITY_PROVIDER` | Full `projects/…/locations/global/workloadIdentityPools/…/providers/…` resource name |
   | `DOCS_DEPLOY_SERVICE_ACCOUNT` | Docs service account email |
   | `DOCS_PUBLISH_ENABLED` | `true` once setup is complete |

5. Run **Documentation** from `main` for initial publication. Accepted docs
   changes on `main` then publish automatically.

Pull requests build with read-only repository access. Only the separate deployment
job on `main` can request a Google identity and publish the artifact. It selects
the docs target from the configured site ID, then deploys only `hosting:docs`.
The configuration contains no functions, database rules or website source.
The build needs no website checkout, production Canvas key or model weights.

Provider setup is described in
[Google's authentication action](https://github.com/google-github-actions/auth#workload-identity-federation-through-a-service-account).
Site targeting is described in
[Firebase's multisite guide](https://firebase.google.com/docs/hosting/multisites).

Verify the overview, quickstart, local search, GitHub edit links and Markdown
download at the custom domain before switching website links. The website's old
`/docs/` URLs can then redirect to this host, retaining article paths and anchors.
Keep the existing preview available until that cutover succeeds. When moving the
website's old `/docs/` routes, redirect `/docs` and `/docs/` to the docs homepage,
and `/docs/:path*` to `https://docs.tensorviz.dev/:path`.

Guide images use the shared Canvas renderer with a source-preview fixture.
They illustrate controls; they do not show measured runtime shapes.

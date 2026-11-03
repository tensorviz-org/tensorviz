# Documentation site

VitePress presents the Markdown already in the repository. From this directory, `npm ci` then `npm run dev` serves a local preview, `npm run build` writes the static site to `site/.vitepress/dist`, and `npm run preview` serves that build. `prepare.mjs` copies the docs into `site/` at build time (those copies are gitignored); edit the repository files, not the copies. `DOCS_BASE` defaults to `/` and is `/tensorviz/` for GitHub Pages.

Pushes to `main` that change the docs or this directory, and matching pull requests, build the site. Deployment is off until a repository admin sets Settings > Pages > Source to GitHub Actions and the repository Actions variable `PAGES_ENABLED` to `true`. With both configured, pushes to `main` and manual workflow runs also deploy; pull requests never deploy. Without the variable, builds succeed without requiring a configured Pages site.

## Editing the user-facing site

- Edit `home.md` for the landing page; it is independent of the repository README.
- `site/.vitepress/theme/InstallOptions.vue` owns the interactive install picker. Verify public package identifiers and download URLs when updating it; published packages and this source checkout have separate versions.
- `home.md` uses a compact install picker without a product screenshot. Article columns stay narrow; only the overview uses the wider layout.
- `docs/quickstart.md` is a complete first-graph tutorial. Keep its model, expected shapes, and source-preview caption consistent.
- `prepare.mjs` preserves readable installation fallbacks inside any `install-options` comment markers and uses the interactive picker for the website.
- `docs/images/` contains annotated renderer screenshots. Preparation copies these assets into the site, so the same Markdown image links work on GitHub and in the website. Provenance is recorded in [the docs guide](../docs/documentation-site.md).
- Article pages expose **Copy page** and **View Markdown** through `PageActions.vue`. The build writes Markdown to `/markdown/` and a curated guide index to `/llms.txt`; both respect `DOCS_BASE`.
- Keep task guides short. Detailed behavior lives in `docs/reference/`, linked from the guides and the collapsed Reference navigation.
- Release-document templates in `docs/public-plugin/` and `docs/standalone-*` stubs are excluded from the user site. They remain available to packaging tools.

After a docs change, run the docs-link check and `npm run build` here. Check the homepage install tabs, copy buttons, Quickstart, navigation, and a mobile viewport. `DOCS_BASE=/tensorviz/ npm run build` checks the GitHub Pages prefix too.

# Documentation site

VitePress presents the Markdown already in the repository. From this directory, `npm ci` then `npm run dev` serves a local preview, `npm run build` writes the static site to `site/.vitepress/dist`, and `npm run preview` serves that build. `prepare.mjs` copies the docs into `site/` at build time (those copies are gitignored); edit the repository files, not the copies. `DOCS_BASE` defaults to `/` and is `/tensorviz/` for GitHub Pages.

Pushes to `main` that change the docs or this directory, and matching pull requests, build the site. Deployment is off until a repository admin sets Settings > Pages > Source to GitHub Actions and the repository Actions variable `PAGES_ENABLED` to `true`. With both configured, pushes to `main` and manual workflow runs also deploy; pull requests never deploy. Without the variable, builds succeed without requiring a configured Pages site.

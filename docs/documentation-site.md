# Build the documentation

Documentation content and the website live in `docs/`:

- `index.md`: website homepage.
- `README.md`: guide index.
- Markdown guides, `reference/` and `images/`: documentation content.
- `.vitepress/`: website configuration, theme and build script.
- `public/`: logos and static assets.

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

The build preserves `/docs/` article URLs, writes the site to `.vitepress/dist/`,
and exports article Markdown for **Copy page**, **View Markdown** and `/llms.txt`.
Generated content stays in the ignored `.vitepress/cache/`; edit the source guides.

GitHub Pages deployment requires Pages to use GitHub Actions and the repository
variable `PAGES_ENABLED=true`. Pull requests build without deploying.

Guide images use the shared Canvas renderer with a source-preview fixture.
They illustrate the controls; they do not show measured runtime shapes.

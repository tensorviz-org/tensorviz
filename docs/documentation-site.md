# Documentation website

The documentation website is built locally with VitePress from the repository Markdown. See [site development](../docs-site/README.md) for commands and deployment configuration.

## Reading paths

- **Overview** shows the product beside a compact installation picker.
- **Quickstart** provides the complete `MlpTiny` example, opening steps, and expected capture shapes.
- **Installation options** covers clients, compatibility, package identities, and source builds.
- Task guides show a focused annotated image, short instructions, and links to the detailed reference.

The updated TensorViz logo, neutral colors, and Geist fonts are shared across these pages. Article columns retain their reading width; the overview has a wider layout that stacks on small screens.

## Give a page to a coding agent

Choose **Copy page** to copy an article as Markdown, or **View Markdown** to open its text. Copy failures keep the Markdown link available. The static build exports one Markdown file per page under `/markdown/` and lists the main task guides in `/llms.txt`. These paths include the configured site prefix, such as `/tensorviz/`.

Markdown is generated from source content, including readable install commands rather than interactive Vue controls. No documentation service or external AI request is needed.

## Image provenance

The three images in `docs/images/` are screenshots of the repository's shared `LocalBrowserApp` and Canvas renderer with numbered documentation annotations:

- `explore-graph.png`: the `graphReview.json` fixture's **before** projection (`MlpTiny`). It is a source preview, not a runtime capture.
- `follow-code.png`: the same source preview with the Code focus control and a mapped layer identified.
- `review-changes.png`: the fixture's before/after projections rendered as a source graph review. The changed revision widens the projection, replaces ReLU with GELU, and adds LayerNorm.

They were rendered from this checkout with a local scripted transport. They demonstrate the source UI and do not establish installed-extension acceptance, runtime shape measurements, or a live editor navigation result. The quickstart's expected shapes describe what a successful capture of its example should produce.

When updating the images, use a public example or a repository fixture. Keep source previews, runtime captures, and source comparisons labeled accurately. Do not include local credentials, private model paths, evaluation data, or unrelated product features.

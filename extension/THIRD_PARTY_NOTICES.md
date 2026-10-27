# Third-party notices

TensorViz's MIT license applies to its own code. Dependencies retain their own
copyrights and licenses.

- `DEPENDENCY_NOTICES.txt` contains the original license and notice files for
  npm packages included in the extension host bundle.
- `media/app/static/js/*.LICENSE.txt` and
  `media/app/viewer/static/js/*.LICENSE.txt` retain the notices extracted from
  the canvas and viewer bundles. Keep them alongside those bundles.
- The tldraw 5.0.1 SDK and assets use the separate
  [tldraw license](../runtime/licenses/tldraw-5.0.1.md), copied verbatim from its upstream
  release. In a packaged VSIX that copy is placed at `licenses/tldraw-5.0.1.md`
  beside the extension. The SDK requires a suitable license when you ship a build
  to other people. TensorViz's MIT license does not relicense tldraw or grant a
  key to downstream builders. A local build does not need a production key.
- `MCP_PARSER_NOTICES.md` retains the agent-configuration parsers' notices.

Python and its dependencies are downloaded into the selected or managed runtime
separately. Their distributions retain their own licenses; they are not included
as Python binaries in this VSIX.

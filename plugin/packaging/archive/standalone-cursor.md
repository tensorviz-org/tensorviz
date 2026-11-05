# Cursor local installation

Verify the archive against its sibling `.sha256` file, then extract it into a
directory you will keep:

```sh
tar -xzf tensorviz-cursor-<version>.tar.gz -C /absolute/path/to/install
```

Merge this server into `~/.cursor/mcp.json` (or Cursor's MCP settings):

```json
{
  "mcpServers": {
    "tensorviz": {
      "command": "node",
      "args": ["/absolute/path/to/install/tensorviz-cursor-<version>/plugins/tensorviz/scripts/launch.cjs"]
    }
  }
}
```

Replace the path and version with your extracted location. The packaged
`plugins/tensorviz/mcp.json` uses `${CURSOR_PLUGIN_ROOT}/scripts/launch.cjs`;
`${CURSOR_PLUGIN_ROOT}` means the extracted `plugins/tensorviz` directory.
Use its absolute path for a manual MCP entry.

Node.js 22+ must be on Cursor's PATH. Restart Cursor and check that TensorViz
tools appear. Keep one TensorViz host enabled. On first capture, grant the
requested permission and choose a model Python and example inputs.

This Cursor archive cannot use the Codex guided installer.
`cursor-agent plugin marketplace add <path>` treats local paths as git remotes;
use the MCP configuration above for this local archive.

See the [plugin guide](https://github.com/tensorviz-org/tensorviz/blob/main/docs/reference/agent-plugin.md)
for usage and permissions.

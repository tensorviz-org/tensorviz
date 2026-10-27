### Connect your coding agent

Open the chat icon in the bottom-right of Canvas to ask about the model or request
a change. Choose **Codex** or **Claude Code**, installed and signed in on the
workspace host. The model and selected layers are attached automatically.
Minimize keeps the draft and conversation; **Stop** cancels active work. Source
changes still open a native diff for **Apply/Discard**. No provider key or global
MCP configuration is needed for Canvas chat. With Remote SSH, install and sign in
to the agent on the remote host. The `tensorviz.agent.codexPath` and
`tensorviz.agent.claudePath` settings can select an executable.

You can also connect from an existing agent chat. MCP is how your existing
coding agent reads the graph: it supplies its own language model, and tensorViz
supplies graph facts and tools. **Connect coding agent (MCP)** starts a standard
Streamable HTTP endpoint for the local VS Code installation. The connection panel
can configure a supported agent after approval or provide its URL for another
compatible client. A status-bar item shows Off, Ready or Connected.

The **tensorViz plugin** installs separately in your coding agent,
once for your user profile. It works across model projects with VS Code closed.
Use that plugin for graphs in Codex or Claude Code; it includes its own MCP
connection. The former VS Code companion plugin is retired.

To use the running VS Code Canvas from an agent instead, choose your agent and
approve **Add TensorViz**. No agent CLI or plugin is required for this connection.
Keep one TensorViz host enabled per agent; do not add the VS Code connection when
you use the plugin. Remote SSH uses explicit forwarding and manual MCP setup.

Not sure which configuration values or input shapes a model needs on first open?
Ask the connected agent to prepare its setup and capture the graph. It can inspect
the model's source, explain missing values and save the result in `tensorviz.toml`.
An agent is optional; the Inputs form enables the same measured graph.

- Ask the agent to open a model with `open_graph`. It continues through missing
  setup until capture succeeds or it reports a specific blocker.
- `get_execution_environment` identifies the Python used for capture. Dependency
  installation targets that environment after native approval; building a package
  from source requires separate approval before its build code runs.
- The agent can inspect the open model, read and focus a Canvas selection, and
  start, poll or cancel validation.
- It proposes edits through the same native diff you review before anything is
  applied.

#### Turn MCP on and off

- **Connect coding agent (MCP)** turns it on and shows the endpoint URL.
- **Disconnect coding agents (MCP)** turns it off and revokes all sessions.
- The opt-in persists across restarts; another VS Code window can take over the
  endpoint when its owner closes. Disconnect disables reconnection across windows.
- The endpoint has no bearer credential. It binds loopback and checks `Host` and
  `Origin`, so any local process can reach it — disconnect when you are done.

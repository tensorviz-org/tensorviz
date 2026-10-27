# Security

## Reporting a vulnerability

Open a private vulnerability report on
[tensorviz-org/tensorviz](https://github.com/tensorviz-org/tensorviz/security/advisories/new).
Do not file a public issue for an unfixed security bug.

Include what you ran, the version or commit, and whether the issue is in the
extension, the plugin host, or the local Python backend.

## Scope

tensorViz executes the Python you open. Treat a workspace as trusted code: Open
as Graph and MCP `validate_model` import and run that code in the interpreter you
select. The local MCP HTTP server binds to loopback and checks Host and Origin.
It is not an internet service. The Python service defaults to `127.0.0.1` and
refuses a non-loopback bind unless `TENSORVIZ_LOCAL_TOKEN` is set.
`AUTH_DISABLED` does not allow a public bind. `backend/auth.py` does not accept
bearer tokens. Firebase Admin is not a backend dependency.

Website sign-in stores a refresh token in the editor secret store. The Firebase
web API key in the extension is a public client identifier for tensorviz.dev, not
a credential that grants admin access.

Do not send launch tokens, `.tensorviz/lock`, secret-storage dumps, or
`tensorviz.toml` files that contain private paths when you report a bug. Those
files can identify your machine and project.

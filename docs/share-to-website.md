# Add graphs to tensorViz

The website application is not in this repository. The extension and plugin clients are.
`extension/src/websiteAccount.ts` holds the public Firebase web configuration
(`apiKey`, `projectId`, and the identity toolkit hosts); the plugin keeps the same
values in `plugin/src/website.ts`. Those values identify
the tensorviz.dev web app. They are not admin secrets. Access control is the
website's rules and its upload API. Firebase Admin is not a backend dependency,
and `backend/auth.py` does not accept bearer tokens. The extension and plugin ship only
public web configuration. Do not add a Firebase Admin private key.
Sign-in and upload are shared by both clients in
`packages/graph-host/src/websiteAccount.ts` and `websiteShare.ts`.

**Publish**, in the VS Code Canvas toolbar, puts the open graph on [tensorviz.dev](https://tensorviz.dev/explore/) in one
click. The coding-agent plugin's **Model graph** panel and browser graph have the same
**Publish** button. The graph arrives private. Its **Share** dialog is where you
choose who can see it and copy its link:

| Access | Who can open it |
| --- | --- |
| Only you | You, signed in. The starting point. |
| tensorViz members | Anyone signed in to tensorViz; it is listed for them and they can reply. |
| Anyone with the link | Anyone with the address, signed in or not. It is not listed. |

The graph page also lets you rename the graph, describe it, start a discussion about
its layers, or delete it. VS Code opens the Share dialog in your browser and offers
**Copy link** when the upload finishes. Open
Canvas in Browser, which cannot sign in, keeps **Export graph** for a
[graph file](graph-files.md).
While the VS Code host connects, the toolbar may briefly show **Export graph**;
it switches to **Publish** as soon as the host is ready.

## Signing in

The first time, **Publish** opens `tensorviz.dev/explore/connect/` in your
browser. Pick a Google account, then let the browser open VS Code; the first time, VS
Code may also ask to let tensorViz open the link. The graph is then added by itself.
**tensorViz: Sign in to tensorViz** does the same from the command palette, and
switches accounts.

Sign-in waits up to ten minutes for the browser to return to VS Code. If it times
out, or your session has expired, the error dialog offers **Retry**. Retrying
**Publish** opens sign-in again and then uploads the same graph. A rejected upload
token is renewed once automatically before asking you to sign in again.

The website sends VS Code a Google ID token, valid for an hour, together with the
random state VS Code started the sign-in with; it sends it only to this extension or
to a loopback address on your computer. The extension exchanges the token with Firebase
Authentication for its own session and keeps only the refresh token, in VS Code's
Secret Storage, shared by your VS Code windows. **tensorViz: Sign out of tensorViz**
deletes it. No token is written to the log.

## Publish from the coding-agent plugin

**Publish** sits in the toolbar of the **Model graph** panel and of the browser graph
the plugin opens. It sends the same graph file as the editor, through the plugin's
local connection, to the plugin host on your computer, which signs in and uploads it.
Publish is a button, not an agent tool. The panel sends it through its private transport
tool, which MCP Apps clients hide from the agent, and never through the read-only one, so a
client that confirms non-read-only calls asks you first.

The first time, a **Sign in to publish** card appears beside the button. Choose
**Continue in browser** to open `tensorviz.dev/explore/connect/` and pick a Google
account. The website returns the sign-in to a one-time address on this computer,
`http://127.0.0.1:<port>/tensorviz/auth`, which only the plugin host answering this
Publish listens on. The browser then shows **Signed in to tensorViz**, and publishing
continues in the graph. The page you return to carries no token. Sign-in waits up to ten
minutes; **Cancel** stops it. Nothing opens a browser until you click.

When the upload finishes, choose **Choose who can see it** to open the graph's Share
dialog, or **Copy link**. If a sign-in times out or expires, **Try again** signs in again
and uploads the same graph. In the panel, your client may ask before it opens the website.

The plugin keeps only the refresh token, in a private file (folder `0700`, file `0600`)
in the plugin's data folder: `website/session.json` under
`~/Library/Application Support/TensorViz/standalone` on macOS, `%LOCALAPPDATA%\TensorViz\standalone`
on Windows, or `$XDG_DATA_HOME/tensorviz/standalone` (default `~/.local/share/...`)
on Linux, or `TENSORVIZ_DATA_DIR` when set. Every chat's plugin host shares it. To
sign out, run `node scripts/launch.cjs --sign-out` from the installed plugin directory.

A graph opened with a capture also includes its portable capture, as in the editor. The
plugin's local service detaches it from your source names without running the model.

## What is sent

Nothing until you click **Publish**. Then the extension or plugin host posts the graph file
Export graph would save ([what it holds](graph-files.md)), gzipped, to the website's
upload API, `tensorviz.dev/api/graphs`, with your sign-in. The website checks the file again,
stores it, draws its card preview, and records your Google account's name and photo as its
author.

## Limits

- Graph files up to 10 MB with 1 to 5,000 blocks, the website's limits.
- One post every 15 seconds; a quicker second one says how long to wait.
- Desktop VS Code and VS Code-family editors, and the coding-agent plugin's panel and
  browser graph on the computer running the plugin. Open Canvas in Browser and
  browser-based VS Code (Codespaces, vscode.dev) cannot add graphs yet.

The website application is not in this repository. Sign-in and upload above are
the extension and plugin clients only.

# Add graphs to tensorViz

The website application is not in this repository. The extension client is.
`extension/src/websiteAccount.ts` holds the public Firebase web configuration
(`apiKey`, `projectId`, and the identity toolkit hosts). Those values identify
the tensorviz.dev web app. They are not admin secrets. Access control is the
website's rules and its upload API. Firebase Admin is not a backend dependency,
and `backend/auth.py` does not accept bearer tokens. The canvas `firebase` npm
package stays for hosted pages. The IDE entry `canvas/src/ide/main.tsx` does not
import it. The extension still ships this public web config. Do not add a
Firebase Admin private key.

**Add to tensorViz**, in the VS Code Canvas model menu, puts the open graph on [tensorviz.dev](https://tensorviz.dev/explore/) in one
click. The graph arrives private. Your browser opens its **Share** dialog, where you
choose who can see it and copy its link:

| Access | Who can open it |
| --- | --- |
| Only you | You, signed in. The starting point. |
| tensorViz members | Anyone signed in to tensorViz; it is listed for them and they can reply. |
| Anyone with the link | Anyone with the address, signed in or not. It is not listed. |

The graph page also lets you rename the graph, describe it, start a discussion about
its layers, or delete it. VS Code offers **Copy link** when the upload finishes. Open
Canvas in Browser, which cannot sign in, keeps **Export graph** for a
[graph file](graph-files.md).

## Signing in

The first time, **Add to tensorViz** opens `tensorviz.dev/explore/connect/` in your
browser. Pick a Google account, then let the browser open VS Code; the first time, VS
Code may also ask to let tensorViz open the link. The graph is then added by itself.
**tensorViz: Sign in to tensorViz** does the same from the command palette, and
switches accounts.

The website sends VS Code a Google ID token, valid for an hour, together with the
random state VS Code started the sign-in with; it sends it only to this extension or
to a loopback address on your computer. The extension exchanges the token with Firebase
Authentication for its own session and keeps only the refresh token, in VS Code's
Secret Storage, shared by your VS Code windows. **tensorViz: Sign out of tensorViz**
deletes it. No token is written to the log.

## What is sent

Nothing until you click **Add to tensorViz**. Then the extension posts the graph file
Export graph would save ([what it holds](graph-files.md)), gzipped, to the website's
upload API, `tensorviz.dev/api/graphs`, with your sign-in. The coding-agent plugin
does not upload graphs. The website checks the file again, stores it, draws its card
preview, and records your Google account's name and photo as its author.

## Limits

- Graph files up to 10 MB with 1 to 5,000 blocks, the website's limits.
- One post every 15 seconds; a quicker second one says how long to wait.
- Desktop VS Code and VS Code-family editors only. Open Canvas in Browser, the plugin
  panel, and browser-based VS Code (Codespaces, vscode.dev) cannot add graphs yet.

The website application is not in this repository. Sign-in and upload above are
the extension client only.

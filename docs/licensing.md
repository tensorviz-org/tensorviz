# Licensing

TensorViz's own code is [MIT](../LICENSE). Copyright (c) 2026 TensorViz
contributors. MIT covers this repository's source. It does not relicense
dependencies.

## tldraw

The canvas and the plugin graph viewer use tldraw 5.0.1. That SDK is under
the tldraw license, copied in [runtime/licenses/tldraw-5.0.1.md](../runtime/licenses/tldraw-5.0.1.md)
from the upstream `v5.0.1` tag. The license allows development use. Local
builds, pull requests, and the dev `package` script do not need a tldraw
production key. Release packaging fails when the key is unset and does not
print it. That is the extension script `package:release` (`verify-package.mjs --release`)
and the plugin packager `node scripts/package.mjs --release`.
`npm run package:source-development --prefix plugin` stays unverified.
A VSIX or plugin you ship to other people needs a tldraw license key that you obtain yourself.

Set the key only in the environment when you build:

```sh
export REACT_APP_TLDRAW_LICENSE_KEY="your-key"
```

The canvas build and the plugin UI build read that variable. An empty value
on a dev `package` produces an unlicensed evaluation build, and tldraw may show
its own watermark. `package:release` refuses that build.
Never commit a key, and never put one in `.env`, `.env.local`, or a workflow.
Those files are gitignored. A key in a pull request or issue will be treated as
compromised.

This repository does not distribute a tldraw production key. Builders who
publish tensorViz need their own agreement with tldraw. The license terms are at
[tldraw.dev/community/license](https://tldraw.dev/community/license).

## Generated dependency notices

`tooling/packaging/bundle-notices.mjs` writes the license text of npm packages
that esbuild actually bundles. `node --test tooling/test/bundle-notices.test.mjs`
checks that step. Keep those notices next to the built host. The committed
summary is [NOTICE](../NOTICE) and [extension/THIRD_PARTY_NOTICES.md](../extension/THIRD_PARTY_NOTICES.md).

## Website client configuration

Sign-in uses the public Firebase web configuration in
`extension/src/websiteAccount.ts` (`apiKey`, `projectId`). Those identifiers are
meant to be in a client. They are not admin credentials. Firebase Admin is not
a backend dependency. `backend/auth.py` does not accept bearer tokens: `AUTH_DISABLED`
selects the dev principal, then `TENSORVIZ_LOCAL_TOKEN`, and any other credential
is 401. The canvas `firebase` npm package stays because hosted pages import it.
`canvas/src/ide/main.tsx` does not. Do not add Firebase Admin private keys,
service accounts, or analytics keys to this tree.

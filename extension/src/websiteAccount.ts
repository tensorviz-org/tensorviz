/** The member's tensorViz website account, which Publish sends graphs to.
 *
 * Sign-in goes through the website. The extension opens tensorviz.dev/explore/connect/
 * with a random state and a callback to its own URI handler; the member picks a Google
 * account there, and the page sends back that state with a Google ID token, valid for an
 * hour. The extension exchanges the token with Firebase Authentication for its own
 * session and keeps only the refresh token, in VS Code's secret storage, which every
 * window shares. The account itself is shared with the plugin
 * (@tensorviz/graph-host/websiteAccount); this module keeps the website's public
 * configuration and binds it. Nothing here imports `vscode`; websiteCommands.ts wires it
 * up. See docs/share-to-website.md.
 */
import {
  WebsiteAccount as SharedWebsiteAccount, websiteConfig as sharedWebsiteConfig,
  type AccountHost, type WebsiteConfig,
} from '@tensorviz/graph-host/websiteAccount';

export {
  connectUrl, EMULATOR_VARIABLE, firebaseCode, SITE_VARIABLE, tokenClaims, WebsiteError, websiteRequest,
  type AccountHost, type Cancellation, type WebsiteConfig, type WebsiteErrorCode, type WebsiteFetch,
  type WebsiteMember, type WebsiteSession,
} from '@tensorviz/graph-host/websiteAccount';

/** The website's own Firebase web app (src/explorer/community-config.js in its repository).
 *  The plugin keeps the same values in plugin/src/website.ts. */
export const WEBSITE: WebsiteConfig = {
  site: 'https://tensorviz.dev',
  apiKey: 'AIzaSyBB-uxMsrJ4lzA_ywP0JdZMxM0vsPNoxxU',
  projectId: 'project-fe5816f4-14e7-42f8-aa0',
  identity: 'https://identitytoolkit.googleapis.com',
  secureToken: 'https://securetoken.googleapis.com',
};

/** The production website or, for development, a loopback website dev server. */
export function websiteConfig(env: Record<string, string | undefined> = process.env): WebsiteConfig {
  return sharedWebsiteConfig(WEBSITE, env);
}

/** Where the website sends the member back: this extension's `/auth` URI in the running
 *  editor. Deliberately not vscode.env.asExternalUri: the `?windowId=` it adds would put a
 *  `?` inside the connect link's query, which vscode.env.openExternal mangles (it escapes
 *  the `?` in Uri.toString(true), then encodeURI escapes that again), so the page refused
 *  the link. VS Code hands the reply to the last active window instead. */
export function editorCallback(uriScheme: string, extensionId: string): string {
  return `${uriScheme}://${extensionId}/auth`;
}

/** The shared account, signed in through tensorviz.dev. */
export class WebsiteAccount extends SharedWebsiteAccount {
  constructor(host: AccountHost) {
    super(host, WEBSITE);
  }
}

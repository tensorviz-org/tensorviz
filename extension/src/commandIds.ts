/** Command ids shared between the module that registers a command and the modules
 *  that reference it.
 *
 * Its own file, and deliberately importing nothing: `views.ts` used to reach into
 * `extension.ts` for this one string, which dragged the entire activation module —
 * and `codeActions.ts`'s static initialisers with it — into every unit test that
 * merely wanted a tree view.
 */

/** Draw a class that has already been named. No surface asks "which class?" to
 *  get in any more; that is `tensorviz.openProject` (step 2.5). */
export const OPEN_AS_GRAPH = 'tensorviz.openAsGraph';

/** The project canvas: the workspace folder, drawn, naming no file and no class. */
export const OPEN_PROJECT = 'tensorviz.openProject';

/** Compatibility alias for Open Canvas in Browser: uses the selected model or
 *  project, never the incidental editor or a legacy folder argument. */
export const OPEN_PROJECT_IN_BROWSER = 'tensorviz.openProjectInBrowser';

/** Sign in to and out of the member's tensorViz website account (websiteCommands.ts). */
export const SIGN_IN_TO_WEBSITE = 'tensorviz.signInToWebsite';
export const SIGN_OUT_OF_WEBSITE = 'tensorviz.signOutOfWebsite';

/** Add to tensorViz: the Canvas bridge runs it with the graph file. Not contributed,
 *  so it never appears in the palette. */
export const SHARE_GRAPH_FILE = 'tensorviz.shareGraphFile';

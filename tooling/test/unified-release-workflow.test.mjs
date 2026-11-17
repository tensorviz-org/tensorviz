import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const read=f=>readFileSync(new URL(`../../${f}`,import.meta.url),'utf8');
test('canonical candidates retain all native and pinned installed migration gates without publisher credentials',()=>{
 const workflow=read('.github/workflows/release-unified.yml');
 for(const platform of ['darwin-arm64','darwin-x64','linux-arm64','linux-x64','win32-arm64','win32-x64'])assert.match(workflow,new RegExp(`platform: ${platform}`));
 assert.match(workflow,/TENSORVIZ_MIGRATION_TARGET_VERSION: '1\.0\.0'/);
 assert.match(workflow,/ref: 8b0b539953c5abe693803c15854d953a00bcd5fe/);
 assert.match(workflow,/tensorviz-org\/tensorviz\/releases\/download/);
 assert.match(workflow,/native-version-transition\.cjs/);
 assert.match(workflow,/needs: \[configuration, assemble, current_migration, extension_docs\]/);
 assert.match(workflow,/actions\/attest-build-provenance@[a-f0-9]{40}/);
 assert.doesNotMatch(workflow,/VSCE_PAT|OVSX_PAT|PLUGIN_PUBLISH_TOKEN|contents: write|continue-on-error/);
});
test('publication reuses the already trusted docs OIDC workflow and verifies before secrets',()=>{
 const docs=read('.github/workflows/docs.yml'),publish=docs.split('\n  publish_unified:\n')[1];assert.ok(publish);
 assert.ok(publish.indexOf('unified-release-preflight.mjs')<publish.indexOf('ref: ${{ steps.release.outputs.source }}'));
 assert.match(publish,/gh run watch "\$CANDIDATE_RUN_ID" --repo tensorviz-org\/tensorviz --exit-status --interval 10/);
 assert.ok(publish.indexOf('gh run watch')<publish.indexOf('unified-release-preflight.mjs'));
 assert.ok(publish.indexOf('admitUnifiedBundle')<publish.indexOf('uses: google-github-actions/auth@v3'));
 assert.ok(publish.indexOf('admitUnifiedBundle')<publish.indexOf('VSCE_PAT:'));
 assert.match(publish,/run-id: \$\{\{ steps.release.outputs.candidate_run_id \}\}/);
 assert.match(publish,/publish-unified-channels\.mjs plugin\/out\/unified-release \. github-draft/);
 assert.doesNotMatch(publish,/PLUGIN_PUBLISH_TOKEN|publish-native-release|sync-github-release|--clobber|continue-on-error/);
});
test('publication continuation is opt-in and dispatches only after the complete bundle succeeds',()=>{
 const workflow=read('.github/workflows/release-unified.yml');
 assert.match(workflow,/publish_after_checks:\n\s+description:[^\n]+\n\s+type: boolean\n\s+default: false/);
 const dispatch=workflow.split('\n  dispatch_publication:\n')[1];assert.ok(dispatch);
 assert.match(dispatch,/needs: \[configuration, bundle\]/);
 assert.match(dispatch,/if: inputs.publish_after_checks && github.repository == 'tensorviz-org\/tensorviz' && github.ref == 'refs\/heads\/main'/);
 assert.match(dispatch,/actions: write/);
 assert.match(dispatch,/gh workflow run docs.yml --repo tensorviz-org\/tensorviz --ref main/);
 assert.match(dispatch,/-f release_source_sha="\$RELEASE_SOURCE"/);
 assert.match(dispatch,/-f candidate_run_id="\$CANDIDATE_RUN_ID"/);
 assert.doesNotMatch(dispatch,/secrets\.|contents: write|id-token: write/);
 assert.doesNotMatch(workflow,/workflow_run:|pull_request:|push:/);
});

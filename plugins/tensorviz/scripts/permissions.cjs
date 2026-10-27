#!/usr/bin/env node
// Explicit user command. Never run by MCP startup or the workflow skill.
const { PluginPermissions, permissionSummary } = require('../runtime/permissions.cjs');

async function confirm(summary = permissionSummary) {
  if (!process.stdin.isTTY || !process.stdout.isTTY) return false;
  const readline = require('node:readline/promises').createInterface({ input: process.stdin, output: process.stdout });
  try {
    const answer = await readline.question(`${summary}\n\nAllow TensorViz? [y/N] `);
    return /^(y|yes)$/i.test(answer.trim());
  } finally { readline.close(); }
}

async function allow(permissions = new PluginPermissions()) {
  if (await permissions.status() === 'granted') return true;
  if (!process.stdin.isTTY || !process.stdout.isTTY) return false;
  const approved = await confirm();
  await permissions.save(approved ? 'granted' : 'denied');
  return approved;
}

async function main() {
  const permissions = new PluginPermissions();
  const action = process.argv[2] ?? '--status';
  if (process.argv.length > 3 || !['--status', '--allow', '--revoke'].includes(action)) throw new Error('Usage: node permissions.cjs [--status|--allow|--revoke]');
  if (action === '--revoke') await permissions.save('denied');
  if (action === '--allow' && !await allow(permissions)) {
    console.log('Permission was not granted. Run this command yourself in a terminal to enable TensorViz.');
    process.exitCode = 1;
  }
  console.log(`TensorViz permission: ${await permissions.status()}`);
}

module.exports = { allow, confirm };
if (require.main === module) main().catch(error => { console.error(error.message); process.exitCode = 1; });

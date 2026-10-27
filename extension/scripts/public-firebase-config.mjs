// Open VSX supports same-line annotations for verified false-positive findings.
// Only the website's explicitly public Firebase client identifier is annotated.
export function annotatePublicFirebaseConfig(bundle, source) {
  const key = source.match(/^  apiKey: '(AIza[\w-]+)',$/m)?.[1];
  if (!key) throw new Error('Expected the website public Firebase identifier.');
  const lines = bundle.split('\n');
  const matches = lines.filter(line => line.includes(key));
  const property = new RegExp(`^\\s*apiKey: ${JSON.stringify(key)},?\\s*$`);
  if (matches.length !== 1 || !property.test(matches[0])) {
    throw new Error('Expected exactly one standalone public Firebase property in the bundle.');
  }
  return lines.map(line => line === matches[0]
    ? `${line} // secret-detector:ignore -- public Firebase app identifier; authorization uses user tokens and server rules`
    : line).join('\n');
}

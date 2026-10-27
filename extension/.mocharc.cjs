// Unit tests that create temporary git repositories exceed Mocha's 2000ms
// default when git init is slow. Suites that need longer set this.timeout.
module.exports = { timeout: 20000 };

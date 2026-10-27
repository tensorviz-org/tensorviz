"use strict";
exports.archive = archive;
exports.gz = gz;
/** A minimal, correct tar writer for building runtime-archive fixtures in-test. No binary
 *  fixtures checked in, no network: every test that needs a tarball builds it byte for byte.
 *
 *  Shared between runtimeAcquire.test.ts (which pins the extractor's behaviour against
 *  hostile archives) and runtime.test.ts (which builds a scripted interpreter the
 *  provisioning flow can actually "run"). */
const zlib = require("node:zlib");
function header(member, size) {
    const block = Buffer.alloc(512);
    const put = (text, offset, length = text.length) => {
        block.write(text.slice(0, length), offset, 'latin1');
    };
    const octal = (value, offset, width) => put(value.toString(8).padStart(width - 1, '0'), offset, width - 1);
    put(member.name, 0, 100);
    octal(member.mode ?? 0o644, 100, 8);
    octal(0, 108, 8); // uid
    octal(0, 116, 8); // gid
    octal(size, 124, 12);
    octal(0, 136, 12); // mtime
    block.write('        ', 148, 'latin1'); // checksum field as spaces while summing
    put(member.type ?? '0', 156, 1);
    put(member.linkname ?? '', 157, 100);
    put('ustar', 257, 6);
    put('00', 263, 2);
    if (member.prefix) {
        put(member.prefix, 345, 155);
    }
    let sum = 0;
    for (const byte of block) {
        sum += byte;
    }
    block.write(sum.toString(8).padStart(6, '0') + '\0 ', 148, 'latin1');
    return block;
}
function entry(member) {
    const data = member.data ?? Buffer.alloc(0);
    const parts = [header(member, data.length), data];
    const padding = (512 - (data.length % 512)) % 512;
    if (padding) {
        parts.push(Buffer.alloc(padding));
    }
    return Buffer.concat(parts);
}
function archive(members) {
    return Buffer.concat([...members.map(entry), Buffer.alloc(1024)]);
}
function gz(plain) {
    return zlib.gzipSync(plain);
}

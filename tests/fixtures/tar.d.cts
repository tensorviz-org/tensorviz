/// <reference types="node" />
export type Member = {
  name: string;
  type?: string;
  mode?: number;
  linkname?: string;
  data?: Buffer;
  prefix?: string;
};
export function archive(members: Member[]): Buffer;
export function gz(plain: Buffer): Buffer;

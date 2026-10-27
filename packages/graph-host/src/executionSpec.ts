import { z } from 'zod';
import { tensorInputSchema } from './tensorInputs';

/** Portable workspace-relative paths; the host and worker also resolve symlinks. */
export const executionPathSchema = z.string().min(1).max(512).refine(value =>
  !value.startsWith('/') && !/[\\:\x00]/.test(value) && value.split('/').every(part =>
    part !== '' && part !== '.' && part !== '..' && !part.startsWith('.env')),
  'Use a workspace-relative path without traversal or environment files.');
const literals = z.record(z.string().min(1).max(128), z.json()).refine(value => {
  const withinDepth = (item: unknown, depth: number): boolean => depth <= 8 &&
    (item === null || typeof item !== 'object' || Object.values(item).every(child => withinDepth(child, depth + 1)));
  return JSON.stringify(value).length <= 8192 && withinDepth(value, 0);
}, 'Configuration must be bounded JSON literals (8 KiB, depth 8).');
export const executionCallableSchema = z.object({
  file: executionPathSchema.refine(value => value.endsWith('.py'), 'Choose a Python file.'),
  callable: z.string().regex(/^[A-Za-z_][A-Za-z0-9_]*$/).max(128),
  kwargs: literals.default({}),
}).strict();
/** A saved tensorviz.toml binding, carried through verbatim. Python's
 *  services.validation.contracts is the authority on its grammar — it can express
 *  `full` with any value, `randint` bounds, literal `values` and per-tensor seeds, none
 *  of which the three-fill vocabulary below can say. Translating it here would mean
 *  substituting an input the user never wrote. */
const profileBinding = z.object({
  args: z.array(z.json()).max(64).default([]),
  kwargs: z.record(z.string().regex(/^[A-Za-z_][A-Za-z0-9_]*$/).max(128), z.json()).default({}),
}).strict();
const inputs = z.union([
  z.object({ factory: executionCallableSchema }).strict(),
  z.object({ binding: profileBinding }).strict(),
  z.object({ fixture: executionPathSchema }).strict(),
  z.object({ args: z.array(tensorInputSchema).max(32).default([]),
    kwargs: z.record(z.string().regex(/^[A-Za-z_][A-Za-z0-9_]*$/).max(128), tensorInputSchema).default({}),
  }).strict().superRefine((value, ctx) => {
    const all = [...value.args, ...Object.values(value.kwargs)];
    if (all.length > 32 || all.reduce((sum, item) => sum + item.shape.reduce((n, d) => n * d, 1), 0) > 4_194_304) {
      ctx.addIssue({ code: 'custom', message: 'Use at most 32 tensors and 4,194,304 total elements.' });
    }
    if (all.some(item => item.fill === 'randn' && !['float32', 'float64', 'float16', 'bfloat16'].includes(item.dtype))) {
      ctx.addIssue({ code: 'custom', message: 'randn requires a floating-point dtype.' });
    }
  }),
]);
export const executionSpecSchema = z.object({
  framework: z.literal('jax').optional(),
  inputs,
  constructorArgs: literals.optional(),
  /** The profile's own `init`, when one was saved. Mutually exclusive with
   *  constructorArgs, and always set for a profile run so a retired
   *  `# tensorviz: init=` comment cannot reach the constructor behind its back. */
  constructorBinding: profileBinding.optional(),
  setup: executionCallableSchema.optional(),
  checkpoint: z.object({ file: executionPathSchema, key: z.string().min(1).max(128).optional(),
    strict: z.boolean().default(true) }).strict().optional(),
  device: z.string().regex(/^(cpu|mps|cuda(?::[0-9]{1,3})?)$/).default('cpu'),
  mode: z.enum(['eval', 'train', 'preserve']).default('eval'),
  seed: z.number().int().min(0).max(Number.MAX_SAFE_INTEGER).nullable().default(0),
  deterministic: z.boolean().default(true),
  gradEnabled: z.boolean().default(false),
  cwd: z.union([z.literal('.'), executionPathSchema]).default('.'),
}).strict().refine(value => !(value.setup && (value.constructorArgs !== undefined || value.constructorBinding !== undefined)),
  'Use either a model setup factory or constructor arguments.')
  .refine(value => !(value.constructorArgs !== undefined && value.constructorBinding !== undefined),
  'Use either a saved profile init binding or constructor arguments.');
export type ExecutionSpec = z.infer<typeof executionSpecSchema>;

/** Profiling and training-step jobs are not part of this build. */
export const executionJobSchema = z.object({
  kind: z.enum(['profile', 'training']),
}).strict().superRefine((_value, ctx) => {
  ctx.addIssue({ code: 'custom', message: 'Profiling and training-step capture are not available.' });
});
export type ExecutionJob = { kind: 'profile' | 'training' };
export const trainingExecutionSpecSchema = executionSpecSchema.safeExtend({
  mode: z.enum(['eval', 'train', 'preserve']).default('train'),
});

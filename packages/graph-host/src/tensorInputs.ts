import { z } from 'zod';

/** Positional CPU tensors only; no expressions, file loading or arbitrary allocation. */
export const tensorInputSchema = z.object({
  shape: z.array(z.number().int().min(1).max(1_000_000)).max(8),
  dtype: z.enum(['float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'bool']),
  fill: z.enum(['zeros', 'ones', 'randn']),
}).strict();
export const tensorInputsSchema = z.array(tensorInputSchema).min(1).max(8).superRefine((inputs, ctx) => {
  if (inputs.reduce((sum, input) => sum + input.shape.reduce((n, dim) => n * dim, 1), 0) > 4_194_304) {
    ctx.addIssue({ code: 'custom', message: 'Inputs may contain at most 4,194,304 elements in total.' });
  }
  if (inputs.some(input => input.fill === 'randn' && !['float32', 'float64', 'float16', 'bfloat16'].includes(input.dtype))) {
    ctx.addIssue({ code: 'custom', message: 'randn requires a floating-point dtype. Use zeros or ones for integer and boolean tensors.' });
  }
});
export type TensorInputs = z.infer<typeof tensorInputsSchema>;

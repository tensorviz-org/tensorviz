/**
 * `tensorviz.traceTier = auto`: draw large models with fake tensors, everything else with
 * real tensors, without building the model twice to find out which it is.
 *
 * The first run of a model is always real (exactly as `real`). Its capture reports the
 * model's parameter count; a later run of the same model uses the fake tier when that count
 * is at least `AUTO_FAKE_MIN_PARAMETERS` (150M). A model whose fake run stopped on a fake-tensor
 * limitation (it needs values: `.item()`, `nonzero`, a lazy module) is remembered and runs
 * real from then on, so it never pays for two runs per trace.
 *
 * Memory is per window and per model (workspace, file, class); a window reload starts over.
 */
import type { TraceOutcome } from './traceWorkerCore';

export type TraceTierSetting = 'real' | 'fake' | 'auto';

/** 150M parameters (~600 MB in float32). Measured end to end (macOS arm64, torch 2.7.1):
 *  fake was slower at 87M (vit_b16 1.85 vs 1.51 s) and 110M (bert_base 1.85 vs 1.67 s) and
 *  faster from 200M (llama_gqa_rope 1.78 vs 2.15 s, t5_base 2.08 vs 2.44 s; llama_1b 1.84 vs
 *  5.35 s). Fake tensors pay a fixed ~0.6 s (`torch._dynamo` import) that only weight
 *  construction and real compute above this size outweigh. */
export const AUTO_FAKE_MIN_PARAMETERS = 150_000_000;

interface Remembered { parameters?: number; fakeLimited?: boolean }

/** The root invocation's parameter count of a canonical capture, or undefined. */
export function capturedParameterCount(outcome: TraceOutcome): number | undefined {
  const capture = (outcome.envelope?.result as { diagnostics?: { semantic_capture?: {
    rootInvocationId?: unknown; invocations?: Array<{ id?: unknown; parameterCount?: unknown }>;
  } } } | null | undefined)?.diagnostics?.semantic_capture;
  if (!capture || !Array.isArray(capture.invocations)) return undefined;
  const root = capture.invocations.find(call => call && call.id === capture.rootInvocationId);
  const count = root?.parameterCount;
  return typeof count === 'number' && Number.isSafeInteger(count) && count >= 0 ? count : undefined;
}

export class AutoTraceTier {
  private readonly models = new Map<string, Remembered>();
  constructor(private readonly minParameters = AUTO_FAKE_MIN_PARAMETERS, private readonly maxModels = 256) {}

  private static key(model: { workspace: string; file: string; cls: string }): string {
    return JSON.stringify([model.workspace, model.file, model.cls]);
  }

  /** The tier to run with. `real` and `fake` are returned unchanged. */
  tier(setting: TraceTierSetting, model: { workspace: string; file: string; cls: string }): 'real' | 'fake' {
    if (setting !== 'auto') return setting;
    const known = this.models.get(AutoTraceTier.key(model));
    return known && !known.fakeLimited && known.parameters !== undefined && known.parameters >= this.minParameters
      ? 'fake' : 'real';
  }

  /** Learn from a finished run that was started with `tier`. Failed runs teach only the
   *  fake-tensor limitation; a run without a capture changes nothing. */
  record(model: { workspace: string; file: string; cls: string }, tier: 'real' | 'fake', outcome: TraceOutcome): void {
    const key = AutoTraceTier.key(model);
    const entry: Remembered = { ...this.models.get(key) };
    if (tier === 'fake' && outcome.fakeTensorFallback) entry.fakeLimited = true;
    const parameters = outcome.ok ? capturedParameterCount(outcome) : undefined;
    if (parameters !== undefined) entry.parameters = parameters;
    if (entry.parameters === undefined && !entry.fakeLimited) return;
    this.models.delete(key);                      // re-insert: most recent last
    this.models.set(key, entry);
    while (this.models.size > this.maxModels) this.models.delete(this.models.keys().next().value!);
  }

  clear(): void { this.models.clear(); }
}

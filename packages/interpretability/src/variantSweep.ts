import type { Intervention, Model, ModelAdapter, WeightProvider } from "@aperture/model-ir";
import { yieldToBrowser } from "./yield.js";

/**
 * Scores each of `variants` (variants[i] applied only to variant i — an
 * empty array is a valid "no intervention" baseline variant) against
 * `predictIndex`, returning one logits row per variant. Prefers the
 * adapter's batched `runAttributionSweep` (a handful of real forward
 * passes total) when it has one; otherwise falls back to one
 * `runInference()` call per variant — correct either way, just far slower
 * once each call is a real computation (a backend model) rather than free
 * in-browser math (the tiny debug models this fallback was written for).
 */
export async function getVariantLogitRows(
  model: Model,
  weightProvider: WeightProvider,
  adapter: ModelAdapter,
  tokenIds: number[],
  predictIndex: number,
  variants: Intervention[][]
): Promise<Float64Array[]> {
  if (adapter.runAttributionSweep) {
    return adapter.runAttributionSweep(model, weightProvider, tokenIds, predictIndex, variants);
  }

  if (!adapter.runInference) throw new Error(`${adapter.displayName} does not support running inference`);
  const rows: Float64Array[] = [];
  for (const variant of variants) {
    await yieldToBrowser();
    const capture = await adapter.runInference(model, weightProvider, tokenIds, variant);
    const vocab = capture.logits.shape[1];
    rows.push(capture.logits.data.slice(predictIndex * vocab, (predictIndex + 1) * vocab));
  }
  return rows;
}

export function argmax(row: Float64Array): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

import type { Intervention, Model, ModelAdapter, WeightProvider } from "@aperture/model-ir";
import { argmax, getVariantLogitRows } from "./variantSweep.js";

export interface TokenAttributionEntry {
  tokenIndex: number;
  /** baselineLogit - occludedLogit, for the target token. Positive = this token was supporting the prediction; negative = it was actively working against it. */
  logitDrop: number;
}

export interface TokenAttributionResult {
  targetTokenId: number;
  baselineLogit: number;
  /** The sequence position whose next-token prediction was attributed — echoed back so a caller that passed a default doesn't have to re-derive it. */
  predictIndex: number;
  entries: TokenAttributionEntry[];
}

/**
 * Occlusion-based attribution: for each input token up to and including
 * `predictIndex`, zero out its embedding (leaving position information
 * intact — see `embedNodeIdFor`'s doc comment) and re-run, measuring how
 * much the top prediction's logit drops. A token whose removal hurts the
 * prediction a lot gets a high score.
 *
 * Only tokens at or before `predictIndex` are ever occluded: causal
 * attention already makes anything after that position invisible to the
 * prediction being attributed, so occluding it would provably measure
 * zero effect — skipping those positions is a real forward-pass count
 * reduction when attributing toward a mid-prompt position, not just a
 * micro-optimization.
 *
 * This is one honest, cheap attribution method among several the project
 * notes discuss (gradient-based, integrated gradients, activation
 * patching, logit difference); it does not claim to be the definitive
 * measure of "importance" — occlusion has well-known blind spots (e.g. it
 * can't see redundant/backup signals two tokens both carry). Gradient-based
 * methods would need a from-scratch autodiff layer over every op in
 * nn-ops, which is out of scope here.
 *
 * Every position's occlusion (plus a baseline) is scored as one batch of
 * variants via getVariantLogitRows — on an adapter with a batched
 * runAttributionSweep (PLAN.md §8.4) this is a couple of real forward
 * passes total instead of S+1 sequential ones.
 */
export async function computeTokenAttribution(
  model: Model,
  weightProvider: WeightProvider,
  adapter: ModelAdapter,
  tokenIds: number[],
  options: { predictIndex?: number } = {}
): Promise<TokenAttributionResult> {
  const S = tokenIds.length;
  const predictIndex = options.predictIndex ?? S - 1;
  const embedNodeId = embedNodeIdFor(model);

  // Only tokens at or before predictIndex are ever occluded — see this
  // function's doc comment for why occluding anything later is provably a
  // zero-effect measurement, not just a skipped optimization.
  const variants: Intervention[][] = [[]];
  for (let t = 0; t <= predictIndex; t++) variants.push([{ nodeId: embedNodeId, operation: "zero", tokenIndex: t }]);

  const rows = await getVariantLogitRows(model, weightProvider, adapter, tokenIds, predictIndex, variants);
  const baseRow = rows[0];
  const targetTokenId = argmax(baseRow);
  const baselineLogit = baseRow[targetTokenId];

  const entries: TokenAttributionEntry[] = [];
  for (let t = 0; t <= predictIndex; t++) {
    entries.push({ tokenIndex: t, logitDrop: baselineLogit - rows[t + 1][targetTokenId] });
  }

  return { targetTokenId, baselineLogit, predictIndex, entries };
}

/**
 * GPT-2 keeps token identity ("wte") and position ("wpe") as separate
 * additive tensors, so zeroing "wte" at one row removes only that token's
 * identity. The Llama family folds both into one "embed" node and injects
 * position later via RoPE (computed from the position index alone, not
 * from the embedding values) — so zeroing "embed" at one row has the same
 * "keep position, remove identity" effect there too.
 */
function embedNodeIdFor(model: Model): string {
  return model.nodes["ln_f"] ? "wte" : "embed";
}

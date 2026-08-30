import type { Intervention, Model, ModelAdapter, ModelNode, WeightProvider } from "@aperture/model-ir";
import { argmax, getVariantLogitRows } from "./variantSweep.js";

export interface HeadAttributionEntry {
  /** The owning attention node's id (e.g. "block.1.attn") — selecting this in the graph/tree jumps straight to it. */
  nodeId: string;
  /** The owning transformer block's display name (e.g. "Transformer Block 1"), for UI labels. */
  blockLabel: string;
  headIndex: number;
  logitDrop: number;
}

export interface HeadAttributionResult {
  targetTokenId: number;
  baselineLogit: number;
  predictIndex: number;
  entries: HeadAttributionEntry[];
  /** True if the model has more (block, head) combinations than this function will attribute — see MAX_COMBINATIONS. */
  truncated: boolean;
}

// Every one of this app's tiny-random presets has well under this many
// (block, head) combinations, so they always get full coverage there. Kept
// low because this cap only ever bites on the *sequential* fallback path
// (no batched runAttributionSweep) — there, each combination is one real
// forward pass, so it bounds a single UI action to a sane number of them.
const MAX_COMBINATIONS_SEQUENTIAL = 64;

// A batched sweep (PLAN.md §8.4) scores hundreds of combinations in a
// handful of real forward passes — measured at ~9s for a full 36-layer,
// 16-head model's 576 combinations — so this cap only exists as a sanity
// ceiling against a pathologically large model, not a real UX concern.
const MAX_COMBINATIONS_BATCHED = 4096;

/**
 * Head-level occlusion attribution: for every (transformer block, attention
 * head) pair, zero out just that head's contribution (the same "zero_head"
 * intervention the Experiment tab exposes manually — see
 * nn-ops/intervene.ts's applyHeadIntervention) and re-run, measuring how
 * much the target token's logit drops. Mirrors computeTokenAttribution's
 * occlusion loop exactly, just swapping "which token" for "which head".
 *
 * `headIndex` here always means a *query* head (0..numHeads-1) — attention's
 * output is numHeads*headDim wide regardless of a GQA model's smaller
 * key/value head count, and that's the dimension applyHeadIntervention
 * actually slices.
 */
export async function computeHeadAttribution(
  model: Model,
  weightProvider: WeightProvider,
  adapter: ModelAdapter,
  tokenIds: number[],
  options: { predictIndex?: number } = {}
): Promise<HeadAttributionResult> {
  const S = tokenIds.length;
  const predictIndex = options.predictIndex ?? S - 1;

  const attentionNodes = Object.values(model.nodes)
    .filter((n) => n.type === "attention")
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

  const combos: { node: ModelNode; headIndex: number }[] = [];
  for (const node of attentionNodes) {
    const numHeads = Number(node.metadata.numHeads ?? model.config.numHeads);
    for (let h = 0; h < numHeads; h++) combos.push({ node, headIndex: h });
  }
  // See the two MAX_COMBINATIONS_* doc comments: only the slow one-request-
  // per-combination fallback needs the tight cap.
  const cap = adapter.runAttributionSweep ? MAX_COMBINATIONS_BATCHED : MAX_COMBINATIONS_SEQUENTIAL;
  const truncated = combos.length > cap;
  const scoped = truncated ? combos.slice(0, cap) : combos;

  const variants: Intervention[][] = [[], ...scoped.map(({ node, headIndex }): Intervention[] => [{ nodeId: node.id, operation: "zero_head", headIndex }])];
  const rows = await getVariantLogitRows(model, weightProvider, adapter, tokenIds, predictIndex, variants);
  const baseRow = rows[0];
  const targetTokenId = argmax(baseRow);
  const baselineLogit = baseRow[targetTokenId];

  const entries: HeadAttributionEntry[] = [];
  scoped.forEach(({ node, headIndex }, i) => {
    entries.push({
      nodeId: node.id,
      blockLabel: (node.parentId ? model.nodes[node.parentId]?.name : undefined) ?? node.name,
      headIndex,
      logitDrop: baselineLogit - rows[i + 1][targetTokenId],
    });
  });

  return { targetTokenId, baselineLogit, predictIndex, entries, truncated };
}

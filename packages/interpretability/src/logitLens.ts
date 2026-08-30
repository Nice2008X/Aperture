import type { ActivationCapture, Model, WeightProvider } from "@aperture/model-ir";
import { gemmaRmsNorm, layerNorm, linear, rmsNorm, softmaxRow, tensorToMatrix, tensorToVector, type Matrix } from "@aperture/nn-ops";
import { yieldToBrowser } from "./yield.js";

export interface LogitLensEntry {
  nodeId: string;
  label: string;
  topTokens: { tokenId: number; prob: number }[];
}

/**
 * Classic logit lens: take the residual stream at every layer, run it
 * through the model's own *final* norm + LM head (the projection those
 * weights were actually trained for — skipping the final norm and just
 * matrix-multiplying raw intermediate states would misrepresent what the
 * model "believes" at that layer), and see how the predicted token
 * distribution sharpens as depth increases.
 *
 * Reuses activations already captured by a normal `runInference()` call —
 * no extra forward pass needed, just extra (cheap, single-row) LM-head
 * projections on tensors that exist already.
 */
export async function computeLogitLens(
  model: Model,
  weightProvider: WeightProvider,
  capture: ActivationCapture,
  options: { tokenIndex?: number; topK?: number } = {}
): Promise<LogitLensEntry[]> {
  // Yield before starting: lets the "loading" state a caller just set
  // actually reach the screen before the (synchronous, uninterrupted)
  // tensor loads and per-layer projections below begin.
  await yieldToBrowser();

  const cfg = model.config;
  const topK = options.topK ?? 5;
  const S = capture.tokenIds.length;
  const tokenIndex = options.tokenIndex ?? S - 1;

  const isGpt2Style = !!model.nodes["ln_f"];
  const finalNormNodeId = isGpt2Style ? "ln_f" : "norm";
  const finalNormNode = model.nodes[finalNormNodeId];
  if (!finalNormNode) throw new Error(`Model has no final-norm node ("${finalNormNodeId}") to logit-lens through`);

  const eps = isGpt2Style ? Number(cfg.extra.layerNormEpsilon ?? 1e-5) : Number(cfg.extra.rmsNormEps ?? 1e-6);
  const normGamma = tensorToVector(await weightProvider.loadTensor(finalNormNode.parameters[0].name));
  const normBeta = isGpt2Style ? tensorToVector(await weightProvider.loadTensor(finalNormNode.parameters[1].name)) : null;
  const normFn: (row: number[]) => number[] = isGpt2Style
    ? (row) => layerNorm([row], normGamma, normBeta!, eps)[0]
    : cfg.extra.rmsNormVariant === "gemma"
      ? (row) => gemmaRmsNorm([row], normGamma, eps)[0]
      : (row) => rmsNorm([row], normGamma, eps)[0];

  const lmHeadRef = model.nodes["lm_head"].parameters[0];
  const lmHeadW = tensorToMatrix(await weightProvider.loadTensor(lmHeadRef.name));

  // "embed" is a directly-captured node for the Llama family; GPT-2 keeps
  // token/positional embeddings separate and sums them on the fly here to
  // reconstruct the pre-block-0 residual stream, since that sum isn't
  // itself a graph node (see gpt2/inference.ts).
  function embeddingHiddenState(): Matrix | null {
    if (capture.activations["embed"]) return tensorToMatrix(capture.activations["embed"]);
    const wte = capture.activations["wte"];
    const wpe = capture.activations["wpe"];
    if (!wte || !wpe) return null;
    const a = tensorToMatrix(wte);
    const b = tensorToMatrix(wpe);
    return a.map((row, r) => row.map((v, c) => v + b[r][c]));
  }

  interface Candidate {
    nodeId: string;
    label: string;
    hidden: Matrix | null;
    /** true for the model's real final-norm output, which must NOT be re-normalized before the LM head. */
    alreadyNormed: boolean;
  }

  const candidates: Candidate[] = [{ nodeId: "embed", label: "Embedding", hidden: embeddingHiddenState(), alreadyNormed: false }];
  for (let i = 0; i < cfg.numLayers; i++) {
    const id = `block.${i}`;
    const t = capture.activations[id];
    candidates.push({ nodeId: id, label: `Block ${i}`, hidden: t ? tensorToMatrix(t) : null, alreadyNormed: false });
  }
  const finalT = capture.activations[finalNormNodeId];
  candidates.push({ nodeId: finalNormNodeId, label: "Final", hidden: finalT ? tensorToMatrix(finalT) : null, alreadyNormed: true });

  const results: LogitLensEntry[] = [];
  for (const c of candidates) {
    const row = c.hidden?.[tokenIndex];
    if (!row) continue;
    // A large vocabulary (real GLM-4/Qwen-class checkpoints run 150K+
    // tokens) makes this projection+sort per layer add up across a deep
    // model — yield so a many-layer run doesn't lock up the tab for its
    // whole duration in one uninterrupted block.
    await yieldToBrowser();
    const normedRow = c.alreadyNormed ? row : normFn(row);
    const logitsRow = linear([normedRow], lmHeadW, null, "out_in")[0];
    const probs = softmaxRow(logitsRow);
    const ranked = probs.map((prob, tokenId) => ({ tokenId, prob })).sort((a, b) => b.prob - a.prob);
    results.push({ nodeId: c.nodeId, label: c.label, topTokens: ranked.slice(0, topK) });
  }
  return results;
}

import type { ActivationCapture, Intervention, Model, Tensor, WeightProvider } from "@aperture/model-ir";
import {
  addMatrices,
  applyActivation,
  applyHeadIntervention,
  applyInterventions,
  applyRopeToHead,
  causalSelfAttention,
  clamp,
  embed,
  gemmaRmsNorm,
  layerNorm,
  linear,
  matrixToTensor,
  mulMatricesElementwise,
  ropeCosSin,
  rmsNorm,
  scaleMatrix,
  tensorToMatrix,
  tensorToVector,
  type Matrix,
} from "@aperture/nn-ops";

export async function runInference(model: Model, weightProvider: WeightProvider, tokenIds: number[], interventions?: Intervention[]): Promise<ActivationCapture> {
  const cfg = model.config;
  const S = tokenIds.length;
  const numHeads = cfg.numHeads;
  const numKeyValueHeads = Number(cfg.extra.numKeyValueHeads ?? numHeads);
  const headDim = Number(cfg.extra.headDim ?? cfg.hiddenSize / numHeads);
  const eps = Number(cfg.extra.rmsNormEps ?? 1e-6);
  const ropeTheta = Number(cfg.extra.ropeTheta ?? 10000);
  const activationKind = String(cfg.extra.activationFunction ?? "silu");
  const hasQkvBias = cfg.extra.qkvBias === true;
  const hasQkNorm = cfg.extra.qkNorm === true;
  const hasFusedQkv = cfg.extra.fusedQkv === true;
  const hasFusedGateUp = cfg.extra.fusedGateUp === true;
  const hasSandwichNorm = cfg.extra.sandwichNorm === true;
  const isLayerNormNoAffine = cfg.extra.normType === "layernorm_no_affine";
  const clipQkv = cfg.extra.clipQkv != null ? Number(cfg.extra.clipQkv) : null;
  const partialRotaryFactor = Number(cfg.extra.partialRotaryFactor ?? 1);
  const rotaryDim = Math.round(headDim * partialRotaryFactor);
  const qDim = numHeads * headDim;
  const kvDim = numKeyValueHeads * headDim;
  const zerosH = new Array(cfg.hiddenSize).fill(0);
  const onesH = new Array(cfg.hiddenSize).fill(1);

  const activations: ActivationCapture["activations"] = {};
  const attentionWeights: ActivationCapture["attentionWeights"] = {};

  const loadMatrix = async (name: string): Promise<Matrix> => tensorToMatrix(await weightProvider.loadTensor(name));
  const loadVector = async (name: string): Promise<number[]> => tensorToVector(await weightProvider.loadTensor(name));
  const record = (nodeId: string, m: Matrix): Matrix => {
    const patched = applyInterventions(nodeId, m, interventions);
    activations[nodeId] = matrixToTensor(patched);
    return patched;
  };

  // OLMo's LayerNorm has no learnable weight or bias — nothing to load from
  // the checkpoint; every other adapter in this family loads a real gamma.
  const loadNormGamma = async (name: string): Promise<number[]> => (isLayerNormNoAffine ? onesH : await loadVector(name));
  const applyNorm = (x: Matrix, gamma: number[]): Matrix =>
    isLayerNormNoAffine ? layerNorm(x, gamma, zerosH, eps) : cfg.extra.rmsNormVariant === "gemma" ? gemmaRmsNorm(x, gamma, eps) : rmsNorm(x, gamma, eps);
  const clipQkvIfSet = (m: Matrix): Matrix => (clipQkv != null ? clamp(m, -clipQkv, clipQkv) : m);

  const embedTokens = await loadMatrix("model.embed_tokens.weight");
  let x = embed(tokenIds, embedTokens);
  if (cfg.extra.embeddingScale === "sqrt_hidden") x = scaleMatrix(x, Math.sqrt(cfg.hiddenSize));
  x = record("embed", x);

  const { cos, sin } = ropeCosSin(S, rotaryDim, ropeTheta);

  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `model.layers.${i}`;
    const blockInput = x;

    const rms1g = await loadNormGamma(`${L}.input_layernorm.weight`);
    const rms1Out = record(`${b}.rms1`, applyNorm(x, rms1g));

    let qW: Matrix, kW: Matrix, vW: Matrix;
    let qB: number[] | null = null, kB: number[] | null = null, vB: number[] | null = null;
    if (hasFusedQkv) {
      // Phi3/Phi4: one qkv_proj weight; Q/K/V are row-slices (out_features), loaded once and sliced locally — same approach GPT-2 uses for its fused c_attn.
      const qkvW = await loadMatrix(`${L}.self_attn.qkv_proj.weight`);
      qW = qkvW.slice(0, qDim);
      kW = qkvW.slice(qDim, qDim + kvDim);
      vW = qkvW.slice(qDim + kvDim, qDim + 2 * kvDim);
    } else {
      qW = await loadMatrix(`${L}.self_attn.q_proj.weight`); // [numHeads*headDim, hidden], out_in
      kW = await loadMatrix(`${L}.self_attn.k_proj.weight`); // [numKVHeads*headDim, hidden]
      vW = await loadMatrix(`${L}.self_attn.v_proj.weight`);
      // Qwen2 keeps a bias on q/k/v (not o_proj) — every other model in this family has none.
      if (hasQkvBias) {
        qB = await loadVector(`${L}.self_attn.q_proj.bias`);
        kB = await loadVector(`${L}.self_attn.k_proj.bias`);
        vB = await loadVector(`${L}.self_attn.v_proj.bias`);
      }
    }
    // OLMo optionally clamps Q/K/V to [-clip_qkv, clip_qkv] right after
    // projection, before QK-norm/RoPE see them — a no-op for every other
    // adapter in this family, whose clipQkv is always null.
    let q = record(`${b}.attn.q`, clipQkvIfSet(linear(rms1Out, qW, qB, "out_in")));
    let k = record(`${b}.attn.k`, clipQkvIfSet(linear(rms1Out, kW, kB, "out_in")));
    const v = record(`${b}.attn.v`, clipQkvIfSet(linear(rms1Out, vW, vB, "out_in")));

    // Qwen3 normalizes each head's Q/K vector (one shared [head_dim] weight
    // reused across every head) right after projection, before RoPE.
    if (hasQkNorm) {
      const qNormG = await loadVector(`${L}.self_attn.q_norm.weight`);
      const kNormG = await loadVector(`${L}.self_attn.k_norm.weight`);
      q = record(`${b}.attn.q_norm`, applyNormPerHead(q, numHeads, headDim, qNormG, eps));
      k = record(`${b}.attn.k_norm`, applyNormPerHead(k, numKeyValueHeads, headDim, kNormG, eps));
    }

    // RoPE is applied per-head; q/k are laid out as concatenated heads, so rotate each head's slice independently.
    // GLM-4 only rotates the first `rotaryDim` dimensions of each head (partial_rotary_factor < 1) — the rest pass through unrotated; every other adapter has rotaryDim === headDim, so this is a no-op change for them.
    q = applyRopePerHead(q, numHeads, headDim, rotaryDim, cos, sin);
    k = applyRopePerHead(k, numKeyValueHeads, headDim, rotaryDim, cos, sin);
    // The rope node conceptually rotates both Q and K; a single Tensor per node
    // can only hold one of them, so it shows the roped Q — directly comparable
    // against the pre-rope value recorded at `.attn.q` above.
    q = record(`${b}.attn.rope`, q);

    const { output: attnHeadsRaw, attentionWeights: headWeights } = causalSelfAttention(q, k, v, numHeads, numKeyValueHeads, headDim);
    const attnRaw = applyHeadIntervention(`${b}.attn`, attnHeadsRaw, interventions, headDim);
    attentionWeights[`${b}.attn`] = headsToTensor(headWeights);

    const oW = await loadMatrix(`${L}.self_attn.o_proj.weight`);
    const attnOutRaw = linear(attnRaw, oW, null, "out_in");
    const attnProjected = record(`${b}.attn.out`, attnOutRaw);
    // "block.N.attn" is the container users actually click in the tree/graph
    // (its leaf children aren't shown there — see graph.ts) so an
    // intervention targeting it must also flow downstream, not just get
    // recorded for display.
    const attnOut = record(`${b}.attn`, attnProjected);

    // GLM-4's sandwich norm: on top of rms1's pre-attention norm, one more
    // RMSNorm on the attention sub-layer's output, right before it joins
    // the residual stream.
    let attnForResidual = attnOut;
    if (hasSandwichNorm) {
      const postAttnG = await loadNormGamma(`${L}.post_self_attn_layernorm.weight`);
      attnForResidual = record(`${b}.attn.post_norm`, applyNorm(attnOut, postAttnG));
    }

    const res1 = record(`${b}.res1`, addMatrices(attnForResidual, blockInput));

    const rms2g = await loadNormGamma(`${L}.post_attention_layernorm.weight`);
    const rms2Out = record(`${b}.rms2`, applyNorm(res1, rms2g));

    let gateW: Matrix, upW: Matrix;
    if (hasFusedGateUp) {
      // Phi3/Phi4: one gate_up_proj weight, split into two equal row-halves.
      const gateUpW = await loadMatrix(`${L}.mlp.gate_up_proj.weight`);
      gateW = gateUpW.slice(0, cfg.intermediateSize);
      upW = gateUpW.slice(cfg.intermediateSize, 2 * cfg.intermediateSize);
    } else {
      gateW = await loadMatrix(`${L}.mlp.gate_proj.weight`);
      upW = await loadMatrix(`${L}.mlp.up_proj.weight`);
    }
    const gateOut = record(`${b}.ffn.gate`, linear(rms2Out, gateW, null, "out_in"));

    const gateAct = record(`${b}.ffn.gate_act`, applyActivation(gateOut, activationKind));

    const upOut = record(`${b}.ffn.up`, linear(rms2Out, upW, null, "out_in"));

    const mulOut = record(`${b}.ffn.mul`, mulMatricesElementwise(gateAct, upOut));

    const downW = await loadMatrix(`${L}.mlp.down_proj.weight`);
    const ffnProjected = record(`${b}.ffn.down`, linear(mulOut, downW, null, "out_in"));
    const ffnOut = record(`${b}.ffn`, ffnProjected);

    let ffnForResidual = ffnOut;
    if (hasSandwichNorm) {
      const postMlpG = await loadNormGamma(`${L}.post_mlp_layernorm.weight`);
      ffnForResidual = record(`${b}.ffn.post_norm`, applyNorm(ffnOut, postMlpG));
    }

    const res2 = record(`${b}.res2`, addMatrices(ffnForResidual, res1));
    // Same reasoning as the attn/ffn aliases above: "block.N" is the node
    // users actually click at the architecture level, so it must be a real
    // intervention point that feeds block N+1, not a discarded display copy.
    x = record(b, res2);
  }

  const normg = await loadNormGamma("model.norm.weight");
  const normOut = record("norm", applyNorm(x, normg));

  const lmHeadRef = model.nodes["lm_head"].parameters[0];
  const lmHeadW = await loadMatrix(lmHeadRef.name); // [vocab, hidden], out_in
  const logits = record("lm_head", linear(normOut, lmHeadW, null, "out_in"));

  return {
    tokenIds,
    tokens: tokenIds.map((id) => `#${id}`),
    activations,
    attentionWeights,
    logits: matrixToTensor(logits),
  };
}

/** Qwen3's QK-Norm: the same [head_dim] RMSNorm weight applied independently to each head's slice (always the standard RMSNorm formula, never the Gemma (1+weight) variant, regardless of the model's main norm). */
function applyNormPerHead(x: Matrix, numHeads: number, headDim: number, gamma: number[], eps: number): Matrix {
  const S = x.length;
  const out: Matrix = Array.from({ length: S }, () => new Array(numHeads * headDim));
  for (let h = 0; h < numHeads; h++) {
    const headSlice = x.map((row) => row.slice(h * headDim, (h + 1) * headDim));
    const normed = rmsNorm(headSlice, gamma, eps);
    for (let s = 0; s < S; s++) {
      for (let d = 0; d < headDim; d++) out[s][h * headDim + d] = normed[s][d];
    }
  }
  return out;
}

/** Rotates only the first `rotaryDim` dimensions of each head (GLM-4's partial_rotary_factor); the remaining `headDim - rotaryDim` pass through unchanged. Every other adapter calls this with rotaryDim === headDim, where the passthrough slice is empty and behavior is identical to rotating the whole head. */
function applyRopePerHead(x: Matrix, numHeads: number, headDim: number, rotaryDim: number, cos: Matrix, sin: Matrix): Matrix {
  const S = x.length;
  const out: Matrix = Array.from({ length: S }, () => new Array(numHeads * headDim));
  for (let h = 0; h < numHeads; h++) {
    const headSlice = x.map((row) => row.slice(h * headDim, (h + 1) * headDim));
    const rotaryPart = headSlice.map((row) => row.slice(0, rotaryDim));
    const roped = rotaryDim > 0 ? applyRopeToHead(rotaryPart, cos, sin) : rotaryPart;
    for (let s = 0; s < S; s++) {
      for (let d = 0; d < rotaryDim; d++) out[s][h * headDim + d] = roped[s][d];
      for (let d = rotaryDim; d < headDim; d++) out[s][h * headDim + d] = headSlice[s][d];
    }
  }
  return out;
}

function headsToTensor(headWeights: number[][][]): Tensor {
  const numHeads = headWeights.length;
  const S = headWeights[0]?.length ?? 0;
  const data = new Float64Array(numHeads * S * S);
  let idx = 0;
  for (let h = 0; h < numHeads; h++) {
    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        const w = headWeights[h][i][j];
        data[idx++] = Number.isFinite(w) ? w : 0;
      }
    }
  }
  return { shape: [numHeads, S, S], dtype: "F32", data };
}

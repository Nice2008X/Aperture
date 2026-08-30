import type { Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@aperture/model-ir";
import { SafetensorsWeightProvider } from "@aperture/tensor-core";
import { loadSafetensorsMetadata } from "@aperture/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@aperture/adapter-llama-family";

// Qwen2 (and Qwen2.5, which reuses the same architecture class and config
// shape — only training data/tokenizer differ) is otherwise a Llama-family
// model: RoPE, standard RMSNorm, SwiGLU gated FFN, GQA. Its one real
// architectural difference is a bias on q_proj/k_proj/v_proj (o_proj has
// none) — a deliberate holdover from the original Qwen design, everywhere
// else in this family has no attention/MLP biases at all.
//
// Deliberately scoped to exactly "qwen2" / "Qwen2ForCausalLM": Qwen3 uses
// its own model_type ("qwen3") and adds real differences (QK-norm) this
// adapter doesn't implement, and multimodal Qwen variants (VL, Audio, ...)
// have different weight structures entirely — an exact match means an
// unsupported checkpoint fails loudly with "no adapter found" instead of
// silently producing wrong output.
const PROVIDER_ID = "qwen-weights";

export const QwenAdapter: ModelAdapter = {
  id: "qwen2",
  displayName: "Qwen2",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "qwen2" || (metadata.architectures ?? []).some((a) => a === "Qwen2ForCausalLM");
  },

  async loadMetadata(source: ModelSource): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "Qwen2ForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "qwen2", qkvBias: true }),
      weightIndex,
      source,
      weightsBuffer,
    };
  },

  buildGraph(metadata: ModelMetadata): Model {
    return buildGraph(metadata, PROVIDER_ID);
  },

  getWeightProvider(metadata: ModelMetadata): WeightProvider {
    if (!metadata.weightsBuffer) throw new Error("No weights buffer available on this metadata");
    return new SafetensorsWeightProvider(PROVIDER_ID, metadata.weightsBuffer);
  },

  runInference,
};

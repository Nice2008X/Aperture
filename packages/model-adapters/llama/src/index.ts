import type { Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@aperture/model-ir";
import { SafetensorsWeightProvider } from "@aperture/tensor-core";
import { loadSafetensorsMetadata } from "@aperture/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@aperture/adapter-llama-family";

const PROVIDER_ID = "llama-weights";

export const LlamaAdapter: ModelAdapter = {
  id: "llama",
  displayName: "Llama",

  canLoad(_source, metadata) {
    if (!metadata) return true; // optimistic default until config.json is fetched
    return metadata.model_type === "llama" || (metadata.architectures ?? []).some((a) => a.startsWith("Llama"));
  },

  async loadMetadata(source: ModelSource): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "LlamaForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "llama" }),
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

import type { ActivationCapture, Intervention, LoadProgressEvent, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@aperture/model-ir";
import { uniqueParameters } from "@aperture/model-ir";
import { loadModel, runAttributionSweep, runInference } from "./http.js";
import { RemoteWeightProvider } from "./weightProvider.js";

/**
 * The one adapter that replaces all 8 hand-written client-side adapters
 * (packages/model-adapters/*) — see PLAN.md §1. Where those had to
 * hand-derive a Model IR graph from a parsed config.json because there was
 * no ML framework in the browser, the backend already built the real graph
 * from a loaded `transformers` model (apps/api's generic graph_builder.py)
 * and just hands it over whole. This adapter's job is purely plumbing:
 * fetch that graph, wrap tensor access in a WeightProvider that talks to
 * the same backend.
 *
 * runInference is intentionally omitted here — phase 1 (PLAN.md §8.1) is
 * static architecture/weight visualization only. Phase 2 adds it, still as
 * a call to the backend rather than any client-side math. Phase 3 adds real
 * intervention support (hook-based ablation/patching, see apps/api's
 * inference.py), hence supportsInterventions: true below.
 */
export const BackendAdapter: ModelAdapter = {
  id: "backend",
  displayName: "GPU Backend",
  supportsInterventions: true,

  canLoad(source: ModelSource): boolean {
    return source.kind === "backend";
  },

  async loadMetadata(source: ModelSource, onProgress?: (event: LoadProgressEvent) => void): Promise<ModelMetadata> {
    if (source.kind !== "backend") {
      throw new Error(`BackendAdapter only loads { kind: "backend" } sources, got "${source.kind}".`);
    }
    const model = await loadModel(source.modelId, "bf16", onProgress, source.quantization ?? null);
    const weightIndex: ModelMetadata["weightIndex"] = {};
    for (const p of uniqueParameters(model)) weightIndex[p.name] = { shape: p.shape, dtype: p.dtype };
    return { architecture: model.architecture, config: model.config, weightIndex, source, prebuiltModel: model };
  },

  buildGraph(metadata: ModelMetadata) {
    if (!metadata.prebuiltModel) throw new Error("BackendAdapter.buildGraph called without a prebuiltModel — loadMetadata must run first.");
    return metadata.prebuiltModel;
  },

  getWeightProvider(metadata: ModelMetadata) {
    if (!metadata.prebuiltModel) throw new Error("BackendAdapter.getWeightProvider called without a prebuiltModel — loadMetadata must run first.");
    if (metadata.source.kind !== "backend") throw new Error(`Expected a "backend" source, got "${metadata.source.kind}".`);
    return new RemoteWeightProvider(metadata.source.modelId, uniqueParameters(metadata.prebuiltModel));
  },

  // PLAN.md §8.2-3 — a real forward pass, run server-side, interventions
  // applied via real forward hooks (apps/api's inference.py) so everything
  // downstream of a targeted node genuinely sees the edited value. `model`
  // is unused: the backend already has everything it needs keyed by model
  // id, which RemoteWeightProvider.id carries (set to it at construction).
  async runInference(_model: Model, weightProvider: WeightProvider, tokenIds: number[], interventions?: Intervention[]): Promise<ActivationCapture> {
    return runInference(weightProvider.id, tokenIds, interventions);
  },

  // PLAN.md §8.4 — batched occlusion sweeps (Token/Head Attribution) for
  // the same reason runInference proxies to the backend: real computation,
  // not free in-browser math, so N variants should cost a handful of real
  // forward passes, not N of them.
  async runAttributionSweep(_model: Model, weightProvider: WeightProvider, tokenIds: number[], predictIndex: number, variants: Intervention[][]): Promise<Float64Array[]> {
    return runAttributionSweep(weightProvider.id, tokenIds, predictIndex, variants);
  },
};

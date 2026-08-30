import type { ParameterRef, Tensor, TensorSlice, WeightProvider } from "@aperture/model-ir";
import { getWeightTensor } from "./http.js";

/**
 * WeightProvider backed by the GPU backend instead of an in-browser
 * safetensors buffer — the server-side counterpart of what
 * SafetensorsWeightProvider's doc comment (packages/tensor-core) already
 * anticipated: "a server doing true byte-range HTTP reads instead of an
 * in-memory buffer — nothing above this provider would change." Every
 * consumer (TensorExplorer, Inspector, ...) is unaffected by this swap.
 */
export class RemoteWeightProvider implements WeightProvider {
  id: string;
  private modelId: string;
  private paramIndex: Map<string, ParameterRef>;

  constructor(modelId: string, parameters: ParameterRef[]) {
    this.id = modelId;
    this.modelId = modelId;
    this.paramIndex = new Map(parameters.map((p) => [p.name, p]));
  }

  async listParameters(): Promise<ParameterRef[]> {
    return [...this.paramIndex.values()];
  }

  async getParameterInfo(parameterId: string): Promise<ParameterRef> {
    const ref = this.paramIndex.get(parameterId);
    if (!ref) throw new Error(`Unknown parameter: ${parameterId}`);
    return ref;
  }

  async loadTensor(parameterId: string, options?: TensorSlice): Promise<Tensor> {
    return getWeightTensor(this.modelId, parameterId, options);
  }
}

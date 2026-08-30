import { useCallback, useState } from "react";
import type { ActivationCapture, Model, ModelAdapter, WeightProvider } from "@aperture/model-ir";
import type { Tokenizer } from "@aperture/tokenizer";

export interface InferenceState {
  status: "idle" | "running" | "ready" | "error";
  error?: string;
  result?: ActivationCapture;
  displayTokens?: string[];
}

export function useInference(model: Model | undefined, weightProvider: WeightProvider | undefined, adapter: ModelAdapter | undefined, tokenizer: Tokenizer | undefined) {
  const [state, setState] = useState<InferenceState>({ status: "idle" });

  const run = useCallback(
    async (prompt: string) => {
      if (!model || !weightProvider || !tokenizer) return;
      if (!adapter?.runInference) {
        setState({ status: "error", error: `${adapter?.displayName ?? "This adapter"} does not support running inference yet.` });
        return;
      }
      setState({ status: "running" });
      try {
        const { ids, displayTokens } = tokenizer.encode(prompt);
        if (ids.length === 0) throw new Error("Prompt tokenized to zero tokens — try a non-empty prompt.");
        const result = await adapter.runInference(model, weightProvider, ids);
        setState({ status: "ready", result, displayTokens });
      } catch (err) {
        setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [model, weightProvider, adapter, tokenizer]
  );

  // Same underlying call as `run`, minus the encode step — for driving
  // inspection off token ids that already exist (e.g. a generated step's
  // prompt+continuation prefix, from useGeneration) rather than re-encoding
  // a text prompt. Doesn't require a tokenizer for the run itself; only
  // uses one (if present) to produce nicer chip labels than raw ids.
  const runTokenIds = useCallback(
    async (tokenIds: number[]) => {
      if (!model || !weightProvider) return;
      if (!adapter?.runInference) {
        setState({ status: "error", error: `${adapter?.displayName ?? "This adapter"} does not support running inference yet.` });
        return;
      }
      setState({ status: "running" });
      try {
        if (tokenIds.length === 0) throw new Error("No tokens to run inference on.");
        const result = await adapter.runInference(model, weightProvider, tokenIds);
        const displayTokens = tokenizer ? tokenIds.map((id) => tokenizer.decodeToken(id)) : tokenIds.map((id) => `#${id}`);
        setState({ status: "ready", result, displayTokens });
      } catch (err) {
        setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [model, weightProvider, adapter, tokenizer]
  );

  const reset = useCallback(() => setState({ status: "idle" }), []);

  return { state, run, runTokenIds, reset };
}

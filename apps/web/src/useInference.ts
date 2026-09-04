import { useCallback, useState } from "react";
import type { ActivationCapture, Model, ModelAdapter, WeightProvider } from "@aperture/model-ir";
import type { Tokenizer } from "@aperture/tokenizer";

export interface InferenceState {
  status: "idle" | "running" | "ready" | "error";
  error?: string;
  result?: ActivationCapture;
  displayTokens?: string[];
  /** Wall-clock time of the `adapter.runInference` call itself (not tokenization) — a real browser measurement of the round trip to the GPU backend, not an estimate. */
  elapsedMs?: number;
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
      // Keeps the previous result/displayTokens in state through "running"
      // (a merge, not a full replace) — Apply/re-run should update the
      // token chips and prediction panel in place once the new result
      // lands, not blank them out and pop them back in.
      setState((prev) => ({ ...prev, status: "running" }));
      try {
        const { ids, displayTokens } = tokenizer.encode(prompt);
        if (ids.length === 0) throw new Error("Prompt tokenized to zero tokens — try a non-empty prompt.");
        const start = performance.now();
        const result = await adapter.runInference(model, weightProvider, ids);
        const elapsedMs = performance.now() - start;
        setState({ status: "ready", result, displayTokens, elapsedMs });
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
      // Keeps the previous result/displayTokens in state through "running"
      // (a merge, not a full replace) — Apply/re-run should update the
      // token chips and prediction panel in place once the new result
      // lands, not blank them out and pop them back in.
      setState((prev) => ({ ...prev, status: "running" }));
      try {
        if (tokenIds.length === 0) throw new Error("No tokens to run inference on.");
        const start = performance.now();
        const result = await adapter.runInference(model, weightProvider, tokenIds);
        const elapsedMs = performance.now() - start;
        const displayTokens = tokenizer ? tokenIds.map((id) => tokenizer.decodeToken(id)) : tokenIds.map((id) => `#${id}`);
        setState({ status: "ready", result, displayTokens, elapsedMs });
      } catch (err) {
        setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [model, weightProvider, adapter, tokenizer]
  );

  const reset = useCallback(() => setState({ status: "idle" }), []);

  return { state, run, runTokenIds, reset };
}

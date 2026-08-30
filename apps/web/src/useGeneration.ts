import { useCallback, useRef, useState } from "react";
import type { WeightProvider } from "@aperture/model-ir";
import { streamGeneration, isGenerationError, type GenerationOptions } from "@aperture/api-client";
import type { Tokenizer } from "@aperture/tokenizer";

export interface GeneratedToken {
  tokenId: number;
  text: string;
}

export interface GenerationState {
  status: "idle" | "streaming" | "done" | "error";
  error?: string;
  /** The prompt's own token ids — a generated token's absolute position in the full sequence is promptTokenIds.length + its index in `tokens`, which is what "inspect this step" needs to reconstruct the prefix to re-run. */
  promptTokenIds: number[];
  tokens: GeneratedToken[];
}

const IDLE_STATE: GenerationState = { status: "idle", promptTokenIds: [], tokens: [] };

/** Real multi-token generation (PLAN.md §8.5), streamed token by token from the backend's real KV cache — a separate mode from useInference's single-forward-pass inspection, sharing only the prompt text a caller typed. */
export function useGeneration(weightProvider: WeightProvider | undefined, tokenizer: Tokenizer | undefined) {
  const [state, setState] = useState<GenerationState>(IDLE_STATE);
  const abortRef = useRef<AbortController | null>(null);

  const generate = useCallback(
    async (prompt: string, options?: GenerationOptions) => {
      if (!weightProvider || !tokenizer) return;
      const { ids } = tokenizer.encode(prompt);
      if (ids.length === 0) {
        setState({ ...IDLE_STATE, status: "error", error: "Prompt tokenized to zero tokens — try a non-empty prompt." });
        return;
      }

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setState({ status: "streaming", promptTokenIds: ids, tokens: [] });

      try {
        for await (const event of streamGeneration(weightProvider.id, ids, options, controller.signal)) {
          if (isGenerationError(event)) throw new Error(event.error);
          setState((prev) => ({
            status: event.done ? "done" : "streaming",
            error: undefined,
            promptTokenIds: prev.promptTokenIds,
            tokens: [...prev.tokens, { tokenId: event.tokenId, text: event.text }],
          }));
        }
      } catch (err) {
        if (controller.signal.aborted) {
          // A user-initiated Stop, not a real failure — keep whatever streamed so far.
          setState((prev) => ({ ...prev, status: "done" }));
        } else {
          setState((prev) => ({ ...prev, status: "error", error: err instanceof Error ? err.message : String(err) }));
        }
      }
    },
    [weightProvider, tokenizer]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(IDLE_STATE);
  }, []);

  return { state, generate, stop, reset };
}

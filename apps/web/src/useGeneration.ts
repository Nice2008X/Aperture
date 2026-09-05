import { useCallback, useRef, useState } from "react";
import type { WeightProvider } from "@aperture/model-ir";
import { streamGeneration, isGenerationError, applyChatTemplate, type GenerationOptions } from "@aperture/api-client";
import type { Tokenizer } from "@aperture/tokenizer";

/** "raw" tokenizes the typed prompt exactly as-is (this app's original behavior — also the only option for a model with no chat_template, e.g. a base/completion checkpoint). "chat" wraps it as a single user message in the model's own trained chat format first (apps/api's apply_chat_template) — what an instruct-tuned model actually expects, and generally the better default. */
export type GenerationMode = "raw" | "chat";

/** The user-adjustable sampling knobs Settings' "Generation" section exposes — seeded from the loaded model's own generation_defaults() (apps/api) on every model load, and resettable back to those same values. Not persisted across sessions (unlike e.g. maxNewTokens): a stale override surviving a switch to a different model would fight that model's own tuned defaults, which is exactly what this feature exists to respect. */
export interface GenerationParams {
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  noRepeatNgramSize: number;
}

/** This app's own fallback — matches apps/api's generation.py module-level DEFAULT_* constants — used before any model has loaded (Settings is reachable from the loader screen too) or if fetching a model's real defaults fails. */
export const DEFAULT_GENERATION_PARAMS: GenerationParams = {
  temperature: 0.7,
  topP: 1.0,
  topK: 0,
  repetitionPenalty: 1.1,
  noRepeatNgramSize: 3,
};

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
    async (prompt: string, options?: GenerationOptions, mode: GenerationMode = "raw") => {
      if (!weightProvider || !tokenizer) return;

      let ids: number[];
      try {
        ids = mode === "chat" ? await applyChatTemplate(weightProvider.id, [{ role: "user", content: prompt }]) : tokenizer.encode(prompt).ids;
      } catch (err) {
        setState({ ...IDLE_STATE, status: "error", error: err instanceof Error ? err.message : String(err) });
        return;
      }
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

import { useCallback, useEffect, useState } from "react";
import type { LoadProgressEvent, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@aperture/model-ir";
import { peekModelType } from "@aperture/hf-client";
import { listModels, LoadCancelledError } from "@aperture/api-client";
import { loadTokenizer, type Tokenizer } from "@aperture/tokenizer";
import { ADAPTERS } from "./adapters.js";

export interface ModelState {
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
  model?: Model;
  metadata?: ModelMetadata;
  weightProvider?: WeightProvider;
  adapter?: ModelAdapter;
  source?: ModelSource;
  /** Present only if the model ships a tokenizer this app understands — inference/activation features need it, static architecture browsing doesn't. */
  tokenizer?: Tokenizer;
  /** Updated as the adapter's loadMetadata() reports progress (a real checkpoint takes tens of seconds) — see ModelLoader for where this renders. */
  loadProgress?: LoadProgressEvent;
}

export function useModel() {
  const [state, setState] = useState<ModelState>({ status: "idle" });

  const loadFromSource = useCallback(async (source: ModelSource) => {
    setState({ status: "loading" });
    try {
      // Read just enough of config.json to know what kind of model this is,
      // *before* any adapter commits to fetching (and possibly misreading)
      // its weights. This is the actual extension point for "lots of
      // different LLM models": each adapter only has to answer canLoad()
      // correctly for its own architecture — nothing else here changes.
      const preview = await peekModelType(source);
      const adapter = ADAPTERS.find((a) => a.canLoad(source, preview));
      if (!adapter) {
        throw new Error(
          `No adapter registered for model_type "${preview.model_type ?? "unknown"}" (architectures: ${(preview.architectures ?? []).join(", ") || "none"}).`
        );
      }

      const metadata = await adapter.loadMetadata(source, (event) => setState((prev) => ({ ...prev, loadProgress: event })));
      const model = adapter.buildGraph(metadata);
      const weightProvider = adapter.getWeightProvider(metadata);

      // Best-effort: this app's real tokenization for a `{ kind: "backend"
      // }` source runs through the same browser-side BPE implementation as
      // every other source (see packages/tokenizer's "backend" branch,
      // fetching tokenizer.json from the backend's static mount) — this
      // just resolves to undefined for a model that ships no
      // tokenizer.json this parser understands, which correctly keeps the
      // inference/activation panels in their "not supported yet" state.
      const tokenizer = await loadTokenizer(source).catch(() => undefined);

      setState({ status: "ready", model, metadata, weightProvider, adapter, source, tokenizer });
    } catch (err) {
      // A Stop click (ModelLoader's cancelLoad) — not a real failure, so
      // this goes back to idle exactly like reset() rather than surfacing
      // an error banner. ModelLoader itself never unmounts across this
      // transition (App.tsx keeps rendering it for any non-"ready"
      // status), so its own `selectedId` is still whatever card the user
      // clicked Load from — that card just re-renders expanded again,
      // with nothing to reset for it.
      if (err instanceof LoadCancelledError) {
        setState({ status: "idle" });
        return;
      }
      setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  const load = useCallback(
    (modelId: string, quantization?: "4bit" | "8bit") => loadFromSource({ kind: "backend", modelId: modelId.trim(), quantization }),
    [loadFromSource]
  );

  const reset = useCallback(() => setState({ status: "idle" }), []);

  // Resume whatever model the backend already has resident on the GPU —
  // otherwise refreshing the page always lands back on the loader screen
  // even though the model never actually unloaded, just this tab's
  // in-memory state did (the backend keeps one model loaded independent
  // of any browser tab — PLAN.md §4.1). Cheap: `load`'s backend call
  // already has an already-resident fast path (re-serves the cached
  // graph, no re-download or re-materializing weights onto the GPU), this
  // just calls it automatically instead of waiting for a catalog click.
  useEffect(() => {
    let cancelled = false;
    listModels()
      .then((entries) => {
        if (cancelled) return;
        const resident = entries.find((e) => e.loaded);
        if (resident) load(resident.id, resident.loadedQuantization ?? undefined);
      })
      .catch(() => {
        // Backend unreachable at startup, or some other hiccup — falls
        // back to the normal idle loader screen, same as without this effect.
      });
    return () => {
      cancelled = true;
    };
  }, [load]);

  return { state, load, reset };
}

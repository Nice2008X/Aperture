import { useCallback, useEffect, useState } from "react";
import type { LoadProgressEvent, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@aperture/model-ir";
import { peekModelType } from "@aperture/hf-client";
import { listModels, LoadCancelledError } from "@aperture/api-client";
import { loadTokenizer, type Tokenizer } from "@aperture/tokenizer";
import { ADAPTERS } from "./adapters.js";
import { useLocalStorageState } from "./useLocalStorageState.js";

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
  // Persisted so a refresh can tell "the user explicitly came back to the
  // loader screen" (App's goHome, via reset() below) apart from "no model
  // has ever been loaded in this browser yet" — only the former should
  // suppress the auto-resume effect further down; a first-ever visit (or
  // any refresh once a model IS loaded) should still resume whatever's
  // resident on the backend, same as before.
  const [stayOnLoaderScreen, setStayOnLoaderScreen] = useLocalStorageState("app:stayOnLoaderScreen", false);

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
      setStayOnLoaderScreen(false);
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
  }, [setStayOnLoaderScreen]);

  const load = useCallback(
    (modelId: string, quantization?: "4bit" | "8bit") => loadFromSource({ kind: "backend", modelId: modelId.trim(), quantization }),
    [loadFromSource]
  );

  // The one place "go back to the loader screen" happens (App's goHome) —
  // marks it so a refresh right after doesn't get auto-resumed back into
  // the model the user just deliberately stepped away from (see
  // stayOnLoaderScreen and the effect below).
  const reset = useCallback(() => {
    setStayOnLoaderScreen(true);
    setState({ status: "idle" });
  }, [setStayOnLoaderScreen]);

  // Resume whatever model the backend already has resident on the GPU —
  // otherwise refreshing the page always lands back on the loader screen
  // even though the model never actually unloaded, just this tab's
  // in-memory state did (the backend keeps one model loaded independent
  // of any browser tab — PLAN.md §4.1). Cheap: `load`'s backend call
  // already has an already-resident fast path (re-serves the cached
  // graph, no re-download or re-materializing weights onto the GPU), this
  // just calls it automatically instead of waiting for a catalog click.
  // Skipped entirely if the user's last action was explicitly navigating
  // back to the loader screen — refreshing there should keep showing the
  // loader screen, not silently jump back into whatever's still resident.
  useEffect(() => {
    if (stayOnLoaderScreen) return;
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
  }, [load, stayOnLoaderScreen]);

  return { state, load, reset };
}

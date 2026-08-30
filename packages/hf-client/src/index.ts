import type { ModelMetadata, ModelSource } from "@aperture/model-ir";
import { parseSafetensorsHeader } from "@aperture/tensor-core";
import { fetchCachedArrayBuffer } from "./modelCache.js";

export { MAX_CACHEABLE_BYTES } from "./modelCache.js";

export function hfResolveUrl(source: Extract<ModelSource, { kind: "huggingface" }>, file: string): string {
  const revision = source.revision ?? "main";
  return `https://huggingface.co/${source.repo}/resolve/${revision}/${file}`;
}

export async function fetchJson<T>(url: string): Promise<T> {
  const bytes = await fetchCachedArrayBuffer(url);
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  return fetchCachedArrayBuffer(url);
}

/** Reads one file out of a `{ kind: "local" }` source's in-memory file map — the local-loading equivalent of a fetch. */
export function readLocalBytes(source: ModelSource, filename: string): ArrayBuffer {
  if (source.kind !== "local") throw new Error(`readLocalBytes called on a non-local source (${source.kind})`);
  const bytes = source.files[filename];
  if (!bytes) throw new Error(`Missing required local file: ${filename}`);
  return bytes;
}

/** Same as `readLocalBytes`, JSON-parsed. */
export function readLocalJson<T>(source: ModelSource, filename: string): T {
  return JSON.parse(new TextDecoder().decode(readLocalBytes(source, filename))) as T;
}

export interface HfConfigPreview {
  model_type?: string;
  architectures?: string[];
}

/** Reads just enough of config.json to let ModelAdapter.canLoad decide, before any adapter commits to fetching weights. */
export async function peekModelType(source: ModelSource): Promise<HfConfigPreview> {
  if (source.kind === "local") return readLocalJson<HfConfigPreview>(source, "config.json");
  // A "backend" source's adapter (BackendAdapter in @aperture/api-client)
  // decides purely on source.kind, not on any config preview — the real
  // architecture detection already happened server-side.
  if (source.kind === "backend") return {};
  return fetchJson<HfConfigPreview>(hfResolveUrl(source, "config.json"));
}

export interface RawSafetensorsMetadata<TConfig> {
  rawConfig: TConfig;
  weightIndex: ModelMetadata["weightIndex"];
  weightsBuffer: ArrayBuffer;
}

/**
 * The fetch sequence every safetensors-backed adapter needs: raw config.json
 * (typed however that adapter likes) plus the safetensors file, with its
 * header already parsed into a name -> {shape, dtype} index. Each adapter
 * turns `rawConfig` into its own normalized ModelConfig from here. Works
 * identically for a Hugging Face source (fetched, cache-backed) and a
 * `{ kind: "local" }` source (files the user already picked, just read
 * straight out of memory) — adapters don't need to know which one it was.
 */
export async function loadSafetensorsMetadata<TConfig>(source: ModelSource): Promise<RawSafetensorsMetadata<TConfig>> {
  if (source.kind === "backend") {
    throw new Error("loadSafetensorsMetadata doesn't apply to a backend-sourced model — its adapter (BackendAdapter) never calls this.");
  }
  const rawConfig =
    source.kind === "local" ? readLocalJson<TConfig>(source, "config.json") : await fetchJson<TConfig>(hfResolveUrl(source, "config.json"));
  const weightsBuffer =
    source.kind === "local" ? readLocalBytes(source, "model.safetensors") : await fetchArrayBuffer(hfResolveUrl(source, "model.safetensors"));
  const { header } = parseSafetensorsHeader(weightsBuffer);

  const weightIndex: ModelMetadata["weightIndex"] = {};
  for (const [name, entry] of Object.entries(header)) {
    weightIndex[name] = { shape: entry.shape, dtype: entry.dtype };
  }

  return { rawConfig, weightIndex, weightsBuffer };
}

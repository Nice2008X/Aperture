import type { ActivationCapture, Intervention, LoadProgressEvent, Model, Tensor, TensorSlice } from "@aperture/model-ir";

/**
 * Base URL of the Aperture GPU backend (apps/api). Deliberately a plain
 * module-level variable rather than reading `import.meta.env` in here —
 * this package is consumed by more than one Vite app's dependency graph,
 * and each consumer already has its own typed env access. Call
 * `setApiBase` once at app startup (see apps/web/src/main.tsx).
 */
let apiBase = "http://127.0.0.1:8000";

export function setApiBase(url: string): void {
  apiBase = url.replace(/\/+$/, "");
}

export function getApiBase(): string {
  return apiBase;
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${apiBase}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Backend request failed (${res.status} ${res.statusText})${text ? `: ${text}` : ""}`);
  }
  return res;
}

export interface CatalogEntry {
  id: string;
  hfRepo: string;
  revision: string;
  displayName: string;
  family: string;
  isMoe: boolean;
  sizeBytes: number;
  downloadedAt: string;
  status: "ready" | "downloading" | "error";
  /** Whether this is the model currently resident on the GPU (the backend holds one at a time — PLAN.md §4.1). */
  loaded: boolean;
  /** The quantization the resident model is actually loaded with — null on every non-resident entry. Lets a refresh-resume re-request the exact same load and hit the backend's already-resident fast path instead of mismatching and triggering a real reload. */
  loadedQuantization: "4bit" | "8bit" | null;
}

export async function listModels(): Promise<CatalogEntry[]> {
  const res = await apiFetch("/api/models");
  return res.json();
}

export type GpuStatus =
  | { available: false }
  | { available: true; name: string; totalBytes: number; freeBytes: number; usedBytes: number };

/** Live VRAM snapshot for the loader screen's GPU bar and per-model fit-check (todo11.txt §6/§7). */
export async function getGpuStatus(): Promise<GpuStatus> {
  const res = await apiFetch("/api/gpu");
  return res.json();
}

export type LoadDtype = "bf16" | "fp16" | "fp32";
export type LoadQuantization = "4bit" | "8bit" | null;

interface LoadProgressWire {
  phase: string;
  desc?: string;
  n?: number;
  total?: number;
}
interface LoadDoneWire {
  done: true;
  graph: Model;
}
interface LoadCancelledWire {
  cancelled: true;
}
interface LoadErrorWire {
  error: string;
}
type LoadEventWire = LoadProgressWire | LoadDoneWire | LoadCancelledWire | LoadErrorWire;

/** Thrown by loadModel when the load was stopped via cancelLoad (a Stop click), never for a real failure — callers should treat this as "back to idle", not as an error to display. */
export class LoadCancelledError extends Error {
  constructor(modelId: string) {
    super(`Load cancelled for ${modelId}.`);
    this.name = "LoadCancelledError";
  }
}

/**
 * Loads (or re-serves, if already resident) a model onto the GPU and
 * returns its full Model IR graph — streamed as SSE server-side (a real
 * multi-GB checkpoint takes tens of seconds), so `onProgress` gets called
 * as each weight materializes rather than the caller just waiting on one
 * long-pending promise. See apps/api's ModelRegistry.load_with_progress
 * for exactly where these events come from.
 */
export async function loadModel(
  modelId: string,
  dtype: LoadDtype = "bf16",
  onProgress?: (event: LoadProgressEvent) => void,
  quantization: LoadQuantization = null
): Promise<Model> {
  const res = await apiFetch(`/api/models/${encodeURIComponent(modelId)}/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dtype, quantization }),
  });
  for await (const event of parseEventStream<LoadEventWire>(res)) {
    if ("cancelled" in event) throw new LoadCancelledError(modelId);
    if ("error" in event) throw new Error(event.error);
    if ("phase" in event) {
      onProgress?.({ phase: event.phase, detail: event.desc, current: event.n, total: event.total });
      continue;
    }
    return event.graph;
  }
  throw new Error("Load stream ended without a result.");
}

/** Stops an in-flight loadModel() call for modelId — see apps/api's ModelRegistry.request_cancel_load/LoadCancelled for how that actually interrupts from_pretrained. The open loadModel() promise rejects with a LoadCancelledError shortly after this resolves. */
export async function cancelLoad(modelId: string): Promise<void> {
  await apiFetch(`/api/models/${encodeURIComponent(modelId)}/load/cancel`, { method: "POST" });
}

export async function unloadModel(modelId: string): Promise<void> {
  await apiFetch(`/api/models/${encodeURIComponent(modelId)}/unload`, { method: "POST" });
}

/** Unloads (if resident) and permanently removes a downloaded model's files from data/models/ — the catalog's "delete" action. */
export async function deleteModel(modelId: string): Promise<void> {
  await apiFetch(`/api/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
}

export async function getGraph(modelId: string): Promise<Model> {
  const res = await apiFetch(`/api/models/${encodeURIComponent(modelId)}/graph`);
  return res.json();
}

/**
 * Fetches one (optionally windowed) weight tensor. `slice.ranges` is
 * per-dimension [start, end) in the tensor's own coordinate space — see
 * apps/web/src/tensor.ts's composeSlice, which is what actually produces
 * these ranges for the Tensor Explorer's windowing.
 *
 * Transport is a raw float32 buffer with shape/dtype as response headers
 * (PLAN.md §4.4), not a JSON number array — real weight matrices are far
 * too large for that to scale.
 */
export async function getWeightTensor(modelId: string, parameterId: string, slice?: TensorSlice): Promise<Tensor> {
  const qs = slice?.ranges ? `?ranges=${encodeURIComponent(JSON.stringify(slice.ranges))}` : "";
  const res = await apiFetch(`/api/models/${encodeURIComponent(modelId)}/tensors/${encodeURIComponent(parameterId)}${qs}`);
  const shape = JSON.parse(res.headers.get("X-Tensor-Shape") ?? "[]") as number[];
  const sourceDtype = res.headers.get("X-Tensor-Source-Dtype") ?? "F32";
  const buf = await res.arrayBuffer();
  const f32 = new Float32Array(buf);
  // model-ir's Tensor.data is always a Float64Array regardless of source
  // dtype (see its doc comment) — widen once here so every consumer
  // downstream (Heatmap, Histogram, nn-ops' tensorToMatrix, ...) needs no
  // special case for a backend-sourced tensor vs. a browser-parsed one.
  const data = new Float64Array(f32.length);
  data.set(f32);
  return { shape, dtype: sourceDtype, data };
}

interface BulkRunTensorEntry {
  name: string;
  shape: number[];
  dtype: string;
  byteOffset: number;
  byteLength: number;
}

interface BulkRunHeader {
  runId: string;
  tokenIds: number[];
  tokens: string[];
  tensors: BulkRunTensorEntry[];
}

/**
 * Decodes apps/api's encode_bulk_run format: a 4-byte little-endian header
 * length, a JSON header (tensor names/shapes/byte offsets), then every
 * tensor's raw float32 bytes back to back — one request for a whole run's
 * activations/attention/logits, chosen over per-node lazy fetching for the
 * reason tensors.py's encode_bulk_run doc comment explains.
 */
function decodeBulkRun(buf: ArrayBuffer): ActivationCapture {
  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen))) as BulkRunHeader;
  const dataStart = 4 + headerLen;

  const activations: Record<string, Tensor> = {};
  const attentionWeights: Record<string, Tensor> = {};
  const routerWeights: Record<string, Tensor> = {};
  const expertAssignment: Record<string, Tensor> = {};
  let logits: Tensor | undefined;

  for (const entry of header.tensors) {
    const start = dataStart + entry.byteOffset;
    // A plain view (`new Float32Array(buf, start, ...)`) would require
    // `start` to be 4-byte aligned, which the variable-length JSON header
    // in front of it doesn't guarantee — `.slice()` copies into a
    // fresh (always-aligned) buffer instead. Widening to Float64Array
    // below is already a full copy regardless (model-ir's Tensor.data
    // contract), so this isn't an extra asymptotic cost.
    const f32 = new Float32Array(buf.slice(start, start + entry.byteLength));
    const data = new Float64Array(f32.length);
    data.set(f32);
    const tensor: Tensor = { shape: entry.shape, dtype: entry.dtype, data };

    if (entry.name === "logits") logits = tensor;
    else if (entry.name.startsWith("activations/")) activations[entry.name.slice("activations/".length)] = tensor;
    else if (entry.name.startsWith("attentionWeights/")) attentionWeights[entry.name.slice("attentionWeights/".length)] = tensor;
    else if (entry.name.startsWith("routerWeights/")) routerWeights[entry.name.slice("routerWeights/".length)] = tensor;
    else if (entry.name.startsWith("expertAssignment/")) expertAssignment[entry.name.slice("expertAssignment/".length)] = tensor;
  }

  if (!logits) throw new Error("Run response had no logits tensor.");
  // lm_head's activation IS the logits — the backend sends it once (see
  // encode_bulk_run) to avoid duplicating a vocab-sized tensor; restore the
  // alias here so activations["lm_head"] still works like any other node.
  activations["lm_head"] = logits;

  return { tokenIds: header.tokenIds, tokens: header.tokens, activations, attentionWeights, routerWeights, expertAssignment, logits };
}

/**
 * `JSON.stringify` doesn't serialize typed arrays as plain arrays — a
 * Float64Array field would come out as `{"0":1,"1":2,...}`, which the
 * backend can't parse as tensor data. `replacementValue.data` is the only
 * place a Tensor (and therefore a Float64Array) appears inside an
 * Intervention, so this only needs to widen that one field.
 */
function serializeIntervention(iv: Intervention): unknown {
  if (!iv.replacementValue) return iv;
  return { ...iv, replacementValue: { ...iv.replacementValue, data: Array.from(iv.replacementValue.data) } };
}

/** Runs a real forward pass on the backend and returns the full captured result — see decodeBulkRun for the wire format. */
export async function runInference(modelId: string, tokenIds: number[], interventions?: Intervention[]): Promise<ActivationCapture> {
  const res = await apiFetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modelId, tokenIds, interventions: interventions?.length ? interventions.map(serializeIntervention) : undefined }),
  });
  return decodeBulkRun(await res.arrayBuffer());
}

/**
 * Scores many independent intervention variants against one target
 * sequence position in a handful of batched forward passes (see apps/api's
 * inference.py's run_attribution_sweep) instead of one request per
 * variant — the backend for ModelAdapter.runAttributionSweep.
 */
export async function runAttributionSweep(modelId: string, tokenIds: number[], predictIndex: number, variants: Intervention[][]): Promise<Float64Array[]> {
  const res = await apiFetch("/api/attribution", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modelId, tokenIds, predictIndex, variants: variants.map((v) => v.map(serializeIntervention)) }),
  });
  const [numVariants, vocab] = JSON.parse(res.headers.get("X-Tensor-Shape") ?? "[0,0]") as [number, number];
  const f32 = new Float32Array(await res.arrayBuffer());
  const rows: Float64Array[] = [];
  for (let i = 0; i < numVariants; i++) {
    const row = new Float64Array(vocab);
    row.set(f32.subarray(i * vocab, (i + 1) * vocab));
    rows.push(row);
  }
  return rows;
}

export interface GenerationOptions {
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
}

export interface GenerationToken {
  tokenId: number;
  text: string;
  done: boolean;
}

export interface GenerationError {
  error: string;
}

export type GenerationEvent = GenerationToken | GenerationError;

export function isGenerationError(event: GenerationEvent): event is GenerationError {
  return "error" in event;
}

/**
 * Parses a `text/event-stream` response body into JSON-decoded events —
 * shared by streamGeneration and streamDownload, the two SSE endpoints
 * (plain `data: {...}\n\n` framing; neither needs sse-starlette's extra
 * machinery server-side, so this doesn't need anything beyond it either).
 * A plain `EventSource` can't POST a body, which both of these need
 * (tokenIds, repo/revision), hence reading the stream by hand via fetch.
 */
async function* parseEventStream<T>(res: Response): AsyncGenerator<T> {
  if (!res.body) throw new Error("Response had no body (streaming isn't supported in this environment).");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const dataLine = chunk.split("\n").find((l) => l.startsWith("data:"));
      if (!dataLine) continue;
      yield JSON.parse(dataLine.slice(5).trim()) as T;
    }
  }
}

/**
 * Streams real multi-token generation (apps/api's /api/generate, SSE) one
 * token at a time as an async generator, using the model's real KV cache
 * server-side. `signal` lets a caller stop mid-stream (e.g. a "Stop"
 * button) — the backend's forward-pass loop keeps running server-side
 * until its next check-in, but the client stream ends immediately either
 * way, matching abort semantics anywhere else `fetch` is used.
 */
export async function* streamGeneration(modelId: string, tokenIds: number[], options: GenerationOptions = {}, signal?: AbortSignal): AsyncGenerator<GenerationEvent> {
  const res = await apiFetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modelId, tokenIds, ...options }),
    signal,
  });
  yield* parseEventStream<GenerationEvent>(res);
}

/** Which kind of file the aggregate byte counter below is currently attributed to — see apps/api's downloads.py's _non_weight_bytes doc comment for how this is derived (a byte threshold, not a literal per-file name — huggingface_hub's own progress hooks don't expose one). Absent entirely when the backend couldn't classify it (e.g. a metadata lookup failure), in which case the UI falls back to a generic "downloading" label. */
export type DownloadPhase = "downloading_config" | "downloading_weights";

export interface DownloadProgress {
  downloadedBytes: number;
  totalBytes: number;
  phase?: DownloadPhase;
}

export interface DownloadDone {
  done: true;
  manifest: CatalogEntry;
}

/** Backs both a user-initiated Cancel and a Pause (see cancelDownload) — the stream just ends here either way; which one it was is purely a frontend UI distinction (Cancel resets the form, Pause keeps the repo id around for Resume). */
export interface DownloadCancelledEvent {
  cancelled: true;
}

export interface DownloadError {
  error: string;
}

export type DownloadEvent = DownloadProgress | DownloadDone | DownloadCancelledEvent | DownloadError;

export function isDownloadDone(event: DownloadEvent): event is DownloadDone {
  return "done" in event && event.done === true;
}

export function isDownloadCancelledEvent(event: DownloadEvent): event is DownloadCancelledEvent {
  return "cancelled" in event && event.cancelled === true;
}

export function isDownloadError(event: DownloadEvent): event is DownloadError {
  return "error" in event;
}

/**
 * Streams download progress (apps/api's /api/models/download, SSE) for
 * pulling a new model from Hugging Face into data/models/ — a no-op fast
 * path server-side (one immediate `done` event) if it's already there.
 */
export async function* streamDownload(repo: string, revision?: string): AsyncGenerator<DownloadEvent> {
  const res = await apiFetch("/api/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo, revision: revision ?? "main" }),
  });
  yield* parseEventStream<DownloadEvent>(res);
}

/**
 * Backs both Pause and Cancel — huggingface_hub always discards a file's
 * partial bytes on an interrupted download, so there's no lower-level
 * distinction between "stop for good" and "stop for now" to make server
 * side (see apps/api's downloads.py DownloadCancelled doc comment). The
 * open streamDownload() call for this repo ends with a
 * DownloadCancelledEvent shortly after this resolves; Resume is just
 * calling streamDownload(repo) again, which naturally continues (every
 * already-finished file is skipped server-side, only the one actively
 * transferring when this was called has to restart).
 */
export async function cancelDownload(repo: string): Promise<void> {
  await apiFetch("/api/models/download/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
}

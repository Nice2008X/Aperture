"""FastAPI app.

Phase 1 (PLAN.md §8.1): catalog, load/unload, graph, and windowed
weight-tensor endpoints.
Phase 2 (PLAN.md §8.2): real forward pass + activation/attention capture,
lazily fetched per node exactly like weights — see tensors.py's doc comment
for why nothing here ever ships a full tensor inline in a JSON response.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .downloads import check_model_support, download_with_progress, request_cancel
from .generation import generate_tokens
from .inference import run_attribution_sweep, run_forward
from .model_registry import ModelRegistry, NoModelLoadedError
from .paths import MODELS_DIR
from .quantization import dequantize_weight
from .tensors import encode_bulk_run, encode_tensor

app = FastAPI(title="Aperture API")

# Permissive for local dev (frontend on a different Vite port than the API);
# tighten allow_origins before any non-local deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Tensor-Shape", "X-Tensor-Dtype", "X-Tensor-Source-Dtype"],
)

registry = ModelRegistry(MODELS_DIR)

# Read-only static access to data/models/<id>/*.json — currently just
# tokenizer.json, so the frontend's existing (already-correct, already
# handles every byte-level/SentencePiece scheme this app supports) BPE
# tokenizer implementation can load it directly instead of this backend
# reimplementing tokenization — see packages/tokenizer's "backend" branch.
if MODELS_DIR.exists():
    app.mount("/data/models", StaticFiles(directory=MODELS_DIR), name="models")


class LoadRequest(BaseModel):
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    quantization: Literal["4bit", "8bit"] | None = None


@app.get("/api/models")
async def list_models():
    return registry.catalog()


def _proc_meminfo() -> dict[str, int]:
    """Parses /proc/meminfo into a {label: bytes} dict — Linux-only, and
    silently empty everywhere else, which just skips the unified-memory
    correction below."""
    info: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            label, _, rest = line.partition(":")
            info[label] = int(rest.strip().split()[0]) * 1024  # "12345 kB" -> bytes
    except (OSError, ValueError, IndexError):
        pass
    return info


@app.get("/api/gpu")
async def gpu_status():
    """Live VRAM snapshot for the loader screen's fit-check (todo11.txt §6/§7)
    — cheap enough to poll on every catalog refresh. `mem_get_info` reports
    the *device's* free/total, not just what this process has allocated, so
    it reflects other processes sharing the GPU too. Returns available:false
    (rather than a 5xx) when there's no CUDA device, since the rest of the
    UI already degrades gracefully without GPU info."""
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    # This app's real deployment target (PLAN.md: "GPU: NVIDIA GB10, 130.6GB
    # total") is a unified-memory Grace Blackwell box where "GPU memory" IS
    # coherent system RAM — cudaMemGetInfo's totalBytes there matches
    # /proc/meminfo's MemTotal exactly. On that hardware, its "free" mirrors
    # MemFree: raw free pages, *not* counting reclaimable page cache the
    # kernel would happily evict for a real allocation — the same gap as
    # `free`'s "free" vs "available" columns. Left as mem_get_info reports
    # it, the status bar and every card's fit-check read tens of GB as
    # "used" that a real model load would in fact have available. Detected
    # by totalBytes matching host MemTotal (within rounding) — a discrete
    # GPU's dedicated VRAM total never coincides with host RAM, so this is
    # a no-op there.
    meminfo = _proc_meminfo()
    mem_total = meminfo.get("MemTotal")
    mem_available = meminfo.get("MemAvailable")
    if mem_total and mem_available and abs(total_bytes - mem_total) / mem_total < 0.05:
        free_bytes = max(free_bytes, min(mem_available, total_bytes))

    return {
        "available": True,
        "name": torch.cuda.get_device_name(device),
        "totalBytes": total_bytes,
        "freeBytes": free_bytes,
        "usedBytes": total_bytes - free_bytes,
    }


# Mirrors the frontend's SAFE_REPO_ID_RE (ModelLoader.tsx) — that one also
# strips a pasted huggingface.co URL down to "org/model" before it ever gets
# here, but this is the actual gate: `repo` flows into both a network
# request to the Hub and (model_id_for) a filesystem directory name, so a
# request that bypasses the frontend entirely still can't hand this
# anything other than a well-formed id.
REPO_ID_PATTERN = r"^[A-Za-z0-9]([A-Za-z0-9._-]{0,94}[A-Za-z0-9])?/[A-Za-z0-9]([A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$"


class DownloadRequest(BaseModel):
    repo: str = Field(pattern=REPO_ID_PATTERN)
    revision: str = "main"


@app.post("/api/models/check-support")
async def check_model_support_route(body: DownloadRequest):
    """Best-effort pre-download check for whether transformers'
    AutoModelForCausalLM actually knows body.repo's architecture (see
    downloads.py's check_model_support) — lets the frontend warn before
    spending bandwidth on a checkpoint this backend can't load (PLAN.md
    §11's needle2 case). Never raises: an inconclusive check (gated repo,
    bad revision, no model_type) comes back as supported: None rather
    than an error, since this is advisory only and shouldn't block a
    download on its own account."""
    return check_model_support(body.repo, body.revision)


@app.post("/api/models/download")
async def download_model_route(body: DownloadRequest):
    """Streams download progress as SSE (PLAN.md §8.6) — a no-op fast path
    if this repo already has a manifest (see download_with_progress), so
    re-selecting an already-downloaded model doesn't wait on a network
    round trip at all."""

    async def event_stream():
        async for event in download_with_progress(MODELS_DIR, body.repo, body.revision):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class CancelDownloadRequest(BaseModel):
    repo: str = Field(pattern=REPO_ID_PATTERN)


@app.post("/api/models/download/cancel")
async def cancel_download_route(body: CancelDownloadRequest):
    """Backs both Pause and Cancel in the UI — see DownloadCancelled's doc
    comment in downloads.py for why those are the same request here.
    request_cancel is a no-op (cancelled: False) if this repo has nothing
    in flight, e.g. a duplicate click or one that arrived just as the
    download finished on its own."""
    cancelled = request_cancel(body.repo)
    return {"cancelled": cancelled}


@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str, body: LoadRequest = LoadRequest()):
    """Streams weight-loading progress as SSE (mirrors /api/models/download
    and /api/generate) — a real ~6GB checkpoint takes tens of seconds to
    materialize onto the GPU, which the old plain-JSON version of this
    endpoint left the frontend with zero feedback for. See
    ModelRegistry.load_with_progress for where the progress numbers
    actually come from."""

    async def event_stream():
        async for event in registry.load_with_progress(model_id, dtype=body.dtype, quantization=body.quantization):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/models/{model_id}/load/cancel")
async def cancel_load_route(model_id: str):
    """Stops an in-flight /load — see ModelRegistry.request_cancel_load /
    LoadCancelled for how that actually interrupts from_pretrained."""
    cancelled = registry.request_cancel_load(model_id)
    return {"cancelled": cancelled}


@app.post("/api/models/{model_id}/unload")
async def unload_model(model_id: str):
    await registry.unload()
    return {"status": "unloaded"}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Unloads (if resident) and removes a downloaded model's directory from
    data/models/ entirely — the catalog's "delete" action, not to be
    confused with /unload which just frees the GPU and keeps the files."""
    try:
        await registry.delete(model_id)
    except FileNotFoundError:
        raise HTTPException(404, f"No model '{model_id}' in data/models.")
    return {"status": "deleted"}


@app.get("/api/models/{model_id}/graph")
async def get_graph(model_id: str):
    try:
        loaded = registry.require_current(model_id)
    except NoModelLoadedError as e:
        raise HTTPException(409, str(e) or f"Model '{model_id}' is not loaded — call /load first.")
    return loaded.ir.model_dump(by_alias=True)


@app.get("/api/models/{model_id}/tensors/{param_name}")
async def get_tensor(model_id: str, param_name: str, ranges: str | None = None):
    try:
        loaded = registry.require_current(model_id)
    except NoModelLoadedError as e:
        raise HTTPException(409, str(e) or f"Model '{model_id}' is not loaded — call /load first.")
    try:
        tensor = loaded.model.get_parameter(param_name)
    except AttributeError:
        raise HTTPException(404, f"Unknown parameter: {param_name}")

    # A quantized weight's real Parameter holds packed codes, not
    # meaningful floats — dequantize_weight returns None for anything
    # that isn't quantized (the overwhelmingly common case), in which
    # case `tensor` (already fetched above) is used as-is.
    dequantized = dequantize_weight(loaded.model, param_name)
    if dequantized is not None:
        tensor = dequantized

    try:
        parsed_ranges = json.loads(ranges) if ranges else None
        payload = encode_tensor(tensor, parsed_ranges)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"Invalid ranges: {e}")

    headers = {
        "X-Tensor-Shape": json.dumps(payload.shape),
        "X-Tensor-Dtype": "F32",
        "X-Tensor-Source-Dtype": payload.source_dtype,
    }
    return Response(content=payload.data, media_type="application/octet-stream", headers=headers)


# ---------------------------------------------------------------------------
# Runs — a real forward pass, cached server-side under a runId (so the lazy
# per-node endpoints below can still serve a specific tensor without
# re-running). POST /runs itself returns everything captured in one bulk
# binary response — see tensors.py's encode_bulk_run for why that's the
# right transport for this app's actual usage pattern (short interpretability
# probes) rather than either a giant JSON blob or hundreds of small requests.
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    modelId: str
    tokenIds: list[int]
    interventions: list[dict] | None = None


@app.post("/api/runs")
async def create_run(body: RunRequest):
    if not body.tokenIds:
        raise HTTPException(400, "tokenIds must be non-empty.")
    try:
        loaded = registry.require_current(body.modelId)
    except NoModelLoadedError as e:
        raise HTTPException(409, str(e) or f"Model '{body.modelId}' is not loaded — call /load first.")

    try:
        async with loaded.inference_lock:
            run_id = run_forward(loaded, body.tokenIds, body.interventions)
    except ValueError as e:
        raise HTTPException(400, str(e))
    activations, attention_weights, router_weights, expert_assignment = loaded.get_run(run_id)
    tokens = loaded.tokenizer.convert_ids_to_tokens(body.tokenIds) if loaded.tokenizer is not None else [f"#{t}" for t in body.tokenIds]

    body_bytes = encode_bulk_run(run_id, body.tokenIds, tokens, activations, attention_weights, router_weights, expert_assignment)
    return Response(content=body_bytes, media_type="application/octet-stream")


def _run_tensor(model_id: str, run_id: str, node_id: str, cache_attr: str, ranges: str | None) -> Response:
    try:
        loaded = registry.require_current(model_id)
    except NoModelLoadedError as e:
        raise HTTPException(409, str(e) or f"Model '{model_id}' is not loaded — call /load first.")
    try:
        activations, attention_weights, _router_weights, _expert_assignment = loaded.get_run(run_id)
    except KeyError:
        raise HTTPException(404, f"Unknown or expired run: {run_id} (only the last few runs are kept in memory).")
    cache = activations if cache_attr == "activations" else attention_weights
    tensor = cache.get(node_id)
    if tensor is None:
        raise HTTPException(404, f"No {cache_attr} captured for node: {node_id}")

    try:
        parsed_ranges = json.loads(ranges) if ranges else None
        payload = encode_tensor(tensor, parsed_ranges)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"Invalid ranges: {e}")

    headers = {
        "X-Tensor-Shape": json.dumps(payload.shape),
        "X-Tensor-Dtype": "F32",
        "X-Tensor-Source-Dtype": payload.source_dtype,
    }
    return Response(content=payload.data, media_type="application/octet-stream", headers=headers)


@app.get("/api/models/{model_id}/runs/{run_id}/activations/{node_id}")
async def get_run_activation(model_id: str, run_id: str, node_id: str, ranges: str | None = None):
    return _run_tensor(model_id, run_id, node_id, "activations", ranges)


@app.get("/api/models/{model_id}/runs/{run_id}/attention/{node_id}")
async def get_run_attention(model_id: str, run_id: str, node_id: str, ranges: str | None = None):
    return _run_tensor(model_id, run_id, node_id, "attention_weights", ranges)


@app.get("/api/models/{model_id}/runs/{run_id}/topk")
async def get_topk(model_id: str, run_id: str, tokenIndex: int, k: int = 5):
    try:
        loaded = registry.require_current(model_id)
    except NoModelLoadedError as e:
        raise HTTPException(409, str(e) or f"Model '{model_id}' is not loaded — call /load first.")
    try:
        activations, _, _, _ = loaded.get_run(run_id)
    except KeyError:
        raise HTTPException(404, f"Unknown or expired run: {run_id} (only the last few runs are kept in memory).")
    logits = activations.get("lm_head")
    if logits is None:
        raise HTTPException(404, "This run has no lm_head activation to compute predictions from.")

    seq_len = logits.shape[1]
    if not (0 <= tokenIndex < seq_len):
        raise HTTPException(400, f"tokenIndex {tokenIndex} out of range for sequence length {seq_len}.")

    row = logits[0, tokenIndex].float()
    probs = torch.softmax(row, dim=-1)
    top = torch.topk(probs, min(k, probs.shape[0]))
    return [{"tokenId": int(i), "prob": float(p)} for p, i in zip(top.values.tolist(), top.indices.tolist())]


# ---------------------------------------------------------------------------
# Attribution sweeps (phase 4, PLAN.md §8.4) — many independent intervention
# variants scored against one target prediction in a handful of batched
# forward passes instead of one real request per variant. See inference.py's
# run_attribution_sweep doc comment for the full rationale.
# ---------------------------------------------------------------------------


class AttributionSweepRequest(BaseModel):
    modelId: str
    tokenIds: list[int]
    predictIndex: int
    variants: list[list[dict]]


@app.post("/api/attribution")
async def attribution_sweep(body: AttributionSweepRequest):
    if not body.tokenIds:
        raise HTTPException(400, "tokenIds must be non-empty.")
    if not body.variants:
        raise HTTPException(400, "variants must be non-empty.")
    try:
        loaded = registry.require_current(body.modelId)
    except NoModelLoadedError as e:
        raise HTTPException(409, str(e) or f"Model '{body.modelId}' is not loaded — call /load first.")

    try:
        async with loaded.inference_lock:
            logits = run_attribution_sweep(loaded, body.tokenIds, body.predictIndex, body.variants)
    except ValueError as e:
        raise HTTPException(400, str(e))

    data = logits.contiguous().cpu().numpy().tobytes()
    headers = {"X-Tensor-Shape": json.dumps(list(logits.shape)), "X-Tensor-Dtype": "F32"}
    return Response(content=data, media_type="application/octet-stream", headers=headers)


# ---------------------------------------------------------------------------
# Generation (phase 5, PLAN.md §8.5) — real multi-token generation, streamed
# over SSE as each token is produced, using the model's real KV cache. See
# generation.py's module doc comment for why this still needs
# inference_lock even though it registers no hooks.
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    modelId: str
    tokenIds: list[int]
    maxNewTokens: int = 64
    temperature: float = 0.7
    topP: float = 1.0
    topK: int = 0


@app.post("/api/generate")
async def generate(body: GenerateRequest):
    if not body.tokenIds:
        raise HTTPException(400, "tokenIds must be non-empty.")
    try:
        loaded = registry.require_current(body.modelId)
    except NoModelLoadedError as e:
        raise HTTPException(409, str(e) or f"Model '{body.modelId}' is not loaded — call /load first.")

    max_new_tokens = min(max(1, body.maxNewTokens), 512)
    temperature = max(0.0, body.temperature)
    top_p = min(max(body.topP, 0.0), 1.0)
    top_k = max(0, body.topK)

    async def event_stream():
        try:
            async with loaded.inference_lock:
                for token_id, text, is_last in generate_tokens(loaded, body.tokenIds, max_new_tokens, temperature, top_p, top_k):
                    payload = json.dumps({"tokenId": token_id, "text": text, "done": is_last})
                    yield f"data: {payload}\n\n"
                    # Without this, the (synchronous, GPU-bound) loop above
                    # never actually hands control back to the event loop
                    # between tokens, so nothing gets flushed to the client
                    # until the whole generation finishes — defeating the
                    # point of streaming.
                    await asyncio.sleep(0)
        except Exception as e:  # noqa: BLE001 — headers are already sent once streaming starts, so an error has to become an SSE event, not an HTTP status
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

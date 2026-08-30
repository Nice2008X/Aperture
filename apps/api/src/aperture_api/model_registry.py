"""Model lifecycle: catalog data/models/, load one model onto the GPU at a
time, keep its built IR cached, unload cleanly before loading another.

Single-resident-model is the deliberate phase-1 choice (PLAN.md §4.1) — the
simplest correct behavior for one GPU. A second load call auto-unloads the
first rather than risking a silent OOM.
"""

from __future__ import annotations

import asyncio
import gc
import json
import queue
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.logging import set_tqdm_hook

from .graph_builder import AttentionHeadInfo, BuiltGraph, RopeDerivation, build_generic_model_ir
from .ir_types import Model as ModelIR

DType = Literal["bf16", "fp16", "fp32"]
_TORCH_DTYPES: dict[DType, torch.dtype] = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
Quantization = Literal["4bit", "8bit"] | None

MAX_CACHED_RUNS = 5


def _quantization_config(quantization: Quantization, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if quantization == "4bit":
        # NF4 (over plain int4) and double-quant (quantizing the
        # quantization constants themselves, a further ~0.4 bits/param)
        # are the standard "best quality per bit" defaults bitsandbytes
        # ships — no reason to expose weaker options here.
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


@dataclass
class LoadedModel:
    model_id: str
    model: torch.nn.Module
    tokenizer: object
    ir: ModelIR
    dtype: DType
    quantization: Quantization
    hook_modules: dict[str, torch.nn.Module]
    adds: list[tuple[str, str, str]]
    muls: list[tuple[str, str, str]]
    ropes: list[RopeDerivation]
    attention_heads: dict[str, AttentionHeadInfo]
    # moe_layer nodeId -> its router submodule — see graph_builder's
    # BuiltGraph.moe_routers doc comment for why this is separate from
    # hook_modules.
    moe_routers: dict[str, torch.nn.Module]
    # moe_layer nodeId -> its synthetic "combine" leaf nodeId — see
    # graph_builder's BuiltGraph.moe_combines doc comment.
    moe_combines: dict[str, str]
    # runId -> {nodeId: tensor}, evicted down to the most recent few runs so
    # repeated inspection (TensorExplorer/AttentionView opening different
    # nodes from the same run) doesn't need a fresh forward pass, without
    # unbounded GPU memory growth across many runs.
    activation_cache: dict[str, dict[str, torch.Tensor]]
    attention_cache: dict[str, dict[str, torch.Tensor]]
    # moe_layer nodeId -> tensor, same eviction policy as the two caches
    # above — router_weight is the softmax'd top-k gate weights
    # [seq, topK], expert_assignment the selected expert ids [seq, topK].
    router_weight_cache: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    expert_assignment_cache: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    run_order: list[str] = field(default_factory=list)
    # Forward passes mutate shared state (hooks registered on this model's
    # own modules) — two concurrent requests interleaving their hook
    # registration/removal on the same modules would corrupt each other's
    # results, not just slow each other down. Every route that calls
    # run_forward/run_attribution_sweep holds this for the duration.
    inference_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def store_run(
        self,
        run_id: str,
        activations: dict[str, torch.Tensor],
        attention_weights: dict[str, torch.Tensor],
        router_weights: dict[str, torch.Tensor] | None = None,
        expert_assignment: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.activation_cache[run_id] = activations
        self.attention_cache[run_id] = attention_weights
        self.router_weight_cache[run_id] = router_weights or {}
        self.expert_assignment_cache[run_id] = expert_assignment or {}
        self.run_order.append(run_id)
        while len(self.run_order) > MAX_CACHED_RUNS:
            oldest = self.run_order.pop(0)
            self.activation_cache.pop(oldest, None)
            self.attention_cache.pop(oldest, None)
            self.router_weight_cache.pop(oldest, None)
            self.expert_assignment_cache.pop(oldest, None)

    def get_run(self, run_id: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if run_id not in self.activation_cache:
            raise KeyError(run_id)
        return (
            self.activation_cache[run_id],
            self.attention_cache[run_id],
            self.router_weight_cache.get(run_id, {}),
            self.expert_assignment_cache.get(run_id, {}),
        )


class NoModelLoadedError(Exception):
    pass


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._loaded: LoadedModel | None = None
        self._lock = asyncio.Lock()

    def catalog(self) -> list[dict]:
        if not self.models_dir.exists():
            return []
        entries = []
        for d in sorted(self.models_dir.iterdir()):
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                entry = json.loads(manifest_path.read_text())
                entry["loaded"] = self._loaded is not None and self._loaded.model_id == entry["id"]
                # Exposed only for the resident entry — the frontend's
                # refresh-resume path (useModel's mount effect) needs to
                # know exactly what quantization is already on the GPU so
                # it can re-request the same thing and hit load_with_progress's
                # fast path, instead of mismatching and triggering a real reload.
                entry["loadedQuantization"] = self._loaded.quantization if entry["loaded"] else None
                entries.append(entry)
        return entries

    async def load_with_progress(self, model_id: str, dtype: DType = "bf16", quantization: Quantization = None) -> AsyncIterator[dict]:
        """Yields `{phase, desc, n, total}` progress events while loading
        (see the tqdm-hook comment below for where these actually come
        from), then exactly one final `{done: True, graph: {...}}` or
        `{error: "..."}`. A no-op fast path if this exact model, at this
        exact dtype/quantization, is already resident — matches the old
        `load()`'s early return, just as a single immediate event instead
        of a plain return value. Requesting the *same* model_id back with
        a *different* dtype/quantization is a real reload, not a fast
        path — those change what's actually materialized on the GPU.
        """
        async with self._lock:
            if self._loaded is not None and self._loaded.model_id == model_id and self._loaded.dtype == dtype and self._loaded.quantization == quantization:
                yield {"done": True, "graph": self._loaded.ir.model_dump(by_alias=True)}
                return

            path = self.models_dir / model_id
            manifest_path = path / "manifest.json"
            if not manifest_path.exists():
                yield {"error": f"No model '{model_id}' in data/models — download it first."}
                return
            manifest = json.loads(manifest_path.read_text())

            self._unload_locked()

            progress_queue: queue.Queue = queue.Queue()

            # transformers' from_pretrained has no tqdm_class-style
            # parameter, but exposes an equivalent (and, unlike
            # huggingface_hub's, never TTY-disabled — see downloads.py's
            # TQDM_POSITION comment for that fight) first-class hook:
            # every tqdm it creates during loading (in practice, one bar
            # tracking "Loading weights" over every real parameter) is
            # built by calling this hook instead, so wrapping its
            # .update() is enough to mirror every advance into our queue.
            def hook(factory, args, kwargs):
                bar = factory(*args, **kwargs)
                real_update = bar.update

                def update(n: int = 1) -> None:
                    real_update(n)
                    progress_queue.put({"phase": "loading_weights", "desc": str(getattr(bar, "desc", "") or ""), "n": bar.n, "total": bar.total})

                bar.update = update
                return bar

            torch_dtype = _TORCH_DTYPES[dtype]
            quant_config = _quantization_config(quantization, torch_dtype)

            def run() -> torch.nn.Module:
                previous_hook = set_tqdm_hook(hook)
                try:
                    # "eager" (not the sdpa/flash-attention default) is what
                    # makes output_attentions=True actually return real
                    # per-head softmax weights instead of None — the whole
                    # point of AttentionView. Slower than sdpa, but this app
                    # is for inspection, not throughput, and prompts are short.
                    # quantization_config=None (the common case) is a no-op —
                    # from_pretrained loads at `dtype` exactly as before
                    # PLAN.md §8.7 added this parameter.
                    m = AutoModelForCausalLM.from_pretrained(
                        path, dtype=torch_dtype, device_map="cuda", attn_implementation="eager", quantization_config=quant_config
                    )
                    m.eval()
                    return m
                finally:
                    set_tqdm_hook(previous_hook)

            task = asyncio.create_task(asyncio.to_thread(run))
            try:
                while True:
                    try:
                        yield progress_queue.get_nowait()
                    except queue.Empty:
                        if task.done():
                            break
                        await asyncio.sleep(0.1)
                model = await task
            except Exception as e:  # noqa: BLE001 — a bad checkpoint, OOM, etc. all become one clear event
                yield {"error": str(e)}
                return

            yield {"phase": "building_graph"}
            tokenizer = AutoTokenizer.from_pretrained(path)
            built: BuiltGraph = build_generic_model_ir(model, model_id, manifest.get("displayName", model_id))

            self._loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=tokenizer,
                ir=built.model,
                dtype=dtype,
                quantization=quantization,
                hook_modules=built.hook_modules,
                adds=built.adds,
                muls=built.muls,
                ropes=built.ropes,
                attention_heads=built.attention_heads,
                moe_routers=built.moe_routers,
                moe_combines=built.moe_combines,
                activation_cache={},
                attention_cache={},
                run_order=[],
            )
            yield {"done": True, "graph": built.model.model_dump(by_alias=True)}

    async def unload(self) -> None:
        async with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self._loaded is None:
            return
        self._loaded = None
        gc.collect()
        torch.cuda.empty_cache()

    def require_current(self, model_id: str | None = None) -> LoadedModel:
        if self._loaded is None:
            raise NoModelLoadedError()
        if model_id is not None and self._loaded.model_id != model_id:
            raise NoModelLoadedError(f"Loaded model is {self._loaded.model_id}, not {model_id}.")
        return self._loaded

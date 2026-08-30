"""Binary tensor transport — PLAN.md §4.4: never ship a full weight/activation
tensor as a JSON number array.

Windowing matches `TensorSlice` in packages/model-ir exactly (per-dimension
[start, end) ranges in the tensor's own coordinate space — see
apps/web/src/tensor.ts's composeSlice/defaultWindow, which is what
TensorExplorer actually sends): the caller gets back only the requested
sub-block, with its *windowed* shape as a response header and a raw float32
buffer as the body.
"""

from __future__ import annotations

import json
import struct

import torch

from .graph_builder import dtype_to_str


class TensorPayload:
    __slots__ = ("data", "shape", "source_dtype")

    def __init__(self, data: bytes, shape: list[int], source_dtype: str):
        self.data = data
        self.shape = shape
        self.source_dtype = source_dtype


def encode_tensor(tensor: torch.Tensor, ranges: list[dict[str, int]] | None = None) -> TensorPayload:
    if ranges is not None:
        if len(ranges) != tensor.dim():
            raise ValueError(f"ranges has {len(ranges)} entries but tensor has {tensor.dim()} dimensions")
        index = tuple(slice(max(0, r["start"]), min(tensor.shape[i], r["end"])) for i, r in enumerate(ranges))
        windowed = tensor[index]
        # Python slice objects (unlike integer indexing) never drop a
        # dimension, so narrowing one dim to a single index — e.g. one
        # expert's row out of a fused MoE weight's [numExperts, out, in]
        # storage, see graph_builder's _expert_param_ref — still leaves a
        # size-1 dim in the result. Squeeze away only the dims the caller
        # actually narrowed to size 1 (never a dim that was already size 1
        # in the source tensor) so a single-expert window comes back as
        # the plain 2D [out, in] matrix Heatmap/Matrix expect, matching
        # what the caller's single-index range meant.
        for dim in reversed(range(windowed.dim())):
            if windowed.shape[dim] == 1 and tensor.shape[dim] != 1:
                windowed = windowed.squeeze(dim)
    else:
        windowed = tensor

    shape = list(windowed.shape)
    # Always transported as float32 regardless of source dtype (bf16/fp16 have
    # no native JS typed-array counterpart worth exposing over the wire) —
    # mirrors model-ir's Tensor.data always being a Float64Array client-side
    # today, just float32 to halve payload size.
    data = windowed.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return TensorPayload(data=data, shape=shape, source_dtype=dtype_to_str(tensor.dtype))


def _squeeze_batch(t: torch.Tensor) -> torch.Tensor:
    """Every captured activation carries a real batch dim (always 1 — this
    app only ever runs one prompt through the model at a time); the
    frontend's Tensor/Matrix conventions (see apps/web's nn-ops Matrix,
    [sequence][dim], no batch axis) predate the backend and don't expect
    one, so it's dropped once here rather than in every consumer."""
    return t.squeeze(0) if t.dim() >= 1 and t.shape[0] == 1 else t


def encode_bulk_run(
    run_id: str,
    token_ids: list[int],
    tokens: list[str],
    activations: dict[str, torch.Tensor],
    attention_weights: dict[str, torch.Tensor],
    router_weights: dict[str, torch.Tensor] | None = None,
    expert_assignment: dict[str, torch.Tensor] | None = None,
) -> bytes:
    """Packs an entire run's captured tensors into one response: a
    length-prefixed JSON header (names/shapes/byte offsets) followed by
    every tensor's raw float32 bytes back to back.

    One request instead of one-per-node (hundreds, for a real model) or one
    giant JSON blob (a large-vocab model's logits alone can run tens of MB
    as JSON text) — see PLAN.md §9 for why this is deliberately *not* the
    fully-lazy per-node design used for weight tensors: real usage here is
    short interpretability probes, not long generations, so one bulk
    transfer is the simpler correct choice today. `GET .../runs/{id}/
    activations/{nodeId}` (tensors.py's encode_tensor, unused by this bulk
    path) stays available if a future generation-scale use case needs true
    per-node laziness for runs too.
    """
    entries: list[dict] = []
    blobs: list[bytes] = []
    offset = 0

    def add(name: str, t: torch.Tensor, squeeze: bool = True) -> None:
        nonlocal offset
        squeezed = _squeeze_batch(t) if squeeze else t
        data = squeezed.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
        entries.append({"name": name, "shape": list(squeezed.shape), "dtype": dtype_to_str(t.dtype), "byteOffset": offset, "byteLength": len(data)})
        blobs.append(data)
        offset += len(data)

    for node_id, t in activations.items():
        # lm_head's activation IS the logits — sent once below as "logits"
        # rather than duplicating a vocab-sized tensor under two names.
        if node_id == "lm_head":
            continue
        add(f"activations/{node_id}", t)
    for node_id, t in attention_weights.items():
        add(f"attentionWeights/{node_id}", t)
    # Router outputs never carry a batch dim to begin with (see
    # inference.py's make_router_hook doc comment — the real forward
    # already flattens batch*seq) — squeezing here would incorrectly drop
    # a genuine seq_len==1 prompt's leading dimension.
    for node_id, t in (router_weights or {}).items():
        add(f"routerWeights/{node_id}", t, squeeze=False)
    for node_id, t in (expert_assignment or {}).items():
        add(f"expertAssignment/{node_id}", t, squeeze=False)
    if "lm_head" in activations:
        add("logits", activations["lm_head"])

    header = json.dumps({"runId": run_id, "tokenIds": token_ids, "tokens": tokens, "tensors": entries}).encode("utf-8")
    return struct.pack("<I", len(header)) + header + b"".join(blobs)

"""Runs a real forward pass over a loaded model, capturing every
module-backed IR node's activation via a forward hook and resolving the
handful of synthesized (non-module) nodes — residual adds, the gated-FFN's
elementwise multiply, RoPE's display value — from those real captures.
See graph_builder.py's BuiltGraph for where hook_modules/adds/muls/ropes
come from, and PLAN.md §4.3 for the design this implements.

Interventions (phase 3, PLAN.md §8.3) piggyback on the same hooks: a forward
hook that returns a value makes PyTorch use it as the module's actual
output, so mutating a targeted node's tensor there is a real re-execution —
everything downstream (captured by hooks on later modules) sees the edited
value, not a cosmetic overlay on the original run's numbers. This is the
direct backend analog of packages/nn-ops/intervene.ts's applyInterventions/
applyHeadIntervention, which did the same thing by construction (each step
was just the next line of one synchronous JS function).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import torch

from .graph_builder import RopeDerivation
from .model_registry import LoadedModel


@dataclass
class InterventionSpec:
    node_id: str
    operation: str
    token_index: int | None
    head_index: int | None
    scale: float | None
    replacement: dict | None  # {"shape": [...], "dtype": "...", "data": [...]} — model-ir's Tensor shape


def _parse_interventions(raw: list[dict]) -> list[InterventionSpec]:
    return [
        InterventionSpec(
            node_id=r["nodeId"],
            operation=r["operation"],
            token_index=r.get("tokenIndex"),
            head_index=r.get("headIndex"),
            scale=r.get("scale"),
            replacement=r.get("replacementValue"),
        )
        for r in raw
    ]


def _unwrap(output: object) -> torch.Tensor:
    """Most hooked modules return a plain tensor; HF's Attention modules
    return (attn_output, attn_weights, ...) — the node's activation is
    always the first element."""
    return output[0] if isinstance(output, tuple) else output  # type: ignore[return-value]


def _apply_rope_reference(q: torch.Tensor, rope: RopeDerivation) -> torch.Tensor:
    """rotate_half-convention RoPE applied to a captured pre-rope Q, purely
    for display (see RopeDerivation's doc comment) — the same formula
    apps/web's nn-ops (ropeCosSin/applyRopeToHead/rotateHalf) implements
    client-side, ported to torch. q: [batch, seq, numHeads*headDim].
    """
    batch, seq, _ = q.shape
    head_dim = rope.head_dim
    num_heads = rope.num_heads
    half = head_dim // 2
    calc_dtype = torch.float32

    inv_freq = 1.0 / (rope.rope_theta ** (torch.arange(0, half, device=q.device, dtype=calc_dtype) * 2 / head_dim))
    positions = torch.arange(seq, device=q.device, dtype=calc_dtype)
    freqs = torch.outer(positions, inv_freq)  # [seq, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, headDim]
    cos = emb.cos()[None, :, None, :]  # [1, seq, 1, headDim]
    sin = emb.sin()[None, :, None, :]

    qh = q.to(calc_dtype).view(batch, seq, num_heads, head_dim)
    x1, x2 = qh[..., :half], qh[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)

    roped = qh * cos + rotated * sin
    return roped.reshape(batch, seq, num_heads * head_dim).to(q.dtype)


def _apply_edit(tensor: torch.Tensor, row_specs: list[tuple[int, InterventionSpec]]) -> torch.Tensor:
    """tensor: [batch, seq, hidden]. `row_specs` is (batchRow, spec) pairs —
    a plain single-prompt run (run_forward) always uses batchRow 0 for
    everything; a batched attribution sweep (run_attribution_sweep) gives
    each independent variant its own row, so N variants cost one forward
    pass instead of N. Mirrors applyInterventions in
    packages/nn-ops/intervene.ts row-for-row (a "row" there is a token
    position; here it's (batch row, token position))."""
    modified = tensor.clone()
    seq_len = modified.shape[1]
    for batch_row, spec in row_specs:
        positions = range(seq_len) if spec.token_index is None else [spec.token_index]
        if spec.operation == "zero":
            for p in positions:
                if 0 <= p < seq_len:
                    modified[batch_row, p, :] = 0
        elif spec.operation == "scale":
            s = spec.scale if spec.scale is not None else 1.0
            for p in positions:
                if 0 <= p < seq_len:
                    modified[batch_row, p, :] *= s
        elif spec.operation == "replace" and spec.replacement is not None:
            rep_shape = spec.replacement["shape"]
            rep = torch.tensor(spec.replacement["data"], dtype=torch.float32, device=modified.device).reshape(rep_shape)
            for p in positions:
                if 0 <= p < seq_len and p < rep_shape[0]:
                    modified[batch_row, p, :] = rep[p].to(modified.dtype)
    return modified


def _make_zero_head_pre_hook(row_specs: list[tuple[int, InterventionSpec]], head_dim: int):
    """Registered on an attention's o_proj as a *pre*-hook — see
    graph_builder's AttentionHeadInfo doc comment for why this, not the
    attention container's own (post-o_proj) hook, is the point where "head
    3" is still a well-defined column slice (mirrors nn-ops'
    applyHeadIntervention exactly, just intercepted at the real o_proj
    call instead of a hand-written one). `row_specs` — see _apply_edit."""

    def pre_hook(_module: torch.nn.Module, args: tuple):
        x = args[0].clone()
        seq_len = x.shape[1]
        for batch_row, spec in row_specs:
            if spec.head_index is None:
                continue
            start = spec.head_index * head_dim
            end = min(x.shape[-1], start + head_dim)
            positions = range(seq_len) if spec.token_index is None else [spec.token_index]
            for p in positions:
                if 0 <= p < seq_len:
                    x[batch_row, p, start:end] = 0
        return (x,) + args[1:]

    return pre_hook


def run_forward(loaded: LoadedModel, token_ids: list[int], interventions: list[dict] | None = None) -> str:
    """Runs one real forward pass (optionally with `interventions` applied
    as each targeted node's activation is computed, so everything
    downstream sees the edited value), caches its activations/attention
    weights on `loaded` under a fresh run id, and returns that id.

    Raises ValueError (the caller turns this into a 400) if an intervention
    targets a node with no real hook to attach to — a derived node like a
    residual add or the gated-FFN's elementwise multiply is never a
    distinct step in the real forward pass (see graph_builder's adds/muls
    doc comments), so there's nowhere causally correct to apply an edit.
    """
    specs = _parse_interventions(interventions or [])

    by_node: dict[str, list[InterventionSpec]] = {}
    zero_head_by_attn: dict[str, list[InterventionSpec]] = {}
    for spec in specs:
        if spec.operation == "zero_head":
            if spec.node_id not in loaded.attention_heads:
                raise ValueError(f"Node '{spec.node_id}' has no attention heads to intervene on — it isn't an attention node with a real output projection.")
            zero_head_by_attn.setdefault(spec.node_id, []).append(spec)
        else:
            if spec.node_id not in loaded.hook_modules:
                raise ValueError(
                    f"Node '{spec.node_id}' isn't a distinct step in the real forward pass (it's a derived/synthetic value reconstructed for "
                    "display, e.g. a residual add or an elementwise multiply) — intervene on a neighboring real node instead."
                )
            by_node.setdefault(spec.node_id, []).append(spec)

    activations: dict[str, torch.Tensor] = {}
    router_weights: dict[str, torch.Tensor] = {}
    expert_assignment: dict[str, torch.Tensor] = {}
    handles = []

    def make_record_hook(node_id: str):
        def hook(_module: torch.nn.Module, _inputs: object, output: object) -> None:
            activations[node_id] = _unwrap(output).detach()

        return hook

    def make_router_hook(moe_node_id: str):
        # A router's real forward returns (logits, topKWeights, topKIndices)
        # with no batch dim (see graph_builder's _build_moe doc comment) —
        # deliberately never registered in hook_modules, so it can't be
        # targeted by an edit hook that would misapply _apply_edit's
        # [batch, seq, hidden] shape assumptions to this different layout.
        # Record-only, capture-side; keyed by the *moe_layer*'s node id
        # (not the router child's) to match ActivationCapture's
        # routerWeights/expertAssignment side channels (PLAN.md §5).
        def hook(_module: torch.nn.Module, _inputs: object, output: object) -> None:
            logits, scores, indices = output
            activations[f"{moe_node_id}.router"] = logits.detach()
            router_weights[moe_node_id] = scores.detach()
            expert_assignment[moe_node_id] = indices.detach()

        return hook

    def make_edit_hook(node_id: str, node_specs: list[InterventionSpec]):
        row_specs = [(0, spec) for spec in node_specs]  # single-prompt run: everything is batch row 0

        def hook(_module: torch.nn.Module, _inputs: object, output: object):
            edited = _apply_edit(_unwrap(output), row_specs)
            activations[node_id] = edited.detach()
            return (edited,) + output[1:] if isinstance(output, tuple) else edited

        return hook

    for node_id, module in loaded.hook_modules.items():
        if node_id in by_node:
            handles.append(module.register_forward_hook(make_edit_hook(node_id, by_node[node_id])))
        else:
            handles.append(module.register_forward_hook(make_record_hook(node_id)))

    for attn_node_id, head_specs in zero_head_by_attn.items():
        info = loaded.attention_heads[attn_node_id]
        row_specs = [(0, spec) for spec in head_specs]
        handles.append(info.o_proj.register_forward_pre_hook(_make_zero_head_pre_hook(row_specs, info.head_dim)))

    for moe_node_id, router_module in loaded.moe_routers.items():
        handles.append(router_module.register_forward_hook(make_router_hook(moe_node_id)))

    try:
        ids_tensor = torch.tensor([token_ids], device=loaded.model.device)
        # The graph's "input" node (graph_builder's Input tokens) has no
        # real nn.Module to hook — it's the raw token-id sequence the model
        # call is given, not a module's output — so without this it would
        # never appear in `activations` at all, and Token Embedding's own
        # "Input" tab would wrongly read as nothing having been captured.
        activations["input"] = ids_tensor.detach()
        with torch.no_grad():
            out = loaded.model(ids_tensor, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # A moe_layer's "combine" leaf (see graph_builder's _build_moe doc
    # comment) has no real hook of its own — nothing distinct computes
    # "the routed experts' combined output" as a separate step — but the
    # moe_layer container's own hook already captured exactly that value,
    # so this just exposes it under the leaf's id too. Must run before the
    # `adds` loop below: a moe FFN's res2 residual-add depends on this
    # leaf id, not the container's.
    for moe_node_id, combine_id in loaded.moe_combines.items():
        if moe_node_id in activations:
            activations[combine_id] = activations[moe_node_id]

    # Derived nodes, resolved in the order graph_builder recorded them —
    # that order already matches real data-flow dependency (e.g. a layer's
    # res1 add is appended right after its two operands are captured), so a
    # single ordered pass is enough; no separate dependency resolution needed.
    for node_id, a_id, b_id in loaded.adds:
        if a_id in activations and b_id in activations:
            activations[node_id] = activations[a_id] + activations[b_id]
    for node_id, a_id, b_id in loaded.muls:
        if a_id in activations and b_id in activations:
            activations[node_id] = activations[a_id] * activations[b_id]
    for rope in loaded.ropes:
        if rope.q_node_id in activations:
            activations[rope.node_id] = _apply_rope_reference(activations[rope.q_node_id], rope)

    # Real per-head softmax attention weights (needs attn_implementation
    #="eager", forced at load time — see model_registry.py) — out.attentions
    # is ordered by layer, which is authoritative regardless of what each
    # architecture happens to name its attention class.
    attention_weights: dict[str, torch.Tensor] = {}
    if out.attentions is not None:
        for i, layer_attn in enumerate(out.attentions):
            if layer_attn is None:
                continue
            node_id = f"block.{i}.attn"
            if node_id in loaded.hook_modules:
                attention_weights[node_id] = layer_attn[0].detach()  # drop batch dim -> [numHeads, seq, seq]

    run_id = uuid.uuid4().hex[:12]
    loaded.store_run(run_id, activations, attention_weights, router_weights, expert_assignment)
    return run_id


# ---------------------------------------------------------------------------
# Batched attribution sweeps (phase 4, PLAN.md §8.4) — occlusion-based
# attribution (Token Attribution, Head Attribution) scores many independent
# "what if we intervened on X" variants against one target prediction. The
# naive approach — one runInference() per variant — is what
# packages/interpretability's computeTokenAttribution/computeHeadAttribution
# still do by default, and it's correct, just one real GPU forward pass +
# HTTP round trip per variant; fine for a tiny in-browser debug model where
# each pass was sub-millisecond, not for a real model. run_attribution_sweep
# gives each variant its own row in one batched forward pass instead — a
# few real passes (chunked to bound memory) instead of dozens-to-hundreds.
# ---------------------------------------------------------------------------

MAX_SWEEP_BATCH = 32


def run_attribution_sweep(loaded: LoadedModel, token_ids: list[int], predict_index: int, variants: list[list[dict]]) -> torch.Tensor:
    """variants[i] is the list of interventions applied only to variant i
    (an empty list is a valid variant — the baseline). Returns
    [numVariants, vocab]: just the predictIndex row of each variant's
    logits, since that's the only thing occlusion-based attribution ever
    reads from a full forward pass's output.
    """
    parsed = [_parse_interventions(v) for v in variants]
    if not (0 <= predict_index < len(token_ids)):
        raise ValueError(f"predictIndex {predict_index} out of range for a {len(token_ids)}-token prompt.")

    chunks: list[torch.Tensor] = []
    for start in range(0, len(parsed), MAX_SWEEP_BATCH):
        chunks.append(_run_variant_batch(loaded, token_ids, parsed[start : start + MAX_SWEEP_BATCH], predict_index))
    return torch.cat(chunks, dim=0)


def _run_variant_batch(loaded: LoadedModel, token_ids: list[int], variant_specs: list[list[InterventionSpec]], predict_index: int) -> torch.Tensor:
    batch_size = len(variant_specs)
    ids_tensor = torch.tensor([token_ids] * batch_size, device=loaded.model.device)

    by_node: dict[str, list[tuple[int, InterventionSpec]]] = {}
    zero_head_by_attn: dict[str, list[tuple[int, InterventionSpec]]] = {}
    for row, specs in enumerate(variant_specs):
        for spec in specs:
            if spec.operation == "zero_head":
                if spec.node_id not in loaded.attention_heads:
                    raise ValueError(f"Node '{spec.node_id}' has no attention heads to intervene on — it isn't an attention node with a real output projection.")
                zero_head_by_attn.setdefault(spec.node_id, []).append((row, spec))
            else:
                if spec.node_id not in loaded.hook_modules:
                    raise ValueError(
                        f"Node '{spec.node_id}' isn't a distinct step in the real forward pass (it's a derived/synthetic value reconstructed for "
                        "display, e.g. a residual add or an elementwise multiply) — intervene on a neighboring real node instead."
                    )
                by_node.setdefault(spec.node_id, []).append((row, spec))

    handles = []

    def make_hook(row_specs: list[tuple[int, InterventionSpec]]):
        def hook(_module: torch.nn.Module, _inputs: object, output: object):
            edited = _apply_edit(_unwrap(output), row_specs)
            return (edited,) + output[1:] if isinstance(output, tuple) else edited

        return hook

    for node_id, row_specs in by_node.items():
        handles.append(loaded.hook_modules[node_id].register_forward_hook(make_hook(row_specs)))
    for attn_node_id, row_specs in zero_head_by_attn.items():
        info = loaded.attention_heads[attn_node_id]
        handles.append(info.o_proj.register_forward_pre_hook(_make_zero_head_pre_hook(row_specs, info.head_dim)))

    try:
        with torch.no_grad():
            # output_attentions/use_cache both off: a sweep only ever reads
            # one logits row per variant, nothing else this run captures.
            out = loaded.model(ids_tensor, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return out.logits[:, predict_index, :].detach().to(torch.float32)

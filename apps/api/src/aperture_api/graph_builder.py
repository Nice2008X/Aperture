"""Generic Model IR builder — the phase-1 replacement for the 8 hand-written
TS adapters in packages/model-adapters/*.

Where the TS adapters had to hand-write every architecture's forward math
(no ML framework in the browser), here `transformers` has already loaded a
real nn.Module tree with real math. This module's only job is to *classify*
that tree into the shared Model IR (packages/model-ir): which submodule is
"the attention block", which Linear is Q vs K vs V, etc. — and to degrade
gracefully (emit a "custom" node) rather than crash when it meets a module
shape it doesn't recognize yet.

Scope note (see PLAN.md §10): this build only special-cases the "plain
dense, pre-norm, gated-FFN, RoPE" shape most current open models share
(Llama/Mistral/Qwen2/Qwen3/Gemma family) — fused QKV, sandwich norms, and
MoE blocks are later-phase additions to the classification rules below,
not a different architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .ir_types import Model, ModelConfig, ModelEdge, ModelNode, NodeType, ParameterRef, TensorSlice, TensorSpec
from .quantization import logical_shape_of, quantized_kind


@dataclass
class RopeDerivation:
    """Reproduces the roped Q shown at a synthesized "rope" node, purely for
    display — see resolve_derivations' rope branch for why this is a
    reference recomputation rather than an intercepted real value."""

    node_id: str
    q_node_id: str
    num_heads: int
    head_dim: int
    rope_theta: float


@dataclass
class AttentionHeadInfo:
    """Where a "zero_head" intervention on an attention node actually has to
    apply — see graph_builder's _build_attention doc comment and
    inference.py's zero_head handling for why "right before o_proj" is the
    only point a single head is still a well-defined tensor slice."""

    o_proj: nn.Module
    head_dim: int


@dataclass
class BuiltGraph:
    model: Model
    """nodeId -> the real nn.Module whose forward hook should populate that node's captured activation."""
    hook_modules: dict[str, nn.Module]
    """(nodeId, aNodeId, bNodeId) — nodeId's activation is a + b (residual adds, never their own hookable module)."""
    adds: list[tuple[str, str, str]] = field(default_factory=list)
    """(nodeId, aNodeId, bNodeId) — nodeId's activation is a * b (gated-FFN's elementwise gate×up)."""
    muls: list[tuple[str, str, str]] = field(default_factory=list)
    ropes: list[RopeDerivation] = field(default_factory=list)
    """attention container nodeId -> where to apply a zero_head intervention targeting it."""
    attention_heads: dict[str, AttentionHeadInfo] = field(default_factory=dict)
    """moe_layer nodeId -> its router submodule. Kept separate from hook_modules
    (see graph_builder's _build_moe doc comment): the router's real forward
    output is a 3-tuple with no batch dimension, so it needs its own
    dedicated capture-only hook in inference.py rather than the generic
    edit-capable one every hook_modules entry gets."""
    moe_routers: dict[str, nn.Module] = field(default_factory=dict)
    """moe_layer nodeId -> its synthetic "combine" leaf nodeId (see
    _build_moe) — inference.py aliases this leaf's activation from the
    moe_layer container's own real hooked value, since no single real
    module call computes "combined expert output" as a distinct step."""
    moe_combines: dict[str, str] = field(default_factory=dict)

_DTYPE_MAP = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def dtype_to_str(dtype: torch.dtype) -> str:
    return _DTYPE_MAP.get(dtype, "F32")


def _is_norm(cls_name: str) -> tuple[bool, NodeType]:
    if "RMSNorm" in cls_name or "RMSNorm".lower() in cls_name.lower():
        return True, "rms_norm"
    if "LayerNorm" in cls_name:
        return True, "layer_norm"
    return False, "custom"


def _is_activation(cls_name: str) -> bool:
    lname = cls_name.lower()
    return any(k in lname for k in ("silu", "gelu", "relu", "swish", "mish", "activation"))


def _cfg_get(cfg: object, name: str, default: object) -> object:
    """getattr(cfg, name, default), but also treats a config attribute that
    *raises* instead of returning a value as "unavailable", the same as a
    plain missing one. Newer transformers configs (Gemma4) make some fields
    (head_dim, num_key_value_heads, ...) genuinely vary per layer, and raise
    AmbiguousGlobalPerLayerAttributeError — a RuntimeError, not an
    AttributeError — when asked for one global value; plain getattr(...,
    default) doesn't catch that. Every value read this way is best-effort
    single-number metadata for generic display/reproduction (see
    build()'s head_dim comment for how a wrong guess here gets caught
    downstream), never worth failing the whole graph build over.
    """
    try:
        return getattr(cfg, name, default)
    except Exception:  # noqa: BLE001
        return default


_Q_NAMES = {"q_proj", "wq", "query"}
_K_NAMES = {"k_proj", "wk", "key"}
_V_NAMES = {"v_proj", "wv", "value"}
_O_NAMES = {"o_proj", "out_proj", "wo", "dense"}
_QKV_NAMES = {"qkv_proj", "wqkv", "c_attn"}
_GATE_NAMES = {"gate_proj", "w1", "wi_0"}
_UP_NAMES = {"up_proj", "w3", "wi_1"}
_DOWN_NAMES = {"down_proj", "w2", "wo"}
_GATE_UP_NAMES = {"gate_up_proj"}


class GraphBuilder:
    """One instance per loaded model. Not reused across models."""

    def __init__(self, model: nn.Module, model_id: str, display_name: str):
        self.model = model
        self.model_id = model_id
        self.display_name = display_name
        self.nodes: dict[str, ModelNode] = {}
        self.edges: list[ModelEdge] = []
        self.hook_modules: dict[str, nn.Module] = {}
        self.adds: list[tuple[str, str, str]] = []
        self.muls: list[tuple[str, str, str]] = []
        self.ropes: list[RopeDerivation] = []
        self.attention_heads: dict[str, AttentionHeadInfo] = {}
        self.moe_routers: dict[str, nn.Module] = {}
        self.moe_combines: dict[str, str] = {}
        # Detected from the actual loaded weights (see _linear_params),
        # not threaded in as a separate "what did the caller ask for"
        # flag — this way it's always honest about what's really on the
        # GPU, the same way graph_builder derives everything else.
        self.quantization_kind: str | None = None
        # Set inside _build_moe the first (and every) time it runs — lets
        # build() add MoE-specific ModelConfig.extra fields only for
        # models that actually have an MoE layer, without threading a
        # separate "is this model MoE" flag in from the caller.
        self.is_moe = False
        self.shared_expert_seen = False

    # -- node/edge helpers, mirroring apps/web's graph.ts node()/edge() -----

    def _node(self, node_id: str, node_type: NodeType, name: str, parent_id: str | None, **kw) -> ModelNode:
        n = ModelNode(id=node_id, type=node_type, name=name, parent_id=parent_id, **kw)
        self.nodes[node_id] = n
        if parent_id is not None:
            self.nodes[parent_id].children.append(node_id)
        return n

    def _edge(self, source: str, target: str, label: str | None = None) -> None:
        self.edges.append(ModelEdge(id=f"{source}->{target}", source=source, target=target, label=label))

    def _hook(self, node_id: str, module: nn.Module) -> None:
        """Marks node_id's activation as capturable directly from `module`'s real forward pass."""
        self.hook_modules[node_id] = module

    def _param_ref(self, path: str, attr_name: str, tensor: torch.Tensor, logical_shape: list[int] | None = None) -> ParameterRef:
        # A quantized weight's *storage* shape can be meaningless for
        # display (a packed 4-bit [256,256] layer's real tensor.shape is
        # [32768,1] — see quantization.py) — `logical_shape`, when given,
        # overrides shape/logicalShape/numElements for that case, while
        # `.bytes` still reflects the real (smaller) packed storage, an
        # honest memory-footprint number. No `slice` involved here (that's
        # a different mechanism, for GPT-2-style fused weights) — a
        # quantized weight is never sliced, so shape and logicalShape are
        # simply equal in this case, both already the logical shape.
        storage_shape = list(tensor.shape)
        storage_n = 1
        for d in storage_shape:
            storage_n *= d
        shape = logical_shape if logical_shape is not None else storage_shape
        n = 1
        for d in shape:
            n *= d
        return ParameterRef(
            name=f"{path}.{attr_name}",
            shape=shape,
            dtype=dtype_to_str(tensor.dtype),
            num_elements=n,
            bytes=storage_n * tensor.element_size(),
            provider_id=self.model_id,
            logical_shape=shape,
        )

    def _expert_param_ref(self, experts_path: str, attr_name: str, stacked: torch.Tensor, expert_idx: int) -> ParameterRef:
        """A modern (fused) MoE checkpoint stores every expert's weight as
        one slab — `stacked` is `[numExperts, ...]`, real per-expert storage
        with no per-expert nn.Module to point a ParameterRef at directly.
        Reuses the same `slice` mechanism the frontend already has for
        GPT-2-style fused QKV weights (see `_param_ref`'s doc comment):
        `name` still points at the one real fused parameter, `slice.ranges`
        narrows dim 0 to this expert's single index, and `shape`/`bytes`
        report just that expert's real slice — not the whole slab's,
        which would make each expert look `numExperts`× too large."""
        per_expert_shape = list(stacked.shape[1:])
        n = 1
        for d in per_expert_shape:
            n *= d
        ranges = [{"start": expert_idx, "end": expert_idx + 1}] + [{"start": 0, "end": d} for d in per_expert_shape]
        return ParameterRef(
            name=f"{experts_path}.{attr_name}",
            shape=per_expert_shape,
            dtype=dtype_to_str(stacked.dtype),
            num_elements=n,
            bytes=n * stacked.element_size(),
            provider_id=self.model_id,
            logical_shape=per_expert_shape,
            slice=TensorSlice(ranges=ranges),
        )

    def _linear_params(self, path: str, module: nn.Linear) -> list[ParameterRef]:
        kind = quantized_kind(module.weight)
        if kind is not None:
            self.quantization_kind = kind
        params = [self._param_ref(path, "weight", module.weight, logical_shape=logical_shape_of(module))]
        if module.bias is not None:
            params.append(self._param_ref(path, "bias", module.bias))
        return params

    def _custom_node(self, node_id: str, path: str, module: nn.Module, parent_id: str) -> ModelNode:
        """Fallback for any module the classifier doesn't recognize — real
        shape/params still show up in the tree/graph, just untyped."""
        params = [self._param_ref(path, name, p) for name, p in module.named_parameters(recurse=False)]
        return self._node(
            node_id,
            "custom",
            type(module).__name__,
            parent_id,
            parameters=params,
            metadata={"note": f"Unrecognized module type ({type(module).__name__}) — shown as-is.", "modulePath": path},
        )

    # -- attention / ffn internals ------------------------------------------

    def _classify_linears(self, container: nn.Module) -> dict[str, tuple[str, nn.Linear]]:
        out: dict[str, tuple[str, nn.Linear]] = {}
        for name, mod in container.named_children():
            if not isinstance(mod, nn.Linear):
                continue
            lname = name.lower()
            role = next(
                (
                    role
                    for role, names in (
                        ("q", _Q_NAMES),
                        ("k", _K_NAMES),
                        ("v", _V_NAMES),
                        ("o", _O_NAMES),
                        ("qkv", _QKV_NAMES),
                        ("gate", _GATE_NAMES),
                        ("up", _UP_NAMES),
                        ("down", _DOWN_NAMES),
                        ("gate_up", _GATE_UP_NAMES),
                    )
                    if lname in names
                ),
                f"other:{name}",
            )
            out[role] = (name, mod)
        return out

    def _find_activation(self, container: nn.Module) -> tuple[str, nn.Module] | None:
        for name, mod in container.named_children():
            if _is_activation(type(mod).__name__):
                return name, mod
        return None

    def _build_attention(
        self, node_id: str, path: str, module: nn.Module, parent_id: str, seq_h: list[int | str], num_heads: int, head_dim: int, rope_theta: float
    ) -> str:
        """Returns the id of the node that feeds the residual add."""
        linears = self._classify_linears(module)
        self._node(node_id, "attention", "Attention", parent_id, inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)])
        self._hook(node_id, module)

        q_id = k_id = None
        if "qkv" in linears:
            name, mod = linears["qkv"]
            qkv_id = f"{node_id}.qkv"
            self._node(
                qkv_id,
                "qkv_projection",
                "QKV Projection",
                node_id,
                inputs=[TensorSpec(dims=seq_h)],
                parameters=self._linear_params(f"{path}.{name}", mod),
                metadata={"note": "Fused Q/K/V weight — not yet split into individual Q/K/V nodes (see PLAN.md §10)."},
            )
            self._hook(qkv_id, mod)
            q_id = k_id = qkv_id
            q_out_features = None
        else:
            q_out_features = None
            for role, ntype, label in (("q", "q_projection", "Q Projection"), ("k", "k_projection", "K Projection"), ("v", "v_projection", "V Projection")):
                if role not in linears:
                    continue
                name, mod = linears[role]
                nid = f"{node_id}.{role}"
                self._node(
                    nid, ntype, label, node_id,
                    inputs=[TensorSpec(dims=seq_h)],
                    outputs=[TensorSpec(dims=["sequence_length", mod.out_features])],
                    parameters=self._linear_params(f"{path}.{name}", mod),
                )
                self._hook(nid, mod)
                if role == "q":
                    q_id = nid
                    q_out_features = mod.out_features
                if role == "k":
                    k_id = nid

        rotary_present = any("RotaryEmbedding" in type(m).__name__ for m in self.model.modules())
        into_o = q_id
        if rotary_present and q_id and k_id:
            rope_id = f"{node_id}.rope"
            self._node(
                rope_id, "rope", "RoPE", node_id,
                metadata={"description": "Rotates Q and K by an angle proportional to sequence position — no learned parameters."},
            )
            self._edge(q_id, rope_id)
            if k_id != q_id:
                self._edge(k_id, rope_id)
            # Not a hookable module — HF's rotary application is inline
            # tensor arithmetic inside Attention.forward(), not a distinct
            # submodule call. Reproduced from the real (pre-rope) Q this
            # node already points at, using the standard "rotate_half"
            # convention (the same formula nn-ops' ropeCosSin/applyRopeToHead
            # implements client-side) and this model's own rope_theta/head_dim
            # — generic across every architecture using that convention, not
            # a per-family special case. Skipped for a fused qkv_proj
            # (q_id == k_id): that node's captured tensor is the whole
            # packed [Q|K|V] width, not Q alone, so the reproduction's
            # "view as numHeads*headDim" assumption doesn't hold — no
            # derivation is safer than a silently wrong one (see PLAN.md §10).
            #
            # Also skipped whenever the q projection's real out_features
            # isn't exactly numHeads*headDim — not just fused QKV can widen
            # it. Qwen3.5's gated attention (attn_output_gate: true) packs a
            # same-width sigmoid gate into q_proj too (per head: [q_head_dim
            # | gate_head_dim], doubling out_features), split back apart by
            # torch.chunk *after* q_proj but *before* rope is ever applied —
            # so the captured q_proj activation genuinely isn't "pre-rope Q"
            # at all, it's "pre-rope Q, interleaved with an unrelated gate".
            # Reshaping that as [numHeads, headDim] doesn't just crash on
            # size (it did, until this check — RuntimeError: shape [...] is
            # invalid for input of size ...) — it would silently produce
            # wrong numbers even if the width happened to still divide
            # evenly, so this is caught the same conservative way as the
            # fused-QKV case, not patched to specifically special-case one
            # more architecture.
            q_width_is_pure = q_out_features is None or q_out_features == num_heads * head_dim
            if q_id != k_id and q_width_is_pure:
                self.ropes.append(RopeDerivation(node_id=rope_id, q_node_id=q_id, num_heads=num_heads, head_dim=head_dim, rope_theta=rope_theta))
            into_o = rope_id

        if "o" in linears:
            name, mod = linears["o"]
            o_id = f"{node_id}.out"
            self._node(
                o_id, "output_projection", "Output Projection", node_id,
                inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)],
                parameters=self._linear_params(f"{path}.{name}", mod),
            )
            self._hook(o_id, mod)
            # o_proj's *input* is always laid out as concatenated
            # [numHeads*headDim] regardless of whether Q/K/V were fused —
            # GQA's repeat_kv and any per-head norm/rope already happened
            # upstream — so "zero_head" is well-defined here independent of
            # the fused-QKV rope-reproduction gap above.
            self.attention_heads[node_id] = AttentionHeadInfo(o_proj=mod, head_dim=head_dim)
            if into_o:
                self._edge(into_o, o_id)
            return o_id

        return into_o or node_id

    def _build_ffn(
        self, node_id: str, path: str, module: nn.Module, parent_id: str, seq_h: list[int | str], intermediate_size: int, entry_feed: str | None = None
    ) -> str:
        linears = self._classify_linears(module)
        self._node(node_id, "ffn", "Feed Forward", parent_id, inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)])
        self._hook(node_id, module)
        inter = ["sequence_length", intermediate_size]

        if "gate" in linears and "up" in linears and "down" in linears:
            gname, gmod = linears["gate"]
            uname, umod = linears["up"]
            dname, dmod = linears["down"]
            gate_id, up_id, down_id = f"{node_id}.gate", f"{node_id}.up", f"{node_id}.down"
            self._node(gate_id, "linear", "Gate Projection", node_id, inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=inter)], parameters=self._linear_params(f"{path}.{gname}", gmod))
            self._node(up_id, "linear", "Up Projection", node_id, inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=inter)], parameters=self._linear_params(f"{path}.{uname}", umod))
            self._node(down_id, "linear", "Down Projection", node_id, inputs=[TensorSpec(dims=inter)], outputs=[TensorSpec(dims=seq_h)], parameters=self._linear_params(f"{path}.{dname}", dmod))
            self._hook(gate_id, gmod)
            self._hook(up_id, umod)
            self._hook(down_id, dmod)
            # gate/up are this section's real entry points — a leaf-only view
            # (the block-detail diagram) can only resolve an edge whose
            # endpoints are both leaves, so wiring `entry_feed` straight to
            # them (like _build_attention already does for q/k/v) instead of
            # only to this container's own (non-leaf) `node_id` is what
            # makes the FFN internals actually render connected there.
            if entry_feed is not None:
                self._edge(entry_feed, gate_id)
                self._edge(entry_feed, up_id)

            act = self._find_activation(module)
            gate_out = gate_id
            if act is not None:
                aname, amod = act
                act_id = f"{node_id}.gate_act"
                self._node(act_id, "activation", type(amod).__name__, node_id, inputs=[TensorSpec(dims=inter)], outputs=[TensorSpec(dims=inter)])
                self._edge(gate_id, act_id)
                self._hook(act_id, amod)
                gate_out = act_id

            mul_id = f"{node_id}.mul"
            self._node(mul_id, "elementwise_mul", "Gate × Up", node_id, inputs=[TensorSpec(dims=inter), TensorSpec(dims=inter)], outputs=[TensorSpec(dims=inter)])
            self._edge(gate_out, mul_id)
            self._edge(up_id, mul_id)
            self._edge(mul_id, down_id)
            # Not a hookable module — the gate/up multiply is inline
            # arithmetic inside MLP.forward(), not a distinct submodule
            # call. Both operands are already real hooked values, so this
            # is an exact (not approximated) elementwise product, computed
            # once activations are in hand — see resolve_derivations.
            self.muls.append((mul_id, gate_out, up_id))
            return down_id

        # Fallback: whatever Linear/activation children exist, wired in
        # declaration order — not the rich gated-FFN shape above, but still
        # real and never a crash. Starting `prev` at `entry_feed` (a real
        # leaf) rather than `node_id` (this container, not a leaf) means the
        # very first child's incoming edge resolves in the leaf-only
        # block-detail view too, same reasoning as the gate/up case above.
        prev = entry_feed if entry_feed is not None else node_id
        for name, mod in module.named_children():
            child_id = f"{node_id}.{name}"
            if isinstance(mod, nn.Linear):
                self._node(child_id, "linear", name, node_id, parameters=self._linear_params(f"{path}.{name}", mod))
                self._hook(child_id, mod)
                self._edge(prev, child_id)
                prev = child_id
            elif _is_activation(type(mod).__name__):
                self._node(child_id, "activation", type(mod).__name__, node_id)
                self._hook(child_id, mod)
                self._edge(prev, child_id)
                prev = child_id
        return prev

    def _build_moe(
        self,
        node_id: str,
        path: str,
        module: nn.Module,
        parent_id: str,
        seq_h: list[int | str],
        num_experts: int,
        top_k: int,
        expert_intermediate_size: int,
        entry_feed: str,
    ) -> str:
        """A Mixture-of-Experts block: a `router` (picks `top_k` experts per
        token) plus `numExperts` `expert` FFNs, of which only the routed-to
        ones actually run for any given token — see PLAN.md §4.2/§8.8.

        Current transformers versions (confirmed empirically against
        Qwen2Moe/Mixtral — see graph_builder's module docstring for the
        general "walk real modules" approach this mirrors) implement every
        MoE block the same generic way regardless of architecture: a
        `*Router`-named submodule (`.forward()` returns a real
        `(logits, topKWeights, topKIndices)` 3-tuple, no batch dim — needs
        its own hook, see `moe_routers`/inference.py, not the generic
        edit-capable one) and a `*Experts`-named submodule holding every
        expert's weight fused into one `[numExperts, ...]` slab (a
        performance/kernel-fusion choice — there's no single per-expert
        nn.Module call to hook for activation capture, which is why expert
        nodes below carry weights only, no hook; see PLAN.md §4.3's note on
        never eagerly running undispatched experts anyway). An optional
        `shared_expert` (structurally a plain dense FFN, always-on
        alongside the routed ones — DeepSeek/Qwen-MoE-style architectures)
        is reused via `_build_ffn` as-is since it's a real hookable module
        with the same gate/up/down shape any dense FFN has.

        Falls back to an older per-expert-nn.Module ModuleList shape
        (pre-fusion architectures) where it exists, since that shape lets
        `_build_ffn` give per-expert nodes real hooks/activation capture
        the fused shape can't — same "richer real shape when it exists,
        graceful generic fallback otherwise" pattern as the rest of this
        file, not a fused-only implementation.

        Router and every expert (plus the shared expert, if present) all
        consume the same `entry_feed` directly — wired here as real
        leaf-to-leaf edges, like `_build_attention`'s q/k/v, so the
        block-detail diagram's leaf-only view can actually resolve them
        (an edge into this container's own `node_id` alone can't, since
        it's not a leaf). Returns a synthetic `combine` leaf instead of
        `node_id` for the same reason on the way out: nothing real
        computes "the routed experts' combined output" as a distinct,
        independently-hookable step (it's inline arithmetic inside the
        fused Experts kernel, same category as gate×up's elementwise
        multiply elsewhere in this file) — but unlike that multiply, its
        value can't be *derived* from already-captured operands either
        (individual expert outputs are never separately captured — see
        above), so inference.py instead aliases its activation straight
        from `node_id`'s own real hook output, which already *is* exactly
        this quantity.

        Resolves router/experts/shared-expert from `module`'s own children —
        the shape every MoE architecture seen so far groups them under one
        wrapper submodule for. See _build_moe_from_siblings for the one
        that doesn't (Gemma4: router/experts are direct decoder-layer
        children, not nested in a wrapper at all).
        """
        router_name = router_mod = None
        experts_name = experts_mod = None
        shared_name = shared_mod = None
        for name, mod in module.named_children():
            cname = type(mod).__name__
            if router_mod is None and "Router" in cname:
                router_name, router_mod = name, mod
            elif experts_mod is None and "Experts" in cname:
                experts_name, experts_mod = name, mod
            elif shared_mod is None and "shared_expert" in name.lower() and not isinstance(mod, nn.Linear):
                shared_name, shared_mod = name, mod

        # Fallback for an architecture whose router is still a bare
        # nn.Linear (hidden -> numExperts) rather than a dedicated *Router
        # class — same shape, just not yet refactored to the newer pattern.
        if router_mod is None:
            for name, mod in module.named_children():
                if isinstance(mod, nn.Linear) and mod.out_features == num_experts:
                    router_name, router_mod = name, mod
                    break

        return self._build_moe_nodes(
            node_id, path, module, parent_id, seq_h, num_experts, top_k, expert_intermediate_size, entry_feed,
            router_name, router_mod, experts_name, experts_mod, shared_name, shared_mod,
        )

    def _build_moe_from_siblings(
        self,
        node_id: str,
        path: str,
        parent_id: str,
        seq_h: list[int | str],
        num_experts: int,
        top_k: int,
        expert_intermediate_size: int,
        entry_feed: str,
        router_name: str,
        router_mod: nn.Module,
        experts_name: str,
        experts_mod: nn.Module,
        dense_name: str | None,
        dense_mod: nn.Module | None,
    ) -> str:
        """Gemma4-style MoE: `router`/`experts` are direct decoder-layer
        children (siblings of `self_attn`/`mlp`), not grouped under one MoE
        wrapper module the way every other architecture _build_moe has seen
        is — so there's no single container to resolve them from, or to
        hook for this node's own activation (Gemma4TextExperts.forward()
        fills that second role instead: it's the real module whose output
        *is* "the routed experts' weighted sum", exactly what `node_id`
        represents here).

        Gemma4's dense `mlp` branch also isn't a fallback for an unrouted
        token the way a `shared_expert` is elsewhere — it runs
        unconditionally for every token, in parallel with the routed
        experts, and the two (independently normed) outputs are summed.
        Structurally that's exactly _build_moe's existing shared_expert
        case, so `dense_mod` is threaded into that same slot rather than
        given its own bespoke handling.
        """
        return self._build_moe_nodes(
            node_id, path, experts_mod, parent_id, seq_h, num_experts, top_k, expert_intermediate_size, entry_feed,
            router_name, router_mod, experts_name, experts_mod, dense_name, dense_mod,
        )

    def _build_moe_nodes(
        self,
        node_id: str,
        path: str,
        hook_target: nn.Module,
        parent_id: str,
        seq_h: list[int | str],
        num_experts: int,
        top_k: int,
        expert_intermediate_size: int,
        entry_feed: str,
        router_name: str | None,
        router_mod: nn.Module | None,
        experts_name: str | None,
        experts_mod: nn.Module | None,
        shared_name: str | None,
        shared_mod: nn.Module | None,
    ) -> str:
        """The actual node/edge building shared by _build_moe (wrapper-
        module architectures) and _build_moe_from_siblings (Gemma4) once
        each has resolved router_mod/experts_mod/shared_mod its own way —
        see both callers' doc comments for what differs and why."""
        self.is_moe = True
        self._node(
            node_id, "moe_layer", "Mixture of Experts", parent_id,
            inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)],
            metadata={"numExperts": num_experts, "numExpertsPerTok": top_k},
        )
        self._hook(node_id, hook_target)

        if router_mod is not None:
            router_id = f"{node_id}.router"
            router_path = f"{path}.{router_name}"
            router_params = (
                [self._param_ref(router_path, "weight", router_mod.weight)] if getattr(router_mod, "weight", None) is not None else []
            )
            self._node(
                router_id, "router", "Router", node_id,
                inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=["sequence_length", num_experts])],
                parameters=router_params,
            )
            self.moe_routers[node_id] = router_mod
            self._edge(entry_feed, router_id)

        expert_ids: list[str] = []
        if experts_mod is not None and hasattr(experts_mod, "gate_up_proj") and hasattr(experts_mod, "down_proj"):
            experts_path = f"{path}.{experts_name}"
            gate_up, down = experts_mod.gate_up_proj, experts_mod.down_proj
            for i in range(num_experts):
                expert_id = f"{node_id}.expert.{i}"
                params = [
                    self._expert_param_ref(experts_path, "gate_up_proj", gate_up, i),
                    self._expert_param_ref(experts_path, "down_proj", down, i),
                ]
                self._node(expert_id, "expert", f"Expert {i}", node_id, parameters=params, metadata={"expertIndex": i})
                self._edge(entry_feed, expert_id)
                expert_ids.append(expert_id)
        elif isinstance(experts_mod, nn.ModuleList):
            for i, expert_mod in enumerate(experts_mod):
                expert_id = f"{node_id}.expert.{i}"
                # A real per-expert nn.Module here means _build_ffn's own
                # `down_id` return is the expert's real, independently
                # hookable output leaf — used below (not `expert_id`
                # itself) as what actually feeds the combine step.
                expert_ids.append(self._build_ffn(expert_id, f"{path}.{experts_name}.{i}", expert_mod, node_id, seq_h, expert_intermediate_size, entry_feed))

        shared_out_id = None
        if shared_mod is not None:
            self.shared_expert_seen = True
            shared_id = f"{node_id}.shared_expert"
            shared_linears = self._classify_linears(shared_mod)
            shared_inter = shared_linears["up"][1].out_features if "up" in shared_linears else expert_intermediate_size
            shared_out_id = self._build_ffn(shared_id, f"{path}.{shared_name}", shared_mod, node_id, seq_h, shared_inter, entry_feed)
            self.nodes[shared_id].name = "Shared Expert"

        combine_inputs = list(expert_ids) + ([shared_out_id] if shared_out_id is not None else [])
        if not combine_inputs:
            return node_id
        combine_id = f"{node_id}.combine"
        self._node(
            combine_id, "residual", "Combine Experts", node_id,
            inputs=[TensorSpec(dims=seq_h)] * len(combine_inputs), outputs=[TensorSpec(dims=seq_h)],
            metadata={
                "note": "The weighted sum of the routed experts' outputs"
                + (", plus the always-on shared/dense branch's contribution." if shared_out_id is not None else ".")
            },
        )
        for src in combine_inputs:
            self._edge(src, combine_id)
        self.moe_combines[node_id] = combine_id
        return combine_id

    # -- top-level build ------------------------------------------------------

    def build(self) -> BuiltGraph:
        # A multimodal checkpoint's top-level config (e.g. Gemma4Config,
        # Qwen3_5Config) is a composite wrapper with no hidden_size/
        # num_attention_heads/etc. of its own — those live on its nested
        # text_config instead. AutoModelForCausalLM resolves some such
        # families (Qwen3.5) straight to a dedicated text-only ...ForCausalLM
        # class whose .config is already the flattened text_config, but
        # others (Gemma4 — no such class exists) load as the full
        # conditional-generation model, .config and all. Falling back to
        # .text_config whenever present covers both without needing to know
        # which case a given family is.
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        H = int(cfg.hidden_size)
        num_layers = int(cfg.num_hidden_layers)
        num_heads = int(cfg.num_attention_heads)
        # Some newer configs (Gemma4) make head_dim a genuinely per-layer-
        # varying value (e.g. wider heads on its sparser "global" attention
        # layers than on its sliding-window ones) — _cfg_get's fallback here
        # is then just a best-effort single number passed into every
        # layer's _build_attention call regardless of its real per-layer
        # head_dim; wherever it's actually wrong for a given layer, that
        # layer's real q-projection width won't match num_heads * head_dim,
        # and the same guard that already protects fused/gated Q
        # projections (see _build_attention's q_width_is_pure) skips
        # reproducing RoPE for it rather than displaying wrong numbers.
        head_dim = int(_cfg_get(cfg, "head_dim", H // num_heads))
        seq_h: list[int | str] = ["sequence_length", H]

        # transformers >=4.54 moved rope_theta off the config's top level and
        # into a `rope_parameters` dict (older checkpoints/versions still
        # expose it as a flat `rope_theta` attribute) — check both. Needed
        # per-layer below (for each attention's rope derivation), so this
        # has to happen before the layer loop rather than after it.
        rope_params = _cfg_get(cfg, "rope_parameters", None) or {}
        rope_theta = rope_params.get("rope_theta") if isinstance(rope_params, dict) else None
        if rope_theta is None:
            rope_theta = float(_cfg_get(cfg, "rope_theta", 10000.0))

        # MoE config, read once regardless of whether this model actually
        # has MoE layers — harmless on a dense model since _build_moe (the
        # only place these are used) then simply never runs. Field names
        # vary slightly across architectures (num_local_experts predates
        # num_experts on some configs), hence the fallbacks.
        num_experts = int(_cfg_get(cfg, "num_experts", _cfg_get(cfg, "num_local_experts", 0)) or 0)
        experts_per_tok = int(_cfg_get(cfg, "num_experts_per_tok", 1) or 1)
        expert_intermediate_size = int(_cfg_get(cfg, "moe_intermediate_size", _cfg_get(cfg, "intermediate_size", 4 * H)) or (4 * H))

        self._node("model", "model", self.display_name, None)
        self._node("input", "input", "Input tokens", "model", outputs=[TensorSpec(dims=["sequence_length"])])

        embed_entry = next(((name, mod) for name, mod in self.model.named_modules() if isinstance(mod, nn.Embedding)), None)
        if embed_entry is None:
            raise ValueError("No nn.Embedding module found — can't locate the token embedding table.")
        embed_path, embed_mod = embed_entry
        self._node(
            "embed", "embedding", "Token Embedding", "model",
            inputs=[TensorSpec(dims=["sequence_length"])], outputs=[TensorSpec(dims=seq_h)],
            parameters=[self._param_ref(embed_path, "weight", embed_mod.weight)],
        )
        self._hook("embed", embed_mod)
        self._edge("input", "embed")

        layer_entry = next(((name, mod) for name, mod in self.model.named_modules() if isinstance(mod, nn.ModuleList) and len(mod) == num_layers), None)
        if layer_entry is None:
            raise ValueError(f"No ModuleList of length {num_layers} (config.num_hidden_layers) found.")
        layer_list_path, layer_list = layer_entry

        self._node("blocks", "block_group", f"Transformer Blocks × {num_layers}", "model", metadata={"count": num_layers})

        prev_out = "embed"
        for i, layer in enumerate(layer_list):
            b = f"block.{i}"
            layer_path = f"{layer_list_path}.{i}"
            self._node(b, "transformer_block", f"Transformer Block {i}", "blocks", inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)])
            # The whole DecoderLayer-equivalent module has its own real
            # forward pass, so this node's activation is a genuine hook —
            # not derived — even though its *children* below include some
            # that aren't (residual adds).
            self._hook(b, layer)
            self._edge(prev_out, b)
            # The real value flowing into this block's residual adds is
            # whatever the previous iteration produced ("embed" for i==0) —
            # captured now, before `prev_out` is reassigned at the bottom of
            # this loop. Deliberately NOT `b`: this block's own node id is
            # hooked to its *output* above, and reusing it here for the
            # skip-connection input would both be wrong data and a cycle.
            block_real_input = prev_out

            attn_entry = None
            ffn_entry = None
            ffn_is_moe = False
            # Gemma4-style: `router`/`experts` sit as direct decoder-layer
            # children alongside (not nested inside, and not instead of) a
            # dense `mlp` — see _build_moe_from_siblings' doc comment.
            # Classified separately from ffn_entry's "MLP or MoE wrapper"
            # check above so a real dense `mlp` on the same layer doesn't
            # get misread as this architecture's whole FFN.
            moe_router_entry = None
            moe_experts_entry = None
            norms: list[tuple[str, nn.Module]] = []
            other_children: list[tuple[str, nn.Module]] = []
            for name, mod in layer.named_children():
                cname = type(mod).__name__
                is_norm, _ = _is_norm(cname)
                if "Attention" in cname:
                    attn_entry = (name, mod)
                elif is_norm:
                    norms.append((name, mod))
                elif "Router" in cname:
                    moe_router_entry = (name, mod)
                elif "Experts" in cname:
                    moe_experts_entry = (name, mod)
                elif "MLP" in cname or "FeedForward" in cname or "MoE" in cname or "Moe" in cname:
                    ffn_entry = (name, mod)
                    ffn_is_moe = "MoE" in cname or "Moe" in cname
                else:
                    other_children.append((name, mod))

            attn_feed = b
            if norms:
                nname, nmod = norms[0]
                _, ntype = _is_norm(type(nmod).__name__)
                norm_id = f"{b}.norm_pre_attn"
                self._node(norm_id, ntype, "Pre-attention Norm", b, inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)], parameters=[self._param_ref(f"{layer_path}.{nname}", "weight", nmod.weight)] if hasattr(nmod, "weight") and nmod.weight is not None else [])
                self._hook(norm_id, nmod)
                self._edge(b, norm_id)
                attn_feed = norm_id

            res1_out = b
            if attn_entry is not None:
                aname, amod = attn_entry
                attn_id = f"{b}.attn"
                out_id = self._build_attention(attn_id, f"{layer_path}.{aname}", amod, b, seq_h, num_heads, head_dim, float(rope_theta))
                for entry_id in (f"{attn_id}.q", f"{attn_id}.k", f"{attn_id}.v", f"{attn_id}.qkv"):
                    if entry_id in self.nodes:
                        self._edge(attn_feed, entry_id)

                res1_id = f"{b}.res1"
                self._node(res1_id, "residual", "Residual Add", b, inputs=[TensorSpec(dims=seq_h), TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)])
                self._edge(out_id, res1_id)
                self._edge(b, res1_id, "skip")
                self.adds.append((res1_id, out_id, block_real_input))
                res1_out = res1_id

            ffn_feed = res1_out
            if len(norms) > 1:
                nname, nmod = norms[1]
                _, ntype = _is_norm(type(nmod).__name__)
                norm_id = f"{b}.norm_pre_ffn"
                self._node(norm_id, ntype, "Pre-FFN Norm", b, inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)], parameters=[self._param_ref(f"{layer_path}.{nname}", "weight", nmod.weight)] if hasattr(nmod, "weight") and nmod.weight is not None else [])
                self._hook(norm_id, nmod)
                self._edge(res1_out, norm_id)
                ffn_feed = norm_id

            block_out = res1_out
            if moe_experts_entry is not None and num_experts > 0:
                ename, emod = moe_experts_entry
                rname, rmod = moe_router_entry if moe_router_entry is not None else (None, None)
                dense_name, dense_mod = ffn_entry if ffn_entry is not None else (None, None)
                ffn_id = f"{b}.ffn"
                ffn_out_id = self._build_moe_from_siblings(
                    ffn_id, layer_path, b, seq_h, num_experts, experts_per_tok, expert_intermediate_size, ffn_feed,
                    rname, rmod, ename, emod, dense_name, dense_mod,
                )
                res2_id = f"{b}.res2"
                self._node(res2_id, "residual", "Residual Add", b, inputs=[TensorSpec(dims=seq_h), TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)])
                self._edge(ffn_out_id, res2_id)
                self._edge(res1_out, res2_id, "skip")
                self.adds.append((res2_id, ffn_out_id, res1_out))
                block_out = res2_id
            elif ffn_entry is not None:
                fname, fmod = ffn_entry
                ffn_id = f"{b}.ffn"
                if ffn_is_moe and num_experts > 0:
                    ffn_out_id = self._build_moe(ffn_id, f"{layer_path}.{fname}", fmod, b, seq_h, num_experts, experts_per_tok, expert_intermediate_size, ffn_feed)
                else:
                    ffn_out_id = self._build_ffn(ffn_id, f"{layer_path}.{fname}", fmod, b, seq_h, int(_cfg_get(cfg, "intermediate_size", 4 * H)), ffn_feed)
                res2_id = f"{b}.res2"
                self._node(res2_id, "residual", "Residual Add", b, inputs=[TensorSpec(dims=seq_h), TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)])
                self._edge(ffn_out_id, res2_id)
                self._edge(res1_out, res2_id, "skip")
                self.adds.append((res2_id, ffn_out_id, res1_out))
                block_out = res2_id

            for name, mod in other_children:
                cid = f"{b}.{name}"
                self._custom_node(cid, f"{layer_path}.{name}", mod, b)
                self._hook(cid, mod)
                self._edge(b, cid)

            prev_out = block_out

        final_norm_entry = next(
            (
                (name, mod)
                for name, mod in self.model.named_modules()
                if _is_norm(type(mod).__name__)[0] and not name.startswith(layer_list_path) and name != ""
            ),
            None,
        )
        norm_out = prev_out
        if final_norm_entry is not None:
            nname, nmod = final_norm_entry
            _, ntype = _is_norm(type(nmod).__name__)
            self._node("norm", ntype, "Final Norm", "model", inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=seq_h)], parameters=[self._param_ref(nname, "weight", nmod.weight)] if hasattr(nmod, "weight") and nmod.weight is not None else [])
            self._hook("norm", nmod)
            self._edge(prev_out, "norm")
            norm_out = "norm"

        vocab_size = int(cfg.vocab_size)
        lm_head_entry = next(((name, mod) for name, mod in self.model.named_modules() if name.endswith("lm_head") and isinstance(mod, nn.Linear)), None)
        tied = bool(_cfg_get(cfg, "tie_word_embeddings", False))
        if lm_head_entry is not None:
            _, lm_head_mod = lm_head_entry
            param_path = embed_path if tied else lm_head_entry[0]
            weight = embed_mod.weight if tied else lm_head_mod.weight
            # Tied means this is really embed_tokens' weight (never
            # quantized — see quantization.py's doc comment), so
            # logical_shape_of only matters on the untied branch; it's a
            # no-op (returns None) on any non-quantized weight regardless.
            lm_head_logical_shape = None if tied else logical_shape_of(lm_head_mod)
            self._node(
                "lm_head", "lm_head", "LM Head", "model",
                inputs=[TensorSpec(dims=seq_h)], outputs=[TensorSpec(dims=["sequence_length", vocab_size])],
                parameters=[self._param_ref(param_path, "weight", weight, logical_shape=lm_head_logical_shape)],
                metadata={"tied": tied, **({"description": "Tied to the token embedding weight (transposed)."} if tied else {})},
            )
            self._hook("lm_head", lm_head_mod)
            self._edge(norm_out, "lm_head")
            output_source = "lm_head"
        else:
            output_source = norm_out

        self._node("output", "output", "Logits", "model", inputs=[TensorSpec(dims=["sequence_length", vocab_size])])
        self._edge(output_source, "output")

        model_config = ModelConfig(
            model_type=str(_cfg_get(cfg, "model_type", "unknown")),
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=H,
            intermediate_size=int(_cfg_get(cfg, "intermediate_size", 4 * H)),
            vocab_size=vocab_size,
            context_length=int(_cfg_get(cfg, "max_position_embeddings", 4096)),
            extra={
                "numKeyValueHeads": int(_cfg_get(cfg, "num_key_value_heads", num_heads)),
                "ropeTheta": float(rope_theta),
                "rmsNormEps": float(_cfg_get(cfg, "rms_norm_eps", _cfg_get(cfg, "layer_norm_eps", 1e-6))),
                "activationFunction": str(_cfg_get(cfg, "hidden_act", "silu")),
                "tiedEmbeddings": tied,
                "quantization": self.quantization_kind,
                **(
                    {
                        "numExperts": num_experts,
                        "numExpertsPerTok": experts_per_tok,
                        "expertIntermediateSize": expert_intermediate_size,
                        **({"sharedExpertCount": int(_cfg_get(cfg, "n_shared_experts", 1) or 1)} if self.shared_expert_seen else {}),
                    }
                    if self.is_moe
                    else {}
                ),
            },
        )

        model_ir = Model(
            id=self.model_id,
            name=self.display_name,
            architecture=type(self.model).__name__,
            config=model_config,
            inputs=self.nodes["input"].outputs,
            outputs=self.nodes["output"].inputs,
            nodes=self.nodes,
            edges=self.edges,
            root_id="model",
        )
        return BuiltGraph(
            model=model_ir,
            hook_modules=self.hook_modules,
            adds=self.adds,
            muls=self.muls,
            ropes=self.ropes,
            attention_heads=self.attention_heads,
            moe_routers=self.moe_routers,
            moe_combines=self.moe_combines,
        )


def build_generic_model_ir(model: nn.Module, model_id: str, display_name: str) -> BuiltGraph:
    return GraphBuilder(model, model_id, display_name).build()

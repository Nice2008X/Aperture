"""bitsandbytes quantization support (PLAN.md §8.7).

A quantized `nn.Linear`'s weight `Parameter` is *packed* storage, not a
plain float tensor — for 4-bit (`Params4bit`) its `.shape` doesn't even
match the logical [out_features, in_features] the layer computes with (a
256×256 layer's packed weight is `[32768, 1]`, not `[256, 256]`); for
8-bit (`Int8Params`) the shape happens to match but the raw int8 values
still need a per-channel scale applied to mean anything as floats. Two
places in this app need to see through that:

- graph_builder.py needs the *logical* shape/byte-count for its
  ParameterRef (module.in_features/out_features, not weight.shape) —
  see `logical_shape_of`.
- tensors.py's weight-fetch endpoint needs real float values, not packed
  codes, for TensorExplorer's heatmap/stats to mean anything — see
  `dequantize_weight`. Both bitsandbytes classes still subclass
  `nn.Linear` (confirmed empirically before relying on it here), so the
  existing `isinstance(mod, nn.Linear)` classification throughout
  graph_builder.py already finds these without any change there.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

QuantKind = Literal["4bit", "8bit"]


def quantized_kind(weight: torch.Tensor) -> QuantKind | None:
    cls = type(weight).__name__
    if cls == "Params4bit":
        return "4bit"
    if cls == "Int8Params":
        return "8bit"
    return None


def logical_shape_of(module: nn.Linear) -> list[int] | None:
    """Returns [out_features, in_features] if `module`'s weight is
    quantized (where `.weight.shape` would be wrong or misleading),
    else None (caller should fall back to `.weight.shape` as normal)."""
    if quantized_kind(module.weight) is None:
        return None
    return [module.out_features, module.in_features]


def dequantize_weight(model: nn.Module, param_name: str) -> torch.Tensor | None:
    """Returns a real [out_features, in_features] float tensor for a
    quantized Linear's weight, or None if `param_name` isn't one (caller
    should fall back to `model.get_parameter(param_name)` as normal).
    """
    param = model.get_parameter(param_name)
    kind = quantized_kind(param)
    if kind is None:
        return None

    if kind == "4bit":
        import bitsandbytes.functional as bnf

        return bnf.dequantize_4bit(param.data, param.quant_state)

    # 8-bit: the raw int8 codes need a per-output-channel scale (SCB) to
    # become real values. That scale is computed once, at load time, by
    # Int8Params._quantize and stored on the weight Parameter itself
    # (param.SCB) — NOT on module.state.SCB, which is a separate, forward
    # pass-only cache that stays None until the module has actually run at
    # least one forward pass (confirmed empirically: fetching a weight
    # before any inference hit `module.state.SCB is None`). Using
    # param.SCB means this works immediately after load, with no forward
    # pass required first.
    scb = param.SCB
    return param.data.float() * scb.view(-1, 1) / 127.0

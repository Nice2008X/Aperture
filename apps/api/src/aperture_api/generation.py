"""Real multi-token generation (PLAN.md §8.5) — the phase-1 through 4 work
covered *inspecting* a single forward pass; this is the other mode PLAN.md
§0 called for: an actual chat-style "generate N tokens" loop, streamed as
they're produced, using the model's real KV cache (not a fresh forward pass
per token — that would recompute the whole prefix every step).

No hooks are registered here (generation doesn't need per-node capture) —
but a generation step still runs the same shared modules run_forward/
run_attribution_sweep hook onto, so callers must still hold
LoadedModel.inference_lock for the whole stream, exactly like those two.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from .model_registry import LoadedModel


def _eos_token_ids(loaded: LoadedModel) -> set[int]:
    gen_cfg = getattr(loaded.model, "generation_config", None)
    eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if eos is None:
        eos = getattr(loaded.tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> int:
    """logits: [vocab], already the target position's raw logits."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled = logits / max(temperature, 1e-5)
    if top_k > 0:
        k = min(top_k, scaled.shape[-1])
        values, indices = torch.topk(scaled, k)
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(0, indices, values)
        scaled = filtered

    probs = torch.softmax(scaled, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest prefix whose cumulative probability already
        # reaches top_p — the token that *crosses* the threshold stays in
        # (excluding it could leave an empty distribution when one token
        # already dominates), everything after it is zeroed.
        drop = (cumulative - sorted_probs) > top_p
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)
        total = probs.sum()
        if total > 0:
            probs = probs / total

    return int(torch.multinomial(probs, 1).item())


def generate_tokens(
    loaded: LoadedModel,
    token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[tuple[int, str, bool]]:
    """Yields (tokenId, decodedText, isLast) one real generation step at a
    time, using the model's real KV cache — each step after the first only
    forwards the one new token, not the whole prefix again."""
    device = loaded.model.device
    input_ids = torch.tensor([token_ids], device=device)
    past_key_values = None
    eos_ids = _eos_token_ids(loaded)

    for step in range(max_new_tokens):
        with torch.no_grad():
            step_input = input_ids if past_key_values is None else input_ids[:, -1:]
            out = loaded.model(step_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values

        next_logits = out.logits[0, -1, :].float()
        next_id = _sample(next_logits, temperature, top_p, top_k)
        text = loaded.tokenizer.decode([next_id]) if loaded.tokenizer is not None else f"#{next_id}"
        is_eos = next_id in eos_ids
        is_last = is_eos or step == max_new_tokens - 1

        yield next_id, text, is_last

        if is_eos:
            return
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

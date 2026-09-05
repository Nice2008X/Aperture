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


DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_REPETITION_PENALTY = 1.1
# Almost never set by model authors (unlike the four above, which come
# straight from generation_config.json when present) — this is a pure
# app-level default, chosen because it's the missing piece that actually
# stops a long generation from looping (see generation_defaults' doc
# comment below for the full story).
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3


def generation_defaults(loaded: LoadedModel) -> dict:
    """Per-model sampling defaults for the frontend's Settings panel —
    sourced from this checkpoint's own generation_config.json (already
    parsed into loaded.model.generation_config at load time) wherever its
    authors actually specified one, falling back to this app's own
    defaults for whatever a checkpoint leaves unset.

    That fallback matters more than it looks: a generation_config.json
    auto-derived by transformers from a bare model config (flagged
    `_from_model_config: true`, e.g. this app's SmolLM2/granite test
    checkpoints) carries no temperature/top_p/top_k/repetition_penalty at
    all — every field reads back as None, not some inherited default. And
    repetition_penalty itself, even on a checkpoint that does specify one
    (Qwen ships 1.1), is exactly what fixes the "repeats the same
    sentence forever" failure real long generations hit with the plain
    temperature/top_p/top_k sampling this app had before — reproduced
    directly against this app's own /api/generate: unconstrained greedy
    decoding past a few hundred tokens degenerates into the exact same
    paragraph looping verbatim, and repetition_penalty plus
    no_repeat_ngram_size together (not either alone) is what eliminates it.
    """
    gen_cfg = getattr(loaded.model, "generation_config", None)

    def _get(name: str, default: float) -> float:
        value = getattr(gen_cfg, name, None) if gen_cfg is not None else None
        return value if value is not None else default

    return {
        "temperature": _get("temperature", DEFAULT_TEMPERATURE),
        "topP": _get("top_p", DEFAULT_TOP_P),
        "topK": _get("top_k", DEFAULT_TOP_K),
        "repetitionPenalty": _get("repetition_penalty", DEFAULT_REPETITION_PENALTY),
        "noRepeatNgramSize": _get("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE),
    }


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    """CTRL-style repetition penalty (Keskar et al., the same formula
    transformers' own RepetitionPenaltyLogitsProcessor uses): a positive
    logit for a token already generated is divided by `penalty` (pushing
    it down), a negative one is multiplied by it (pushing it down too,
    since multiplying a negative by >1 makes it more negative) — so
    penalty > 1.0 discourages repeats, 1.0 is a no-op. Scoped to just this
    step's newly-generated tokens (not the original prompt) — deliberately
    lets a "repeat this back to me" style prompt still work, at the cost
    of not suppressing verbatim prompt echoing the way transformers'
    default (which penalizes the full input_ids) would.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    ids = torch.tensor(sorted(set(generated_ids)), device=logits.device)
    scores = logits[ids]
    logits[ids] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def _ban_repeated_ngrams(logits: torch.Tensor, generated_ids: list[int], ngram_size: int) -> torch.Tensor:
    """Hard-bans whichever token(s) would recreate an n-gram already seen
    in this step's generated output — transformers' own
    NoRepeatNGramLogitsProcessor technique. Where repetition_penalty only
    discourages repeats, this makes an exact repeat of any run of
    `ngram_size` tokens impossible outright, which is what actually closes
    off the self-reinforcing loop repetition_penalty alone still lets a
    long enough greedy run fall back into (verified empirically — see
    generation_defaults' doc comment). O(n) per step, same as the
    decode-buffering redecode above; fine at the lengths this app
    generates, not optimized for very long runs.
    """
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return logits
    prefix = tuple(generated_ids[-(ngram_size - 1):]) if ngram_size > 1 else ()
    banned = {
        generated_ids[i + ngram_size - 1]
        for i in range(len(generated_ids) - ngram_size + 1)
        if tuple(generated_ids[i : i + ngram_size - 1]) == prefix
    }
    if banned:
        logits[torch.tensor(sorted(banned), device=logits.device)] = float("-inf")
    return logits


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
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> Iterator[tuple[int, str, bool]]:
    """Yields (tokenId, decodedText, isLast) one real generation step at a
    time, using the model's real KV cache — each step after the first only
    forwards the one new token, not the whole prefix again.

    decodedText is never decoded from a single token id in isolation. A
    byte-level BPE tokenizer routinely splits one Unicode character (most
    emoji, ZWJ sequences, and plenty of non-Latin script) across several
    tokens; decoding just-arrived token N alone hands the tokenizer an
    incomplete UTF-8 byte sequence, which comes back as U+FFFD ("unrecognized
    characters" in the UI). Instead this re-decodes the *whole* sequence
    generated so far every step and emits only the newly-completed suffix,
    withholding it whenever the redecoded text still ends in U+FFFD (i.e. the
    tail is still an unresolved partial character, pending more tokens) —
    the same technique transformers' own TextStreamer uses. skip_special_tokens
    also means a stray control token (whether the final EOS or one the model
    unexpectedly emits mid-stream) never surfaces as literal bracketed text
    either.
    """
    device = loaded.model.device
    input_ids = torch.tensor([token_ids], device=device)
    past_key_values = None
    eos_ids = _eos_token_ids(loaded)

    generated_ids: list[int] = []
    printed_len = 0

    for step in range(max_new_tokens):
        with torch.no_grad():
            step_input = input_ids if past_key_values is None else input_ids[:, -1:]
            out = loaded.model(step_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values

        next_logits = out.logits[0, -1, :].float()
        next_logits = _apply_repetition_penalty(next_logits, generated_ids, repetition_penalty)
        next_logits = _ban_repeated_ngrams(next_logits, generated_ids, no_repeat_ngram_size)
        next_id = _sample(next_logits, temperature, top_p, top_k)
        is_eos = next_id in eos_ids
        is_last = is_eos or step == max_new_tokens - 1

        generated_ids.append(next_id)
        if loaded.tokenizer is not None:
            full_text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if full_text.endswith("�") and not is_last:
                text = ""  # tail byte sequence not resolved yet — wait for more tokens
            else:
                text = full_text[printed_len:]
                printed_len = len(full_text)
        else:
            text = f"#{next_id}"

        yield next_id, text, is_last

        if is_eos:
            return
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)


def has_chat_template(loaded: LoadedModel) -> bool:
    return bool(getattr(loaded.tokenizer, "chat_template", None))


def apply_chat_template(loaded: LoadedModel, messages: list[dict]) -> list[int]:
    """Wraps `messages` in this model's own trained chat format — via
    transformers' real Jinja implementation (tokenizer_config.json's
    chat_template), not a hand-rolled approximation — and tokenizes the
    result into plain token ids, ready to feed generate_tokens/run_forward
    exactly like any other id list. Raises ValueError if this tokenizer has
    no chat_template at all; callers should check has_chat_template first
    (surfaced to the frontend so it can offer "raw prompt" as the only
    option for a model with no chat format, e.g. a base/completion model).
    """
    if not has_chat_template(loaded):
        raise ValueError("This model's tokenizer has no chat_template.")
    encoded = loaded.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    # Depending on the transformers version, this returns either a plain
    # list[int] or a dict-like BatchEncoding with an "input_ids" key.
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return list(input_ids)

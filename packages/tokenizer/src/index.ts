import type { ModelSource } from "@aperture/model-ir";
import { fetchJson, hfResolveUrl, readLocalJson } from "@aperture/hf-client";
import { getApiBase } from "@aperture/api-client";
import { bpeMerge } from "./bpe.js";
import { gpt2ByteDecode, gpt2Pretokenize, resolveByteLevel, type PreTokenizerSpec } from "./gpt2Pretokenize.js";
import { normalizerHasPrepend, normalizerRequiresNFC, spBpeDecodePieces, spBpePretokenize, type NormalizerSpec } from "./llamaPretokenize.js";

export interface EncodeResult {
  ids: number[];
  /** raw vocab strings — mostly useful for debugging the tokenizer itself */
  tokens: string[];
  /** human-readable per-token text, suitable for UI chips (spaces/newlines restored) */
  displayTokens: string[];
}

export interface Tokenizer {
  encode(text: string): EncodeResult;
  decode(ids: number[]): string;
  decodeToken(id: number): string;
}

/**
 * tokenizer.json's `post_processor` is what actually injects special
 * tokens around the encoded sequence — for the SentencePiece-style
 * tokenizers (Llama/Mistral/Gemma) this is a TemplateProcessing step that
 * always prepends `<bos>`, not something `add_bos_token` in
 * tokenizer_config.json controls (that field is often absent/null even
 * when tokenizer.json's own post_processor still adds it unconditionally,
 * e.g. Gemma). A byte-level tokenizer (GPT-2/Qwen) instead gets a plain
 * `ByteLevel` post_processor, which is a no-op here — it only concerns
 * offset trimming, not special tokens — so only `TemplateProcessing` is
 * handled below; every other type (including `ByteLevel` and `None`)
 * leaves encode() unchanged, matching today's behavior for those models.
 *
 * Skipping this entirely (as this file did before) is silently wrong for
 * any model that relies on it: confirmed against a real forward pass
 * (gemma-4-E4B, transformers 5.16.1) that "The cat sat on the" predicts
 * " mat" at 63.6% with `<bos>` correctly prepended, vs. the nonsensical
 * " cat" at 47% (with `<eos>` a close second) when it's left out — i.e.
 * exactly the "wrong top prediction" / "generate produces nothing" bug
 * this was causing.
 */
type TemplateStep = { SpecialToken: { id: string; type_id: number } } | { Sequence: { id: string; type_id: number } };

interface TemplateProcessingSpec {
  type: "TemplateProcessing";
  single: TemplateStep[];
  special_tokens: Record<string, { id: string; ids: number[]; tokens: string[] }>;
}

interface RawTokenizerJson {
  model: {
    type: string;
    vocab: Record<string, number>;
    merges: Array<string | [string, string]>;
  };
  pre_tokenizer?: PreTokenizerSpec | null;
  normalizer?: NormalizerSpec | null;
  post_processor?: { type: string } | TemplateProcessingSpec | null;
}

/** Applies a `single`-sequence TemplateProcessing template around one already-encoded body — the only shape this app's callers ever need (one prompt, never a sentence pair). */
function applyTemplate(template: TemplateProcessingSpec, bodyIds: number[], bodyTokens: string[]): { ids: number[]; tokens: string[] } {
  const ids: number[] = [];
  const tokens: string[] = [];
  for (const step of template.single) {
    if ("SpecialToken" in step) {
      const spec = template.special_tokens[step.SpecialToken.id];
      ids.push(spec.ids[0]);
      tokens.push(spec.tokens[0]);
    } else {
      ids.push(...bodyIds);
      tokens.push(...bodyTokens);
    }
  }
  return { ids, tokens };
}

/**
 * Loads a Hugging Face "fast tokenizer" (tokenizer.json) and returns a BPE
 * encoder/decoder. Detects which pre-tokenization scheme it needs from the
 * file itself — GPT-2's flat ByteLevel, Qwen's Sequence-of-[custom regex
 * Split, ByteLevel] (same byte-level family, different word-splitting
 * rules and an NFC normalizer), or the SentencePiece-style scheme Llama,
 * Mistral, and Gemma use — rather than assuming any one shape. Every case
 * here was added because a real tokenizer.json didn't match the simpler
 * assumption that came before it (see resolveByteLevel's and
 * normalizerHasPrepend's doc comments for the specific ones).
 */
export async function loadTokenizer(source: ModelSource): Promise<Tokenizer> {
  const raw = await loadRawTokenizerJson(source);
  return buildTokenizer(raw);
}

async function loadRawTokenizerJson(source: ModelSource): Promise<RawTokenizerJson> {
  if (source.kind === "local") return readLocalJson<RawTokenizerJson>(source, "tokenizer.json");
  if (source.kind === "huggingface") return fetchJson<RawTokenizerJson>(hfResolveUrl(source, "tokenizer.json"));
  // A backend-sourced model's tokenizer.json is a real file sitting in
  // data/models/<id>/ — served statically (see apps/api's main.py) so this
  // already-correct BPE implementation (handles GPT-2 byte-level, Qwen's
  // Sequence pretokenizer, and SentencePiece-style Llama/Mistral/Gemma
  // schemes) can run against it directly, rather than the backend
  // reimplementing tokenization in Python.
  const res = await fetch(`${getApiBase()}/data/models/${encodeURIComponent(source.modelId)}/tokenizer.json`);
  if (!res.ok) throw new Error(`No tokenizer.json for backend model "${source.modelId}" (${res.status} ${res.statusText}).`);
  return res.json() as Promise<RawTokenizerJson>;
}

/** Pure: turns an already-fetched tokenizer.json into a BPE encoder/decoder — see loadTokenizer for how the JSON gets here for each ModelSource kind. */
function buildTokenizer(raw: RawTokenizerJson): Tokenizer {
  if (raw.model.type !== "BPE") {
    throw new Error(`Unsupported tokenizer model type: "${raw.model.type}" (only BPE fast tokenizers are supported)`);
  }

  const vocab = new Map<string, number>(Object.entries(raw.model.vocab));
  const idToToken = new Map<number, string>([...vocab.entries()].map(([token, id]) => [id, token]));

  const mergeRank = new Map<string, number>();
  raw.model.merges.forEach((m, i) => {
    const key = Array.isArray(m) ? `${m[0]} ${m[1]}` : m;
    mergeRank.set(key, i);
  });

  const { isByteLevel, splitSteps } = resolveByteLevel(raw.pre_tokenizer);
  const needsNFC = isByteLevel && normalizerRequiresNFC(raw.normalizer);
  const stripsArtificialLeadingSpace = !isByteLevel && normalizerHasPrepend(raw.normalizer);
  const template = raw.post_processor?.type === "TemplateProcessing" ? (raw.post_processor as TemplateProcessingSpec) : null;

  function encode(text: string): EncodeResult {
    const normalized = needsNFC ? text.normalize("NFC") : text;
    const wordSymbolGroups = isByteLevel
      ? gpt2Pretokenize(normalized, splitSteps ?? undefined)
      : [spBpePretokenize(text, vocab, raw.normalizer)];

    const bodyIds: number[] = [];
    const bodyTokens: string[] = [];
    for (const symbols of wordSymbolGroups) {
      for (const piece of bpeMerge(symbols, mergeRank)) {
        const id = vocab.get(piece);
        if (id === undefined) continue; // shouldn't happen for well-formed BPE output
        bodyIds.push(id);
        bodyTokens.push(piece);
      }
    }

    const { ids, tokens } = template ? applyTemplate(template, bodyIds, bodyTokens) : { ids: bodyIds, tokens: bodyTokens };

    // per-token display pieces keep their natural leading space (if any) —
    // only the *whole-sequence* decode() strips the single artificial one
    // that a Prepend normalizer step adds at the very start. A prepended
    // special token (e.g. "<bos>") round-trips through this unchanged —
    // neither gpt2ByteDecode nor spBpeDecodePieces touch a piece with no
    // byte-fallback/space-marker syntax of their own.
    const displayTokens = tokens.map((t) => (isByteLevel ? gpt2ByteDecode(t) : spBpeDecodePieces([t])));
    return { ids, tokens, displayTokens };
  }

  function decode(ids: number[]): string {
    const pieces = ids.map((id) => idToToken.get(id) ?? "");
    if (isByteLevel) return pieces.map(gpt2ByteDecode).join("");
    const text = spBpeDecodePieces(pieces);
    return stripsArtificialLeadingSpace && text.startsWith(" ") ? text.slice(1) : text;
  }

  function decodeToken(id: number): string {
    const piece = idToToken.get(id) ?? "";
    return isByteLevel ? gpt2ByteDecode(piece) : spBpeDecodePieces([piece]);
  }

  return { encode, decode, decodeToken };
}

// The SentencePiece-derived BPE scheme used by Llama, Mistral, and Gemma's
// tokenizer.json: normalize (varies per model — see below), then feed the
// WHOLE normalized string as one sequence of individual Unicode codepoints
// into BPE — there is no GPT-2-style word-splitting regex. A codepoint with
// no single-character vocab entry falls back to its raw UTF-8 bytes, each
// shown as a "<0xXX>" token.
//
// The normalizer is NOT assumed to be "prepend ▁, then replace spaces with
// ▁" — that's Llama's specific normalizer sequence, but it isn't universal:
// Gemma's tokenizer.json has only the Replace step, no Prepend. Reading the
// actual normalizer spec out of tokenizer.json (rather than hardcoding
// Llama's shape) is what makes this correct for all three.

const SPACE_MARKER = "▁"; // U+2581

export interface NormalizerSpec {
  type: string;
  prepend?: string;
  pattern?: { String?: string };
  content?: string;
  normalizers?: NormalizerSpec[];
}

export function applyNormalizer(text: string, spec: NormalizerSpec | null | undefined): string {
  if (!spec) return text;
  switch (spec.type) {
    case "Sequence":
      return (spec.normalizers ?? []).reduce((t, step) => applyNormalizer(t, step), text);
    case "Prepend":
      return (spec.prepend ?? "") + text;
    case "Replace":
      return spec.pattern?.String !== undefined ? text.replaceAll(spec.pattern.String, spec.content ?? "") : text;
    default:
      return text;
  }
}

export function normalizerHasPrepend(spec: NormalizerSpec | null | undefined): boolean {
  if (!spec) return false;
  if (spec.type === "Prepend") return true;
  if (spec.type === "Sequence") return (spec.normalizers ?? []).some(normalizerHasPrepend);
  return false;
}

/** Qwen's tokenizer.json (and others) apply Unicode NFC normalization before pre-tokenizing — a no-op for plain ASCII prompts, but real for accented/composed characters. */
export function normalizerRequiresNFC(spec: NormalizerSpec | null | undefined): boolean {
  if (!spec) return false;
  if (spec.type === "NFC") return true;
  if (spec.type === "Sequence") return (spec.normalizers ?? []).some(normalizerRequiresNFC);
  return false;
}

/** One "word" (the whole normalized string) as BPE-ready initial symbols, with byte fallback applied per-codepoint. */
export function spBpePretokenize(text: string, vocab: Map<string, number>, normalizerSpec: NormalizerSpec | null | undefined): string[] {
  const normalized = applyNormalizer(text, normalizerSpec);
  const symbols: string[] = [];
  for (const ch of normalized) {
    if (vocab.has(ch)) {
      symbols.push(ch);
    } else {
      for (const byte of new TextEncoder().encode(ch)) {
        symbols.push(`<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`);
      }
    }
  }
  return symbols;
}

/** Reverses "▁" back to spaces and fuses consecutive byte-fallback tokens back into UTF-8 text. Does not strip anything — callers decide whether a leading space is an artifact of Prepend or real text. */
export function spBpeDecodePieces(pieces: string[]): string {
  let out = "";
  let byteBuf: number[] = [];
  const flushBytes = () => {
    if (byteBuf.length) {
      out += new TextDecoder("utf-8", { fatal: false }).decode(Uint8Array.from(byteBuf));
      byteBuf = [];
    }
  };
  for (const piece of pieces) {
    const byteMatch = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
    if (byteMatch) {
      byteBuf.push(parseInt(byteMatch[1], 16));
      continue;
    }
    flushBytes();
    out += piece.replaceAll(SPACE_MARKER, " ");
  }
  flushBytes();
  return out;
}

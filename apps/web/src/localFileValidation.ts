import { formatBytes } from "./format.js";

export interface FileCheck {
  ok: boolean;
  error?: string;
  warning?: string;
}

const TEXT_SNIFF_BYTES = 4096;
/** config.json/tokenizer.json are normally a few KB to a few hundred KB — a file many times that size claiming to be one is much more likely a misclick (e.g. picking the weights file twice) than a real config. */
const CONFIG_MAX_BYTES = 5 * 1024 * 1024;
/** Not a hard limit — the browser can technically hold more — just the point past which loading everything into memory as one buffer is likely to be slow or risk running out of memory. */
const WEIGHTS_WARN_BYTES = 300 * 1024 * 1024;

/** Reads only a small prefix and checks it decodes as clean UTF-8 with no embedded NUL bytes — the cheap, reliable way to tell "this is actually text/JSON" from "this is binary content someone renamed to .json", without reading a potentially huge file in full just to reject it. */
async function looksLikeText(file: File): Promise<boolean> {
  const prefix = new Uint8Array(await file.slice(0, TEXT_SNIFF_BYTES).arrayBuffer());
  if (prefix.includes(0)) return false;
  try {
    new TextDecoder("utf-8", { fatal: true }).decode(prefix);
    return true;
  } catch {
    return false;
  }
}

/** config.json / tokenizer.json: must be small, valid UTF-8 text, and parse as JSON — catches a mislabeled binary file (e.g. a `.safetensors` renamed to `.json`) or a truncated/corrupt download before the app commits to loading it. */
export async function checkJsonFile(file: File): Promise<FileCheck> {
  if (file.size === 0) return { ok: false, error: "File is empty." };
  if (file.size > CONFIG_MAX_BYTES) {
    return { ok: false, error: `${formatBytes(file.size)} is too large for a config/tokenizer file — expected JSON text, typically a few KB. Check you picked the right file.` };
  }
  if (!(await looksLikeText(file))) {
    return { ok: false, error: "This doesn't look like a text/JSON file — its content looks binary." };
  }
  try {
    JSON.parse(await file.text());
  } catch (err) {
    return { ok: false, error: `Not valid JSON: ${err instanceof Error ? err.message : String(err)}` };
  }
  return { ok: true };
}

/** .safetensors: validated by actually reading its header (an 8-byte little-endian length prefix followed by that many bytes of JSON) rather than trusting the file extension — catches a mislabeled file immediately with a clear message instead of failing deep inside the model adapter later. Only the header-sized prefix is read, never the whole (possibly huge) file. */
export async function checkWeightsFile(file: File): Promise<FileCheck> {
  if (file.size < 8) return { ok: false, error: "File is too small to be a safetensors file." };
  const prefix = await file.slice(0, 8).arrayBuffer();
  const headerLength = Number(new DataView(prefix).getBigUint64(0, true));
  if (!Number.isFinite(headerLength) || headerLength <= 0 || headerLength > 100_000_000 || 8 + headerLength > file.size) {
    return { ok: false, error: "This doesn't look like a valid safetensors file (its header is malformed) — check you picked the right file." };
  }
  try {
    const headerBytes = await file.slice(8, 8 + headerLength).arrayBuffer();
    JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(headerBytes));
  } catch {
    return { ok: false, error: "This doesn't look like a valid safetensors file (its header isn't valid JSON)." };
  }
  const warning =
    file.size > WEIGHTS_WARN_BYTES
      ? `This file is ${formatBytes(file.size)} — loading a model this large in the browser may be slow or run out of memory.`
      : undefined;
  return { ok: true, warning };
}

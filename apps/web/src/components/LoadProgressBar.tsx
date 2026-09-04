import { formatBytes, formatCount } from "../format.js";

export interface UnifiedLoadProgress {
  /** Free-form phase key, same "generic, not an enum" philosophy as model-ir's LoadProgressEvent — looked up in PHASE_LABELS below, with the raw string as a fallback for any phase this component doesn't specifically know how to word. */
  phase: string;
  /** Progress within this phase, if known. */
  current?: number;
  total?: number;
  /** How to format current/total — "bytes" for a real network transfer (the HF download, where the backend forwards a real Content-Length), "count" for a materialization step measured in items (parameters materialized onto the GPU) rather than bytes. */
  unit: "bytes" | "count";
}

const PHASE_LABELS: Record<string, string> = {
  downloading: "Downloading from Hugging Face",
  // Not a literal "this exact file is now downloading" — huggingface_hub's
  // own progress hooks never expose one (see apps/api's downloads.py
  // _non_weight_bytes doc comment). Derived from a byte threshold instead:
  // once cumulative transfer bytes pass everything that isn't a weights
  // shard, whatever's still incoming almost certainly is one.
  downloading_config: "Downloading tokenizer & config files",
  downloading_weights: "Downloading model weights",
  loading_weights: "Loading weights onto GPU",
  building_graph: "Building architecture graph",
};

/**
 * One shared progress bar for both halves of "get a model ready" —
 * downloading it from Hugging Face into data/models/ (byte-accurate) and
 * then materializing it onto the GPU (count-accurate: one tick per real
 * parameter tensor `from_pretrained` loads, not a byte count). The two
 * phases are sequential and never simultaneous (a download always
 * finishes, then triggers the load), so ModelLoader renders exactly one
 * of these at a time rather than two separately-tracked progress blocks —
 * previously near-duplicate JSX/CSS for what is, from the user's point of
 * view, one continuous "getting your model ready" wait.
 */
export function LoadProgressBar({ progress }: { progress: UnifiedLoadProgress }) {
  const { phase, current, total, unit } = progress;
  const pct = total ? Math.min(100, ((current ?? 0) / total) * 100) : undefined;
  const format = unit === "bytes" ? formatBytes : formatCount;
  const label = PHASE_LABELS[phase] ?? phase;

  return (
    <div className="load-progress">
      <div className="load-progress-label">{label}</div>
      <div className={"load-progress-track" + (pct === undefined ? " indeterminate" : "")}>
        <div className="load-progress-fill" style={pct === undefined ? undefined : { width: `${pct}%` }} />
      </div>
      <span className="load-progress-status">{current !== undefined && total !== undefined ? `${format(current)} / ${format(total)}` : "Starting…"}</span>
    </div>
  );
}

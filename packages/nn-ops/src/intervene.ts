import type { Intervention } from "@aperture/model-ir";
import type { Matrix } from "./index.js";

/**
 * Applies whichever "zero" / "scale" / "replace" Interventions target
 * `nodeId` to its just-computed activation, returning the (possibly
 * edited) matrix that the forward pass should actually continue with. A
 * no-op fast path when nothing targets this node, which is the common
 * case for every node in every run that isn't specifically under
 * investigation.
 *
 * "zero_head" is handled separately by `applyHeadIntervention` below, at
 * the one point in a forward pass where per-head slicing is actually
 * meaningful — see its doc comment for why.
 */
export function applyInterventions(nodeId: string, m: Matrix, interventions: Intervention[] | undefined, headDim?: number): Matrix {
  if (!interventions || interventions.length === 0) return m;
  const relevant = interventions.filter((iv) => iv.nodeId === nodeId && iv.operation !== "zero_head");
  if (relevant.length === 0) return m;

  const out = m.map((row) => row.slice());
  for (const iv of relevant) {
    const rows = iv.tokenIndex != null ? [iv.tokenIndex] : out.map((_, i) => i);
    for (const r of rows) {
      if (r < 0 || r >= out.length) continue;
      switch (iv.operation) {
        case "zero":
          out[r] = out[r].map(() => 0);
          break;
        case "scale": {
          const s = iv.scale ?? 1;
          out[r] = out[r].map((v) => v * s);
          break;
        }
        case "replace": {
          const src = iv.replacementValue;
          if (!src || src.shape.length !== 2 || r >= src.shape[0]) break;
          const cols = src.shape[1];
          for (let c = 0; c < out[r].length && c < cols; c++) out[r][c] = src.data[r * cols + c];
          break;
        }
      }
    }
  }
  return out;
}

/**
 * Zeroes one attention head's columns in the *pre-output-projection*
 * concatenated-heads tensor — the only point in the computation where
 * "head 3" is still a well-defined slice. Once output_projection mixes the
 * concatenated heads back down to hidden_size, individual heads no longer
 * correspond to any column range, so this must run right after
 * `causalSelfAttention`, not at the attention node's normal record() point
 * (which reflects output_projection's result).
 */
export function applyHeadIntervention(nodeId: string, m: Matrix, interventions: Intervention[] | undefined, headDim: number): Matrix {
  if (!interventions || interventions.length === 0) return m;
  const relevant = interventions.filter((iv) => iv.nodeId === nodeId && iv.operation === "zero_head" && iv.headIndex != null);
  if (relevant.length === 0) return m;

  const out = m.map((row) => row.slice());
  for (const iv of relevant) {
    const rows = iv.tokenIndex != null ? [iv.tokenIndex] : out.map((_, i) => i);
    const start = iv.headIndex! * headDim;
    const end = Math.min(out[0]?.length ?? 0, start + headDim);
    for (const r of rows) {
      if (r < 0 || r >= out.length) continue;
      for (let c = Math.max(0, start); c < end; c++) out[r][c] = 0;
    }
  }
  return out;
}

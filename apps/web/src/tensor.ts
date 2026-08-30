import type { ParameterRef, TensorSlice } from "@aperture/model-ir";

export function parameterKey(ref: ParameterRef): string {
  return `${ref.name}#${ref.slice?.ranges ? JSON.stringify(ref.slice.ranges) : "full"}`;
}

/**
 * A ParameterRef may already stand for a slice of its underlying tensor
 * (e.g. Q projection = a column slice of GPT-2's fused c_attn weight).
 * A "window" is a further slice requested by the user in the Tensor
 * Explorer (e.g. "show rows 0-63"), expressed in the ref's *logical*
 * (already-sliced) coordinate space. This composes the two into the single
 * slice the WeightProvider needs, in the underlying tensor's coordinates.
 */
export function composeSlice(ref: ParameterRef, window?: TensorSlice): TensorSlice {
  const ranges = ref.shape.map((dimSize, i) => {
    const base = ref.slice?.ranges?.[i] ?? { start: 0, end: dimSize };
    const w = window?.ranges?.[i];
    if (!w) return base;
    const start = base.start + Math.max(0, w.start);
    const end = Math.min(base.end, base.start + w.end);
    return { start, end: Math.max(start, end) };
  });
  return { ranges };
}

/** Default viewing window: cap each dim at `maxDim` so huge tensors don't get fully materialized. */
export function defaultWindow(logicalShape: number[], maxDim = 64): TensorSlice {
  return { ranges: logicalShape.map((dim) => ({ start: 0, end: Math.min(dim, maxDim) })) };
}

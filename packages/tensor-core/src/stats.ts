export interface TensorStats {
  count: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  zeros: number;
  sparsity: number;
  percentiles: { p1: number; p25: number; p50: number; p75: number; p99: number };
  histogram: { binStart: number; binEnd: number; count: number }[];
}

const EMPTY_PERCENTILES = { p1: 0, p25: 0, p50: 0, p75: 0, p99: 0 };

function computePercentiles(data: Float64Array): TensorStats["percentiles"] {
  const sorted = Float64Array.from(data).sort();
  const at = (p: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))];
  return { p1: at(0.01), p25: at(0.25), p50: at(0.5), p75: at(0.75), p99: at(0.99) };
}

export function computeStats(data: Float64Array, bins = 24): TensorStats {
  const count = data.length;
  if (count === 0) {
    return { count: 0, min: 0, max: 0, mean: 0, std: 0, zeros: 0, sparsity: 0, percentiles: EMPTY_PERCENTILES, histogram: [] };
  }

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let zeros = 0;
  for (let i = 0; i < count; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    if (v === 0) zeros++;
  }
  const mean = sum / count;

  let sqDiff = 0;
  for (let i = 0; i < count; i++) {
    const d = data[i] - mean;
    sqDiff += d * d;
  }
  const std = Math.sqrt(sqDiff / count);

  const histogram: TensorStats["histogram"] = [];
  const span = max - min || 1;
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < count; i++) {
    let b = Math.floor(((data[i] - min) / span) * bins);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    counts[b]++;
  }
  for (let b = 0; b < bins; b++) {
    histogram.push({ binStart: min + (span * b) / bins, binEnd: min + (span * (b + 1)) / bins, count: counts[b] });
  }

  return { count, min, max, mean, std, zeros, sparsity: zeros / count, percentiles: computePercentiles(data), histogram };
}

import { useEffect, useRef } from "react";

interface HeatmapProps {
  data: Float64Array;
  rows: number;
  cols: number;
  cellSize?: number;
}

// Browsers cap an individual canvas dimension well below what a real tensor
// axis can reach — an LM Head activation is [sequence_length, vocab], and a
// 150K+ token vocabulary is normal for several of this app's own presets
// (GLM-4, Qwen2/3, Gemma's tiny-random checkpoint even reaches 256K). Without
// a cap, `cols * cellSize` blows past every browser's max canvas dimension
// and the canvas silently renders nothing — no error, just a blank pane.
// Capping the OUTPUT grid (not just clamping cellSize toward zero) means a
// huge axis still produces a real, readable image instead of an unusably
// thin sliver.
const MAX_CELLS_PER_AXIS = 768;

function diverging(value: number, absMax: number): [number, number, number] {
  if (absMax === 0) return [255, 255, 255];
  const t = Math.max(-1, Math.min(1, value / absMax));
  if (t >= 0) {
    // white -> red
    const r = 255;
    const g = Math.round(255 * (1 - t));
    const b = Math.round(255 * (1 - t));
    return [r, g, b];
  }
  // white -> blue
  const s = -t;
  const r = Math.round(255 * (1 - s));
  const g = Math.round(255 * (1 - s));
  const b = 255;
  return [r, g, b];
}

export function Heatmap({ data, rows, cols, cellSize }: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const outRows = Math.min(rows, MAX_CELLS_PER_AXIS);
  const outCols = Math.min(cols, MAX_CELLS_PER_AXIS);
  const downsampled = outRows !== rows || outCols !== cols;
  const size = cellSize ?? Math.max(2, Math.min(24, Math.floor(480 / Math.max(outRows, outCols))));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = outCols * size;
    canvas.height = outRows * size;

    // One pass over the real data: track the global |value| max for the
    // color scale (as before) and, per output bucket, the single
    // largest-magnitude value that falls in it — a max-abs downsample
    // reads truthfully as "does this region contain anything notable?",
    // where an average would quietly wash out a lone outlier.
    const bucketVal = new Float64Array(outRows * outCols);
    const bucketAbs = new Float64Array(outRows * outCols).fill(-1);
    let absMax = 0;
    for (let r = 0; r < rows; r++) {
      const or_ = Math.min(outRows - 1, Math.floor((r * outRows) / rows));
      const rowBase = r * cols;
      for (let c = 0; c < cols; c++) {
        const v = data[rowBase + c];
        const a = Math.abs(v);
        if (a > absMax) absMax = a;
        const bIdx = or_ * outCols + Math.min(outCols - 1, Math.floor((c * outCols) / cols));
        if (a > bucketAbs[bIdx]) {
          bucketAbs[bIdx] = a;
          bucketVal[bIdx] = v;
        }
      }
    }

    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let or_ = 0; or_ < outRows; or_++) {
      for (let oc = 0; oc < outCols; oc++) {
        const v = bucketVal[or_ * outCols + oc];
        const [red, green, blue] = diverging(v, absMax);
        for (let py = 0; py < size; py++) {
          for (let px = 0; px < size; px++) {
            const x = oc * size + px;
            const y = or_ * size + py;
            const idx = (y * canvas.width + x) * 4;
            img.data[idx] = red;
            img.data[idx + 1] = green;
            img.data[idx + 2] = blue;
            img.data[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data, rows, cols, outRows, outCols, size]);

  return (
    <div className="heatmap-wrap">
      <canvas ref={canvasRef} className="heatmap-canvas" title="Diverging color scale: blue = negative, red = positive, white = ~0" />
      {downsampled && (
        <div className="heatmap-downsample-note">
          Downsampled from {rows.toLocaleString()} × {cols.toLocaleString()} to {outRows} × {outCols} — each cell shows the largest-magnitude value in its region.
        </div>
      )}
    </div>
  );
}

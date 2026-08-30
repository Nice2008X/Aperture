import type { Model } from "@aperture/model-ir";
import { totalParameterCount, totalParameterBytes } from "@aperture/model-ir";
import { formatBytes, formatCount } from "../format.js";

export function ModelInfoBar({ model }: { model: Model }) {
  const params = totalParameterCount(model);
  const bytes = totalParameterBytes(model);
  const dtype = Object.values(model.nodes).find((n) => n.parameters.length > 0)?.parameters[0]?.dtype ?? "—";

  const stats: [string, string][] = [
    ["Architecture", model.architecture],
    ["Parameters", formatCount(params)],
    ["Layers", String(model.config.numLayers)],
    ["Attention heads", String(model.config.numHeads)],
    ["Hidden size", String(model.config.hiddenSize)],
    ["Intermediate size", String(model.config.intermediateSize)],
    ["Vocabulary", model.config.vocabSize.toLocaleString()],
    ["Context length", model.config.contextLength.toLocaleString()],
    ["Dtype", dtype],
    ["Weights (on GPU backend)", formatBytes(bytes)],
  ];

  return (
    <div className="model-info-bar">
      <div className="model-info-name">{model.name}</div>
      <div className="model-info-stats">
        {stats.map(([label, value]) => (
          <div key={label} className="model-info-stat">
            <div className="model-info-stat-value">{value}</div>
            <div className="model-info-stat-label">{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

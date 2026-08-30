import type { Model } from "@aperture/model-ir";
import { totalParameterCount, totalParameterBytes } from "@aperture/model-ir";
import { formatBytes, formatCount } from "../format.js";
import { useTranslation } from "./LanguageContext.js";

/**
 * Three loosely-themed clusters instead of one flat run of 10 same-weight
 * stats — "what model" / "what shape" / "what it costs to run" reads faster
 * at a glance than a row where params and dtype carry equal visual weight.
 * Nothing here is dropped, just grouped; each group renders as one
 * `·`-joined line rather than its own boxes.
 */
export function ModelInfoBar({ model }: { model: Model }) {
  const { t } = useTranslation();
  const params = totalParameterCount(model);
  const bytes = totalParameterBytes(model);
  const dtype = Object.values(model.nodes).find((n) => n.parameters.length > 0)?.parameters[0]?.dtype ?? "—";

  const groups: [string, string][] = [
    [
      t("modelInfo.groupModel"),
      t("modelInfo.modelLine")
        .replace("{arch}", model.architecture)
        .replace("{params}", formatCount(params))
        .replace("{layers}", String(model.config.numLayers)),
    ],
    [
      t("modelInfo.groupArchitecture"),
      t("modelInfo.architectureLine")
        .replace("{heads}", String(model.config.numHeads))
        .replace("{hidden}", String(model.config.hiddenSize))
        .replace("{mlp}", String(model.config.intermediateSize))
        .replace("{vocab}", model.config.vocabSize.toLocaleString()),
    ],
    [
      t("modelInfo.groupRuntime"),
      t("modelInfo.runtimeLine")
        .replace("{dtype}", dtype)
        .replace("{bytes}", formatBytes(bytes))
        .replace("{context}", model.config.contextLength.toLocaleString()),
    ],
  ];

  return (
    <div className="model-info-bar">
      <div className="model-info-name">{model.name}</div>
      {/* Only the last (Runtime) group is allowed to shrink+ellipsize — see
          .model-info-group:last-child — so a narrow window (or a long model
          name eating into the available width) degrades to "BF16 · 32.8 MB…"
          instead of the row running underneath .top-right-controls. */}
      <div className="model-info-groups">
        {groups.map(([label, value]) => (
          <div key={label} className="model-info-group">
            <div className="model-info-group-label">{label}</div>
            <div className="model-info-group-value">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

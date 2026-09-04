import { useEffect } from "react";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  open: boolean;
  modelType: string | null;
  busy?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

/**
 * Warns before downloading a repo whose config.json `model_type` isn't
 * registered in transformers' AutoModelForCausalLM (apps/api's
 * downloads.py check_model_support) — the download itself would likely
 * succeed, but loading it afterwards would fail (see PLAN.md §11's
 * `needle2` case). Advisory only: "Download anyway" is a real, supported
 * choice for a model this check just hasn't seen before, not a dead end.
 */
export function UnsupportedModelDialog({ open, modelType, busy, onCancel, onConfirm }: Props) {
  const { t } = useTranslation();

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onCancel();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      className="save-model-dialog-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="save-model-dialog" role="dialog" aria-modal="true" aria-label={t("loader.unsupportedModelTitle")}>
        <div className="save-model-dialog-header">
          <span>{t("loader.unsupportedModelTitle")}</span>
          <button className="save-model-dialog-close" onClick={onCancel} aria-label={t("loader.close")} title={t("loader.close")}>
            ×
          </button>
        </div>
        <p className="save-model-dialog-desc">{t("loader.unsupportedModelDesc")}</p>
        <div className="unknown-model-dialog-detected">
          <span className="unknown-model-dialog-detected-label">{t("loader.unsupportedModelLabel")}</span>
          <span className="unknown-model-dialog-detected-value">{modelType ?? t("loader.unsupportedModelUnknown")}</span>
        </div>
        <div className="save-model-dialog-actions">
          <button className="save-model-dialog-cancel" disabled={busy} onClick={onCancel}>
            {t("loader.cancel")}
          </button>
          <button className="delete-model-dialog-confirm" disabled={busy} onClick={onConfirm}>
            {t("loader.downloadAnyway")}
          </button>
        </div>
      </div>
    </div>
  );
}

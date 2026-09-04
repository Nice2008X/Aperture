import { useEffect } from "react";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  open: boolean;
  modelName?: string;
  busy?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

/**
 * Replaces a plain window.confirm() for the catalog's "delete this
 * checkpoint" action — that's destructive (removes files from disk,
 * unrecoverable) and deserved a dialog that actually names the model
 * instead of a browser-chrome popup easy to blitz through. Defaults to
 * Cancel in every sense that matters, same as the reference dialogs this is
 * modeled on: no button auto-focused into looking primary, Escape/backdrop
 * click both cancel, and nothing is deleted unless "Delete" is clicked
 * deliberately.
 */
export function DeleteModelDialog({ open, modelName, busy, onCancel, onConfirm }: Props) {
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
      <div className="save-model-dialog" role="dialog" aria-modal="true" aria-label={t("loader.deleteModelTitle")}>
        <div className="save-model-dialog-header">
          <span>{t("loader.deleteModelTitle")}</span>
          <button className="save-model-dialog-close" onClick={onCancel} aria-label={t("loader.close")} title={t("loader.close")}>
            ×
          </button>
        </div>
        <p className="save-model-dialog-desc">{t("loader.deleteModelDesc")}</p>
        <div className="unknown-model-dialog-detected">
          <span className="unknown-model-dialog-detected-label">{t("loader.deleteModelLabel")}</span>
          <span className="unknown-model-dialog-detected-value">{modelName}</span>
        </div>
        <div className="save-model-dialog-actions">
          <button className="save-model-dialog-cancel" disabled={busy} onClick={onCancel}>
            {t("loader.cancel")}
          </button>
          <button className="delete-model-dialog-confirm" disabled={busy} onClick={onConfirm}>
            {busy ? t("loader.deleting") : t("loader.delete")}
          </button>
        </div>
      </div>
    </div>
  );
}

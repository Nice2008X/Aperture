import { useEffect, useRef } from "react";
import { formatBytes } from "../format.js";
import { useTranslation } from "./LanguageContext.js";

export interface SaveModelFile {
  filename: string;
  bytes: number;
}

interface Props {
  open: boolean;
  files: SaveModelFile[];
  onCancel: () => void;
  onConfirm: () => void;
}

/** Confirmation modal for "Save model" — names every file about to be downloaded (and its size) before triggering the browser's Save dialogs, rather than silently kicking off up to three downloads on a single click. */
export function SaveModelDialog({ open, files, onCancel, onConfirm }: Props) {
  const { t } = useTranslation();
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onCancel();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onCancel]);

  if (!open) return null;

  const totalBytes = files.reduce((sum, f) => sum + f.bytes, 0);

  return (
    <div
      className="save-model-dialog-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="save-model-dialog" ref={dialogRef} role="dialog" aria-modal="true" aria-label={t("app.saveModelDialogTitle")}>
        <div className="save-model-dialog-header">
          <span>{t("app.saveModelDialogTitle")}</span>
          <button className="save-model-dialog-close" onClick={onCancel} aria-label={t("loader.close")} title={t("loader.close")}>
            ×
          </button>
        </div>
        <p className="save-model-dialog-desc">{t("app.saveModelDialogDesc")}</p>
        <ul className="save-model-dialog-files">
          {files.map((f) => (
            <li key={f.filename}>
              <span className="save-model-dialog-filename">{f.filename}</span>
              <span className="save-model-dialog-filesize">{formatBytes(f.bytes)}</span>
            </li>
          ))}
        </ul>
        {files.length > 1 && (
          <div className="save-model-dialog-total">
            <span>{t("app.saveModelDialogTotal")}</span>
            <span>{formatBytes(totalBytes)}</span>
          </div>
        )}
        <div className="save-model-dialog-actions">
          <button className="save-model-dialog-cancel" onClick={onCancel}>
            {t("app.cancel")}
          </button>
          <button className="save-model-dialog-confirm" onClick={onConfirm}>
            {t("app.saveModelDialogConfirm")}
          </button>
        </div>
      </div>
    </div>
  );
}
